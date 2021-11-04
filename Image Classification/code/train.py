"""
Author: Colin Wang, Jerry Chan, Bingqi Zhou
"""
from torch.utils.data import DataLoader, Subset
from torch.optim import lr_scheduler
from tqdm import tqdm
from util import *
import sys
import torch
import numpy as np
import model
from itertools import chain

class trainer():
    def __init__(self, loader, test_loader, model_name, \
        batch_size=20, lr=1e-4, weight_mode='uniform', \
        device='cuda', weight_decay=0, eps=1e-08, amsgrad=False, \
        num_of_classes=20, model_path = 'model/param.pth', freeze = True, save_mode='loss'):
        """
        Initialization: Set training info and hyperparameters
        """
        assert model_name in ['baseline', 'custom', 'resnet18', 'vgg16']
        self.freeze = freeze
        self.model_name = model_name
        self.save_mode =  'loss'
        self.loader = loader
        self.test_loader = test_loader
        self.used_valid_indices = []
        self.info = {'num_of_classes': num_of_classes, 'device': device, \
                     'save_path': model_path, 'model_name': model_name}
        self.hparam = {'batch_size': batch_size, 'lr': lr, 'weight_mode': weight_mode,\
                        'weight_decay': weight_decay, 'eps': eps, 'amsgrad': amsgrad}
        self.stat = {'train_acc':[], 'train_loss':[], 'val_acc':[], \
                    'val_loss':[], 'test_acc':[], 'test_loss':[]}      

    def initialize_model(self):
        """
        initialize model, optimizer, and loss function
        """
        # build model
        if self.info['model_name'] == 'baseline':
            self.model = model.baseline_Net(classes = self.info['num_of_classes'])
            self.model.to(self.info['device'])
            self.model.initialize_weights(self.hparam['weight_mode'])
        elif self.info['model_name'] == 'custom':
            self.model = model.custom_Net(classes = self.info['num_of_classes'])
            self.model.to(self.info['device'])
            self.model.initialize_weights(self.hparam['weight_mode'])
        # if the model is vgg16 or resnet18, we freeze the pretrained layers
        elif self.info['model_name'] == 'vgg16':
            self.model = model.vgg(self.info['num_of_classes'])
            self.model.to(self.info['device'])
            if self.freeze: 
                self.model.freeze()
                self.model.initialize_weights(self.hparam['weight_mode'])
        elif self.info['model_name'] == 'resnet18':
            self.model = model.resnet(self.info['num_of_classes'])
            self.model.to(self.info['device'])
            if self.freeze: 
                self.model.freeze()
                self.model.initialize_weights(self.hparam['weight_mode'])
        else: pass        
        
        # apply optimizer
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),\
            lr=self.hparam['lr'], eps=self.hparam['eps'], weight_decay=self.hparam['weight_decay'])
        
        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss()
                                          
    def split_data(self):
        """
        validation split
        """
        # get training and validation indices
        train_indices, valid_indices = get_indices(self.used_valid_indices, 600, 30, 3)
        
        # get data
        dataset_train = Subset(self.loader.dataset, train_indices)
        dataset_valid = Subset(self.loader.dataset, valid_indices)
        
        # packed data into DataLoader
        self.train_loader, self.val_loader = DataLoader(\
            dataset=dataset_train, shuffle=True, batch_size=self.hparam['batch_size']),\
                                             DataLoader(\
            dataset=dataset_valid, shuffle=True, batch_size=self.hparam['batch_size'])
        self.used_valid_indices.extend(valid_indices)

    def batch_learn_core(self, inputs, labels, train=False):
        """
        run a single batch of learning process and collect stats
        """
        
        # forward prop
        inputs = inputs.type(torch.cuda.FloatTensor)
        labels = labels.type(torch.cuda.LongTensor)                    
        outputs = self.model(inputs)
        
        # collect stats
        acc = find_accuracy(outputs, labels)
        loss = self.criterion(outputs, labels)
        
        # back prop
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return acc, loss

    def epoch_learn_core(self, loader, train=False):
        """
        run a single epoch of learning process and collect stats
        """
        batch_loss, batch_acc = [], []
        
        # iterate through the dataset
        for inputs, labels in loader:
            # pass each batch into the model
            acc, loss = self.batch_learn_core(inputs, labels, train=train)
            batch_acc.append(acc)
            batch_loss.append(loss.item())
        epoch_acc = np.mean(batch_acc)
        epoch_loss = np.mean(batch_loss)
        return epoch_acc, epoch_loss

    def train(self, epochs=100, K=2, validate_option=True):
        """
        K-fold validation
        """
        self.reset_stats()
        # K-fold
        for k in range(K):
            # data/model prep
            self.split_data()
            self.add_sublists()
            self.initialize_model()
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=epochs//4, gamma=0.5)
            for epoch in tqdm(range(epochs), file=sys.stdout):
                # train model on training set
                self.model.train()
                epoch_acc, epoch_loss = self.epoch_learn_core(\
                    self.train_loader, train=True)
                # print('train accuracy: ',end = str(epoch_acc))
                self.stat['train_acc'][k].append(epoch_acc)
                self.stat['train_loss'][k].append(epoch_loss)
                self.scheduler.step()
                # validate model on validation set
                if validate_option: self.validate(k)
            
            # test model on testing test
            self.predict()
        

    def validate(self, k):
        """
        Validate the model. Save the model's parameter if it gets a higher accuarcy
        than the previos models
        """
        # testing
        self.model.eval()
        epoch_acc, epoch_loss = self.epoch_learn_core(\
                    self.val_loader, train=False)
        
        # save if it gets a higher accuarcy
        if self.save_mode == 'loss':
            if len(self.stat['val_loss'][k]) == 0 or epoch_loss <= min(self.stat['val_loss'][k]):
                torch.save(self.model.state_dict(), self.info['save_path'])
        elif self.save_mode == 'acc':
            if len(self.stat['val_acc'][k]) == 0 or epoch_acc >= max(self.stat['val_acc'][k]):
                torch.save(self.model.state_dict(), self.info['save_path'])
        # record stats
        # print('validation accuracy: ',end = str(epoch_acc))
        self.stat['val_acc'][k].append(epoch_acc)
        self.stat['val_loss'][k].append(epoch_loss)

    def predict(self, path=''):
        """
        test the model with test set
        """
        # load model
        self.load_best_model(path)
        self.model.eval()
        
        # test model
        epoch_acc, epoch_loss = self.epoch_learn_core(\
                    self.test_loader, train=False)
        self.stat['test_acc'].append(epoch_acc)
        self.stat['test_loss'].append(epoch_loss)

    def get_training_stats(self):
        """ training stats getter """
        return tuple([np.mean(lst, axis = 0) for lst in \
        [self.stat['train_acc'], self.stat['train_loss'], \
         self.stat['val_acc'], self.stat['val_loss']]])

    def get_test_stats(self):
        """ testing stats getter """
        return np.mean(self.stat['test_acc']), np.mean(self.stat['test_loss'])

    def add_sublists(self):
        """ training stats getter """
        for lst in [self.stat['train_acc'], self.stat['train_loss'], \
            self.stat['val_acc'], self.stat['val_loss']]: lst.append([])

    def reset_stats(self):
        """ reset testing / training stats """
        for lst in [self.stat['train_acc'], self.stat['train_loss'], self.stat['val_acc'], \
            self.stat['val_loss'], self.stat['test_acc'], self.stat['test_loss']]: lst = []

    def load_best_model(self, path=''):
        """ load the best preforming model """
        if path == '':
            self.model.load_state_dict(torch.load(self.info['save_path']))
        else: self.model.load_state_dict(torch.load(path))

    def weight_maps(self, name = None):
        """ 
        plot weight maps for the first CNN layer 
        """
        assert self.model_name in ['resnet18', 'vgg16']
        if name == None:
            name = self.model_name + ' first layer weight maps'
        if self.model_name == 'vgg16':
            plot_kernels(self.model.pretrain.features[0].weight, name)
        elif self.model_name == 'resnet18':
            plot_kernels(self.model.pretrain.conv1.weight, name)
            
          