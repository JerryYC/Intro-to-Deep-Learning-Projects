################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
# Finely implemented and refined by Colin Wang, Jerry Chan
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

from copy import deepcopy
from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
from pycocotools.coco import COCO
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.tokenize import word_tokenize

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, 
# checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment 
# and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):

        config_data = read_file_in_dir('./', name + '.json')
        self.config_data = config_data
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)
        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader,\
        self.__test_loader = get_datasets(self.config_data)
        self.config_data['model']['vocab_size'] = len(self.__vocab)
        
        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__device = config_data['model']['device']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None

        # Init Model
        self.__model = get_model(config_data)

        # Configure loss function, optimizer, scheduler
        self.__criterion = torch.nn.CrossEntropyLoss()
        if config_data['model']['finetune']:
            self.__optimizer = torch.optim.Adam( 
                # encoder
                [{'params':self.__model.resnet.parameters(), 
                  'weight_decay': self.config_data['experiment']['weight_decay'],
                  'lr':self.config_data['experiment']['learning_rate']/10},
                # decoder
                 {'params':[ p[1]  for p in self.__model.named_parameters() \
                 					if not 'resnet' in p[0]],
                  'weight_decay': self.config_data['experiment']['weight_decay'],
                  'lr':self.config_data['experiment']['learning_rate']}
                ],
                eps=self.config_data['experiment']['eps'],
                amsgrad=self.config_data['experiment']['amsgrad'])
            
        else:
            self.__optimizer = torch.optim.Adam(\
            	[p for p in self.__model.parameters() if p.requires_grad], 
            lr=self.config_data['experiment']['learning_rate'], 
            eps=self.config_data['experiment']['eps'], 
            weight_decay=self.config_data['experiment']['weight_decay'],
            amsgrad=self.config_data['experiment']['amsgrad'])
        
        self.__scheduler = torch.optim.lr_scheduler.StepLR(self.__optimizer, 
        		self.config_data['experiment']['step_size'],
                gamma=self.config_data['experiment']['scheduler_gamma'])
        
        # configure early stop
        self.early_stop = self.config_data['experiment']['early_stop']
        self.patience = self.config_data['experiment']['patience']

        # initialize model
        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists 
    # to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)
        if os.path.exists(self.__experiment_dir):
        	# load stats
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 
            	'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 
            	'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            # load model states
            state_dict = torch.load(os.path.join(self.__experiment_dir, 
            	'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])
            self.__scheduler.load_state_dict(state_dict['scheduler'])
        else:
            os.makedirs(self.__experiment_dir)

    # move the model to the correct device
    def __init_model(self):
        self.__model.to(self.__device)
        self.__criterion.to(self.__device)

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        
        # show an image example in each epoch
        image = self.__train_loader.dataset[26][0]
        image = np.einsum("ijk->jki",image)
        min_val = image.min()
        max_val = image.max()
        image = (image - min_val)
        image /= (max_val - min_val)
        plt.figure()
        plt.imshow(image) 
        plt.show()
        
        # loop over the dataset multiple times
        for epoch in range(start_epoch, self.__epochs):  
            print("Epoch {} - Start with LR {}".format(epoch+1, 
            	self.__optimizer.param_groups[0]['lr']))
            start_time = datetime.now()
            self.__current_epoch = epoch

            # train, validate, update
            train_loss = self.__train()
            val_loss = self.__val()
            self.__scheduler.step()

            # update and save the best model
            if len(self.__val_losses) == 0 or val_loss < min(self.__val_losses):
                print("Model Saved")
                self.__best_model = deepcopy(self.__model)
                self.__save_model('best')

            # trigger early stop if condition satisfied
            if self.early_stop and len(self.__val_losses) > self.patience and \
            val_loss > max(self.__val_losses[-self.patience:]):
                print('Early stop triggered at epoch {}'.format(epoch))
                break

            # record stats
            print("Epoch {} - Finished with Validation Loss = {}".\
            	format(epoch+1,val_loss))
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            
            # sample a single image
            self.__model.eval()
            image = self.__train_loader.dataset[26][0]
            output_indices = self.__model.sample(\
            	image.view(1,*image.shape).cuda(), device = "cuda",
                                  ** self.config_data['generation'])
            predicted_captions = self.__idx_to_caption(output_indices)
            print(" ".join(predicted_captions[0]))
            print()
            self.__model.train()

        self.plot_stats()

    def batch_learn_core(self, inputs, labels, train=False, test=False):
    	# move data to correct device
        if self.__device == 'cuda':
            inputs = inputs.cuda()
            labels = labels.cuda()

        # forward prop
        if test:
            outputs = self.__best_model(inputs, labels)
        else:
            outputs = self.__model(inputs, labels)

        # calculate loss
        lengths = [len(cap) for cap in labels]
        targets = pack_padded_sequence(labels, lengths, batch_first=True)[0]
        loss = self.__criterion(outputs, targets)

        # back prop
        if train:
            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()
        return loss.item()

    def __train(self):
        self.__model.train()
        training_loss = []

        # train with mini batches
        for i, (images, captions, _) in enumerate(self.__train_loader):
            print( "{:.2f}".format(i / len(self.__train_loader)) , end = '\r')
            batch_loss = self.batch_learn_core(images, captions, 
            	train=True, test=False)
            training_loss.append(batch_loss)
        return np.mean(training_loss)

    def __val(self):
        self.__model.eval()
        val_loss = []

        # validate with mini batches
        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                batch_loss = self.batch_learn_core(images, captions, 
                	train=False, test=False)
                val_loss.append(batch_loss)
        return np.mean(val_loss)
    
    def __idx_to_caption(self, captions_idx):
    	# detach data
        captions_idx = captions_idx.cpu().numpy()
        str_lst = []

        # map index to word
        for i in range(captions_idx.shape[0]):
            str_lst.append([self.__vocab.idx2word[idx] for idx in captions_idx[i,:] \
                if self.__vocab.idx2word[idx] not in ['<start>', '<end>', '<pad>']])
        return str_lst

    def test(self):
    	# load configuration and data
        max_len = self.config_data['generation']['max_length']
        mode = self.config_data['generation']['mode']
        temperature = self.config_data['generation']['temperature']
        test_annotation_file = self.config_data['dataset']['test_annotation_file_path']
        self.__coco = COCO(test_annotation_file)
        print('-----------------------------')
        
        # load best model and states
        if self.__best_model == None:
            self.__best_model = get_model(self.config_data)
            state_dict = torch.load(os.path.join(self.__experiment_dir, 
            	'latest_model.pt'))
            self.__best_model.load_state_dict(state_dict['model'])
        self.__visualization = []
        self.__best_model.eval()

        test_loss = []
        bleu1_scores = []
        bleu4_scores = []
        with torch.no_grad():
            for k, (images, captions, img_ids) in enumerate(self.__test_loader):
                print( "{:.2f}".format(k / len(self.__test_loader)) , end = '\r')
                
                # move data to correct device
                if self.__device == 'cuda':
                    images = images.cuda()
                    captions = captions.cuda()

                # fetch caption predictions
                output_indices = self.__best_model.sample(images, max_len, 
                	mode, temperature, self.__device)
                predicted_captions = self.__idx_to_caption(output_indices)

                # compare predictions with original captions
                for i in range(len(img_ids)):
                    ann_ids = self.__coco.getAnnIds(imgIds=[img_ids[i]])
                    image_captions = [word_tokenize(\
                    	self.__coco.anns[ann_id]['caption'].lower()) \
                    	for ann_id in ann_ids]
                    predicted_caption = predicted_captions[i]

                    # get BLEU1 and BLEU4 scores
                    bleu1_scores.append(bleu1(image_captions, predicted_caption))
                    bleu4_scores.append(bleu4(image_captions, predicted_caption))

                    # integrate data for visualization
                    self.__visualization.append((img_ids[i],
                        [' '.join(image_caption).lower() \
                        	for image_caption in image_captions],
                        ' '.join(predicted_caption)))

                # calculate loss on testing data set
                outputs = self.__best_model(images, captions)
                lengths = [len(cap) for cap in captions]
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                loss = self.__criterion(outputs, targets).item()
                test_loss.append(loss)

        # get results
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}"\
        .format(np.mean(test_loss), np.mean(bleu1_scores), np.mean(bleu4_scores))
        self.__log(result_str)
        return np.mean(test_loss), np.mean(bleu1_scores), np.mean(bleu4_scores)

    def generate_visualization(self):
    	# helper function to generate visualizations
        index = np.random.choice(len(self.__visualization))
        root = os.path.join(self.config_data['dataset']['images_root_dir'], 'test')
        path = self.__coco.loadImgs(self.__visualization[index][0])[0]['file_name'];
        image = Image.open(os.path.join(root, path)).convert('RGB')
        return image, self.__visualization[index][1], self.__visualization[index][2]

    def __save_model(self, tp = 'latest'):
    	# helper function to save model
        root_model_path = os.path.join(self.__experiment_dir, 
        							   '{}_model.pt'.format(tp))
        model_dict = self.__model.state_dict()
        optimizer_dict = self.__optimizer.state_dict()
        scheduler_dict = self.__scheduler.state_dict()
        state_dict = {'model': model_dict, 
        			  'optimizer': optimizer_dict, 
        			  'scheduler': scheduler_dict}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
    	# helper function to save stats
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)
        write_to_file_in_dir(self.__experiment_dir, 
        	'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 
        	'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
    	# helper function to log stats
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
    	# helper function to log epoch stats
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, 
        								 val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
    	# helper function to save and plot stats
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
    
    def get_model(self):
        return self.__best_model
    def get_loss(self):
        return self.__val_losses, self.__training_losses
    def get_data(self):
            return self.__train_loader
