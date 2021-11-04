"""
Author: Colin Wang
"""
import train
import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

    ###############################
    # DON'T CHANGE ANY CODE ABOVE #
    ###############################

###############################################################################################################

    #########################
    # CHANGE ANY CODE BELOW #
    #########################

def main():

    #############
    # SECTION 1 #
    #############
    # initialize image augmentation
    train_transforms = transforms.Compose([
    transforms.Resize(256),                         # resize image
    transforms.CenterCrop(224),                     # center crop to reach 224x224
    transforms.RandomHorizontalFlip(),              # flip horizontally randomlly
    transforms.RandomVerticalFlip(),                # flip vertically randomly
    transforms.RandomRotation(30, expand=False),    # random rotate within 30 degrees
    transforms.ToTensor(),                          # convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                                    # normalize to imagenet mean and std

    test_transforms = transforms.Compose([
    transforms.Resize(256),                         # resize image
    transforms.CenterCrop(224),                     # center crop to reach 224x224
    transforms.ToTensor(),                          # convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                                    # normalize to imagenet mean and std

    
    #############
    # SECTION 2 #
    #############
    # initialize models and hyperparameters
    model_name =     'baseline'   # options: baseline, custom, vgg16, resnet18
    batch_size =      20          # number of images per batch
    lr =              1e-4        # learning rate
    weight_mode =    'uniform'    # xavier weight initialization mode, options: 'uniform', 'normal'
    device =         'cuda'       # don't change this options: 'cuda', 'cpu'
    weight_decay =    5e-4        # L2 penalty rate
    eps =             1e-08       # eps
    amsgrad =         False       # AMSGrad variant
    num_of_classes =  20          # number of classes in the dataset
    freeze =          True        # if true, all except last layer for vgg16 and resnet18 will be freezed
    model_path =      'model/{}{}.pth'.format(model_name,"" if freeze else "_not_freezed") 
                                  # model save path
    save_mode =       'acc'       # options: 'acc', 'loss'

    #############
    # SECTION 3 #
    #############
    # initialize training details
    epochs =          4           # Number of epochs, at least 4 (or scheduler will complain)
    K =               1           # Number of folds
    validate_option = True        # Validation option

###############################################################################################################
    
    ###############################
    # DON'T CHANGE ANY CODE BELOW #
    ###############################
    
    # Load dataset and dataloader
    root = 'birds_dataset/images/'
    train_dataset = datasets.bird_dataset(root,'birds_dataset/train_list.txt', transform = train_transforms)
    test_dataset = datasets.bird_dataset(root,'birds_dataset/test_list.txt', transform = test_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # initializer trainer
    trainer = train.trainer(train_dataloader, test_dataloader, model_name, \
        batch_size=batch_size, lr=lr, weight_mode=weight_mode, \
        device=device, weight_decay=weight_decay, eps=eps, amsgrad=amsgrad, \
        num_of_classes=num_of_classes,\
        model_path = model_path, freeze = freeze, save_mode = save_mode)

    # train, validate, predict, and save the best model
    trainer.train(epochs=epochs, K=K, validate_option=True)

    # get summary of training, validation and test stats
    train_acc, train_loss, val_acc, val_loss = trainer.get_training_stats()
    test_acc, test_loss = trainer.get_test_stats()

    # display test stats
    print('test accuracy: {}'.format(test_acc))
    print('test loss: {}'.format(test_loss))

    # plot training and validation stats
    f = plt.figure(figsize=(12,5))
    ax = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    # plot training and validation accuracy
    ax.plot(train_acc, label = "Training accuracy")
    ax.plot(val_acc, label = "Validation accuracy")
    ax.legend(loc='upper left')
    # plot training and validation loss
    ax2.plot(train_loss, label = "Training loss")
    ax2.plot(val_loss, label = "Validation loss")
    ax2.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()
