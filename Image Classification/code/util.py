"""
Author: Colin Wang, Jerry Chan, Bingqi Zhou
"""
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import model
import datasets
import train
import numpy as np
import random
import matplotlib.pyplot as plt
def get_indices(used_valid_indices, size, step, num):
    """
    Get indices for train-valid split
    params:
        used_valid_indices: indices that cannot be used for validation
        size: range of indices
        step: number of samples in each class
        num: number of samples in each class for validation
    returns:
        train_indices: indices for training set
        valid_indices: indices for validation set
    """
    valid_indices = []
    for i in range(0, size, step):
        available_indices = list(range(i, i+step))
        for used_idx in used_valid_indices:
            try:
                available_indices.remove(used_idx)
            except ValueError: pass
        indices = random.sample(available_indices, num)
        valid_indices.extend(indices)
    train_indices = list(range(0,size))
    for valid_idx in valid_indices:
        train_indices.remove(valid_idx)
    return train_indices, valid_indices

def reverse_one_hot(encoding):
    """
    Reverse one hot encodings
    params:
        encoding: softmax output encoding
    returns:
        np.argmax(encoding, axis=1): predicted label
    """
    return np.argmax(encoding, axis=1)

def find_accuracy(predicted, target):
    """
    find accuracy between prediction and targets
    params:
        predicted: softmax output encoding
        target: actual labels
    returns:
        1 - np.count_nonzero(predicted - target) / predicted.shape[0]:
        percent of correct predictions
    """
    # print(predicted, target)
    predicted, target = reverse_one_hot(predicted.cpu().detach().numpy()), \
                        target.cpu().detach().numpy()
    return 1 - np.count_nonzero(predicted - target) / predicted.shape[0]

def plot_kernels(tensor,name = "weight maps"):
    """
    plot CNN kernel
    """
    # tensor to (num of kernal, height,  width, 3) np array
    tensor = tensor.cpu().data
    tensor = np.array(tensor)
    tensor = np.einsum('ijkl->ilkj', tensor)
    
    # scale the value to [0,1]
    min_val = tensor.min()
    max_val = tensor.max()
    tensor = (tensor - min_val) / (max_val - min_val)
    
    # plot images
    num_kernels = tensor.shape[0]
    num_cols = num_kernels ** (1/2)
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    fig.suptitle(name, fontsize=24)
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
    
    # show images
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()
    
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def draw_feature_map(model, layer, dataset, fig_row, fig_column, fig_path, fig_title):
    layer.register_forward_hook(get_activation('layer'))
    data, _ = dataset[0]
    data = data.cuda()
    data.unsqueeze_(0)
    output = model(data)

    act = activation['layer'].squeeze().cpu()
    fig, axes = plt.subplots(fig_row, fig_column, figsize=(25,25))
    fig.suptitle(fig_title, fontsize = 20)

    # plot figure
    i = 0
    for idx1 in range(axes.shape[0]):
        for idx2 in range(axes.shape[1]):
            if (i < activation['layer'].shape[1]):
                axes[idx1, idx2].imshow(act[i])
                i += 1
            else:
                break
    fig.savefig(fig_path)
    
def draw_weight_map(layer, fig_row, fig_column, fig_path, fig_title):
    # get weights
    weights = layer.weight
    weights = weights.cpu().data
    
    # scale the value to [0,1]
    min_val = weights.min()
    max_val = weights.max()
    weights = (weights - min_val) / (max_val - min_val)
    print(weights)
    
    # plot config
    fig, axes = plt.subplots(fig_row, fig_column, figsize=(25,25))
    fig.suptitle(fig_title, fontsize = 20)
    # plot figure
    i = 0
    for idx1 in range(axes.shape[0]):
        for idx2 in range(axes.shape[1]):
            axes[idx1, idx2].imshow(weights[i])
            i += 1
    fig.savefig(fig_path)
    plt.show()
