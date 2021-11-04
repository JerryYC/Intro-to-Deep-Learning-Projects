################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
# Finely implemented and refined by Colin Wang, Jerry Chan
################################################################################

import torch.nn as nn
import torch
import torchvision.models as models
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

# Build and return the model here based on the configuration.
def get_model(config_data):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    if config_data['model']['device'] == 'cuda':
        return baseline(config_data).cuda()
    return baseline(config_data)
    
class baseline(nn.Module):
    def __init__(self, config_data):
        super(baseline, self).__init__()
        self.config = config_data['model']
        self.model_type = config_data['model']['model_type']

        # encoder
        resnet = models.resnet50(pretrained = True)
        pretrain_outshape = resnet.fc.in_features
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(pretrain_outshape, 
        						self.config['embedding_size'])
        self.bn = nn.BatchNorm1d(self.config['embedding_size'], momentum=0.01)

        # decoder
        self.embedding = nn.Embedding(self.config['vocab_size'], 
        							  self.config['embedding_size'])
        if self.model_type == 'baseline':
            self.rnn = nn.LSTM(input_size = self.config['embedding_size'],
                               hidden_size = self.config['hidden_size'], 
                               batch_first=True)
        if self.model_type == 'arch2':
            self.rnn = nn.LSTM(input_size = self.config['embedding_size'] * 2,
                               hidden_size = self.config['hidden_size'], 
                               batch_first=True)
        if self.model_type == 'rnn':
            self.rnn = nn.RNN(input_size = self.config['embedding_size'],
                              hidden_size = self.config['hidden_size'], 
                              batch_first=True)
        self.fc = nn.Linear(self.config['hidden_size'], 
        					self.config['vocab_size'])

        # freeze the encoder parameters if not fine-tune
        if not self.config['finetune']:
            self.freeze()       
        
    def forward(self, images, captions):
    	# getting image feature vectors
        features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))

        # embed/concatenate image feature vectors with caption embeddings
        if self.model_type == 'arch2':
            captions = torch.cat([torch.zeros((captions.shape[0],1), 
            					 device = 'cuda', 
            					 dtype = torch.long), 
                                 captions[:,:-1]],1)
        embed = self.embedding(captions) # batch_size, seq_len, embed_size
        if self.model_type == 'baseline' or self.model_type == 'rnn':
            embed = torch.cat((features.unsqueeze(1), embed), dim = 1)
        lengths = [len(cap) for cap in captions]
        if self.model_type == 'arch2':
            features = features.view([features.shape[0], 1, -1])\
            				   .repeat((1,captions.shape[1] ,1))
            embed = torch.cat([features, embed],2)

        # pack sequential embeddings and send to rnn
        # getting results with a linear layer
        embed = pack_padded_sequence(embed, lengths, batch_first=True)
        rnn_outputs, _ = self.rnn(embed)
        out = self.fc(rnn_outputs[0])
        return out

    def sample(self, images, max_length, mode, temperature, device):
    	# initialize variables
        softmax = nn.Softmax(dim=1)
        sampled_ids = []

        # getting image feature vectors
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features)).unsqueeze(1)
        inputs = features

        # create initial paddings if model is arch2
        if self.model_type == 'arch2':
            image_features = inputs.clone()
            inputs = self.embedding(torch.zeros((images.shape[0],1),
            						device = 'cuda', 
            						dtype = torch.long))
        states = None

        # generate captions word by word
        for i in range(max_length):
            if self.model_type == 'arch2':
                inputs = torch.cat([image_features, inputs], 2)
            rnn_outputs, states = self.rnn(inputs, states)    
            rnn_outputs = rnn_outputs.squeeze(1) 
            out = self.fc(rnn_outputs)
            caption_idx = self.generate_idx(mode, 
            	out, softmax, temperature, device)
            sampled_ids.append(caption_idx)
            inputs = self.embedding(caption_idx).unsqueeze(1)

        # integrate
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

    def generate_idx(self, mode, out, softmax, temperature, device):
    	# determinstic mode, where we use greedy search algorithm
        if mode == 'deterministic':
            _, caption_idx = out.max(1)

        # stochastic mode, where we soften the distribution by a factor, 
        # and then sampling from this distribution with softmax
        elif mode == 'stochastic':
            soft_out = softmax(out/temperature)
            p = soft_out.data.cpu().numpy()
            caption_idx = [np.random.choice(p.shape[1], p=p[j]) \
            				for j in range(p.shape[0])]
            caption_idx = torch.tensor(caption_idx, dtype=torch.long)
            if device == 'cuda':
                caption_idx = caption_idx.cuda()
        else:
            raise Exception('Failed to recognize generation mode')
        return caption_idx
    
    def freeze(self):
    	# freeze all parameters in resnet 
    	# (except for the last layer that is not included in the resnet model)
        for param in self.resnet.parameters():
            param.requires_grad_(False)
