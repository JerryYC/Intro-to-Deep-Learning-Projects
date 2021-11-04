"""
Author: Colin Wang, Jerry Chan
"""
import torch.nn as nn
import torchvision.models as models
class baseline_Net(nn.Module):
    """
    Baseline model
    """
    def __init__(self, classes):
        super(baseline_Net, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, classes),
        )

    # intiailize weights for all convolutional layers and linear layers
    def initialize_weights(self, mode='uniform'):
        for seq in [self.b1, self.b2, self.b3, self.b4, self.fc1, self.fc2]:
            for layer in seq:
                if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
                    if mode =='uniform':
                        nn.init.xavier_uniform(layer.weight)
                    elif mode =='normal':
                        nn.init.xavier_normal(layer.weight)
                    else:
                        raise Exception()
                    nn.init.zeros_(layer.bias)

    # forward propogation
    def forward(self, x):
        out1 = self.b2(self.b1(x))
        out2 = self.b4(self.b3(out1))
        out_avg = self.avg_pool(out2)
        out_flat = out_avg.view(-1, 256)
        out4 = self.fc2(self.fc1(out_flat))
        return out4

# customized CNN model
class custom_Net(nn.Module):

    def __init__(self, classes):
        super(custom_Net, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 3, 5, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(64, 192, 5, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.b5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.b6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classes)
        )

    # intiailize weights for all convolutional layers and linear layers
    def initialize_weights(self, mode='uniform'):
        for seq in [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.fc1, self.fc2]:
            for layer in seq:
                if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
                    if mode =='uniform':
                        nn.init.xavier_uniform_(layer.weight)
                    elif mode =='normal':
                        nn.init.xavier_normal(layer.weight)
                    else:
                        raise Exception()

    def forward(self, x):
        # residual
        out0 = x + self.b1(x)
        out1 = self.b2(out0)
        out2 = self.b4(self.b3(out1))
        out3 = self.b6(self.b5(out2))
        out_avg = self.avg_pool(out3)
        out_flat = out_avg.view(-1, 256*6*6)
        out4 = self.fc2(self.fc1(out_flat))
        return out4
    
class vgg(nn.Module):
    """
    Pretrained VGG model
    """
    def __init__(self, classes):
        super(vgg, self).__init__()
        # retrive pretrained model
        self.add_module("pretrain",models.vgg16_bn(pretrained=True))
        # remove the last fc layer and add our own fc layer
        pretrain_outshape = self.pretrain.classifier[-1].in_features
        self.pretrain.classifier = nn.Sequential(*list(self.pretrain.classifier.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(pretrain_outshape, classes),
        )
        
    # forward propogation   
    def forward(self, x):
        return self.fc(self.pretrain(x))
    
    def initialize_weights(self, mode='normal'):
        # only initialize the last fc layer
        for layer in self.fc:
            if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
                if mode =='uniform':
                    nn.init.xavier_uniform(layer.weight)
                elif mode =='normal':
                    nn.init.xavier_normal(layer.weight)
                else:
                    raise Exception()
                nn.init.zeros_(layer.bias)

    def freeze(self):
        """
        freeze the pretrained layers
        """
        for parameter in self.pretrain.parameters():
            parameter.requires_grad = False
    
    def freeze_partial(self):
        """
        freeze the pretrained layers except the last layer
        """
        self.freeze()
        for parameter in self.pretrain.classifier.parameters():
            parameter.requires_grad = True
            
    def unfreeze(self):
        """
        unfreeze the pretrained layers
        """
        for parameter in self.parameters():
            parameter.requires_grad = True
        
class resnet(nn.Module):
    """
    Pretrained ResNet model
    """
    def __init__(self, classes):
        super(resnet, self).__init__()
        # retrive pretrained model
        self.add_module("pretrain",models.resnet18(pretrained = True))
         # remove the last fc layer and add our own fc layer
        pretrain_outshape = self.pretrain.fc.in_features
        self.pretrain.fc = Identity()
        self.fc = nn.Sequential(
            nn.Linear(pretrain_outshape, classes)
        )
       
    def forward(self, x):
        return self.fc(self.pretrain(x))
    
    def initialize_weights(self, mode='normal'):
        # only initialize the last fc layer
        for layer in self.fc:
            if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
                if mode =='uniform':
                    nn.init.xavier_uniform(layer.weight)
                elif mode =='normal':
                    nn.init.xavier_normal(layer.weight)
                else:
                    raise Exception()
                nn.init.zeros_(layer.bias)

    def freeze(self):
        """
        freeze the pretrained layers
        """
        for parameter in self.pretrain.parameters():
            parameter.requires_grad = False
    
    def freeze_partial(self):
        """
        freeze the pretrained layers except the last layer
        """
        self.freeze()
        for parameter in self.pretrain.layer4[1].parameters():
            parameter.requires_grad = True
        for parameter in self.pretrain.avgpool.parameters():
            parameter.requires_grad = True
            
    def unfreeze(self):
        """
        unfreeze the pretrained layers
        """
        for parameter in self.parameters():
            parameter.requires_grad = True

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x 
