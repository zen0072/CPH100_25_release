"""
Simple CNN architectures for PathMNIST classification.
Includes MLP baseline and CNN variants for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    """
    Simple MLP model: Flatten input then run through hidden layers.
    """
    
    def __init__(self, num_classes=9):
        super(MLPModel, self).__init__()
        
        # PathMNIST images are 3x28x28 = 2352 features
        input_size = 3 * 28 * 28
        
        # TODO: Add your own MLP architecture here
        self.flatten = nn.Flatten()
        self.h1= nn.Linear(input_size, 512)
        self.h2= nn.Linear(512, 256)
        self.h3=nn.Linear(256,num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        x = self.flatten(x)         
        x = self.relu(self.h1(x))
        x = self.relu(self.h2(x))
        return self.h3(x)
       # raise NotImplementedError("MLPModel is not implemented")

class CNNModel(nn.Module):
    """
    Simple CNN model: TODO: Add your own architecture here
    """
    
    def __init__(self, num_classes=9):
        super(CNNModel, self).__init__()
        
        # TODO: Add your own CNN architecture here
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential (
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256,num_classes)

        )
    
    def forward(self, x):
        return self.classifier(self.features(x))
        #raise NotImplementedError("CNNModel is not implemented")

def get_model(model_name, num_classes=9):
    """Get model by name."""
    if model_name == 'mlp':
        return MLPModel(num_classes)
    elif model_name == 'cnn':
        return CNNModel(num_classes)
    else:
        #TODO: add your models names here
        raise ValueError("Unknown model: {}".format(model_name))

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
