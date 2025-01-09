import os
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10

class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None, download=False):          
        super(MyCIFAR10, self).__init__(root=root, train=train, transform=transform, download=download)
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)
            
        return img, target