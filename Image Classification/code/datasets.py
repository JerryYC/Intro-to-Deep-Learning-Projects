"""
Author: Colin Wang
"""
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class bird_dataset(Dataset):
    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self,root,file_path, transform = None):
        data = np.loadtxt(file_path, delimiter = ' ', dtype=str)
        self.root = root
        self.y = data[:,1].astype(int)
        self.files = data[:,0]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item, nmode = 'imagenet'):
        # load data
        fp = osp.join(self.root, self.files[item])
        # convert to rgb
        img = Image.open(fp).convert('RGB')
        # transform data if possible
        if self.transform:
            img = self.transform(img)
        return img, self.y[item]
