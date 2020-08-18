import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image
import random
from torchvision import transforms


class train_dataset(data.Dataset):
    def __init__(self, dataset_path=''):
        super(train_dataset, self).__init__()
        fh = open(dataset_path + '/train.txt', 'r')
        imgs = []
        for image_names in fh:
            image_names = image_names.rstrip()
            names = image_names.split('/')
            imgs.append((names[0], names[1]))
        self.imgs = imgs
        self.dataset_path = dataset_path
        self.transform = transforms.Compose(
            [transforms.CenterCrop(256),
             transforms.ToTensor()])

    def __getitem__(self, index):

        image_name, label_name = self.imgs[index]
        image = Image.open(self.dataset_path + "/train/" +
                           image_name).convert('RGB')
        label = Image.open(self.dataset_path + "/train_labels/" +
                           label_name).convert('L')

        image = self.transform(image)
        label = self.transform(label) * 255
        return image, label

    def __len__(self):
        return len(self.imgs)
