import os
from random import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from skimage.feature import hog
from skimage import data, exposure

global t
t = 0


class DataGenerator(torch.utils.data.Dataset):

    def __init__(self, dataroot, patlist, batch_size, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.RandomRotation(45), transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(), transforms.RandomCrop(256)])):
        # print('__init__')
        self.batch_size = batch_size
        self.dataroot = dataroot

        self.list_pat = patlist
        self.list_frames = []
        self.list_masks = []
        for i, p in enumerate(self.list_pat):
            if ('Video' in p):
                elList = sorted(os.listdir(os.path.join(dataroot, p, 'images')))
                for x in elList:
                    self.list_frames.append(os.path.join(p, 'images', x))
                    self.list_masks.append(os.path.join(p, 'labels', x))
            #self.transform_image = transform
            #if len(self.transform_image.transforms) > 2:
            #    self.transform_image.transforms.insert(len(self.transform_image.transforms),
             #                                          transforms.ColorJitter(brightness=0.2))
            self.transform = transform

            if len(self.transform.transforms) > 2:
                self.transform_image = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(45), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                                           transforms.RandomCrop(256), transforms.ColorJitter(brightness=0.2)])

            else:
                self.transform_image = transform

    def __len__(self):
        # print('__len__')
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        i = len(self.list_frames)
        # print('The lenght of list_ids is {0}'.format(len(self.list_IDs)))
        # print('The division is {0}'.format(len(self.list_IDs) / self.batch_size))
        # print('The divison after the np.flor is {0}'.format(i))
        return i

    def __getitem__(self, index):
        # print('__getitem__')
        if torch.is_tensor(index):
            index = index.tolist()
        # print(index)
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # file_list_temp = [self.list_IDs[k] for k in indexes]
        # print('The lenght of the file temp is {0}'.format(len(file_list_temp)))
        x, y, hog_image, hog_vector = self.__data_generation(index)

        # print('The lenght of X is {0}'.format(len(X)))
        # print('The lenght of y_new is {0}'.format(len(y_new)))
        return x, y, hog_image, hog_vector

    def __data_generation(self, index):

        x_file_path = os.path.join(self.dataroot, self.list_frames[index])
        y_file_path = os.path.join(self.dataroot, self.list_masks[index])
        X = Image.open(x_file_path)
        X = X.resize((448, 448), Image.NEAREST)
        y = Image.open(y_file_path)
        y = y.resize((448, 448), Image.BILINEAR)

        if self.transform is not None:
            seed = np.random.randint(2147483647)
            # random.seed(seed)
            torch.manual_seed(seed)
            X = self.transform_image(X)
            # random.seed(seed)
            torch.manual_seed(seed)
            y = (self.transform(y) * 255).long()
            y = torch.squeeze(y)

            X_inverted = X.permute(1, 2, 0).contiguous()
            X_numpy = np.array(X_inverted)

            hog_vector, hog_image = hog(X_numpy, orientations=8, pixels_per_cell=(16, 16),
                                        cells_per_block=(1, 1), visualize=True)
            hog_image_tensor = torch.from_numpy(hog_image)
            hog_vector = torch.from_numpy(hog_vector).float()

        return X, y, hog_image_tensor, hog_vector
