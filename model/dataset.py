import torch
from torch import nn
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm as tqdm
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pickle
import torchvision
import requests
import zipfile
from pathlib import Path
from params import *

def club_transforms(grayscale=False,convert=True):
  transform_list=[]
  if grayscale==True:
    transform_list.append(transforms.Grayscale(1))
  if convert==True:
    transform_list.append(transforms.ToTensor())
    if grayscale==True:
      transform_list.append(transforms.Normalize((0.5,),(0.5,)))
    else:
      transform_list.append(transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)))
  return transforms.Compose(transform_list)

class TextDataset():
  def __init__(self, base_path = '/content/IAM-32.pickle',  num_examples = 15, target_transform=None):

        #IN THE ABOVE FUNCTION THE PARAMETERS ARE SUBJECT TO CHANGE ACCORDING TO THE CODE
        self.NUM_EXAMPLES = num_examples #THIS IS THE NUMBER OF EXAMPLES PER AUTHOR??

        #base_path = DATASET_PATHS
        file_to_store = open(base_path, "rb") #HERE "rb" INDICATES BINARY MODE i.e. THE FILE IS OPENED IN BINARY MODE
        #NOW TO EXTRACT DATA FROM THIS BINARY FILE WE PICKLE MODULE
        self.IMG_DATA = pickle.load(file_to_store)['train']
        #print(self.IMG_DATA)##THIS LINE IS AN ADDITION,NEEDS TO BE COMMENTED OUT LATER
        #IN THE ABOVE LINE train data of the base_path FILE IS ASSIGNED TO IMG_DATA FOR FURTHER TRAINING PURPOSE
        self.IMG_DATA  = dict(list( self.IMG_DATA.items()))#[:NUM_WRITERS])
        #print (self.IMG_DATA)##THIS LINE IS AN ADDITION,NEEDS TO BE COMMENTED OUT LATER
        #The above line creates a dictionary consisting of the data of original dictionary in tuple form
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']
        #IN BELOW STATEMENT AUTHOR_ID IS OBTAINED
        self.author_id = list(self.IMG_DATA.keys())
        #print(self.author_id)##THIS LINE IS AN ADDITION,NEEDS TO BE COMMENTED OUT LATER
        #THE ABOVE PRINT STATEMENT SHOWS TAHT THERE ARE AROUND 360 UNIQUE AUTHOR IDS
        #A tranform attribute is assigned the function club_transforms
        self.transform = club_transforms(grayscale=True)
        #print(self.transform)##THIS LINE IS AN ADDITION,NEEDS TO BE COMMENTED OUT LATER
        #WHAT IS target_transform??
        self.target_transform = target_transform
        self.collate_fn = TextCollator()


  def __len__(self):
        return len(self.author_id)##339

  def __getitem__(self, index):

        #RECEIVES INDEX

        NUM_SAMPLES = self.NUM_EXAMPLES
        #print(NUM_SAMPLES)


        author_id = self.author_id[index]
        #print(author_id)

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        #print(self.IMG_DATA_AUTHOR) ##FOR ONE SINGLE AUTHOR WE HAVE AMPLE OF DATA
        random_idxs = np.random.choice(len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace = True)##IF REPLACE IS TRUE THAT MEANS THAT SAMPLES MAY REPEAT
        #print(random_idxs)
        #THE random_idxs stores the ids any random 15 samples per author
        rand_id_real = np.random.choice(len(self.IMG_DATA_AUTHOR))
        #print(rand_id_real)
        real_img = self.transform(self.IMG_DATA_AUTHOR[rand_id_real]['img'].convert('L')) #THE L MODE MAPS IMAGE TO BLACK AND WHITE MODE
        #print(real_img)
        real_labels = self.IMG_DATA_AUTHOR[rand_id_real]['label'].encode()
        #print(real_labels)

        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        labels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]
        #print(imgs)
        #print(labels)
        max_width = 192 #[img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:

            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros(( img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg
            #print(img)
            imgs_pad.append(self.transform((Image.fromarray(img))))
            #print(imgs_pad)
            imgs_wids.append(img_width)
            #print(imgs_wids)
        imgs_pad = torch.cat(imgs_pad, 0)

        item = {'simg': imgs_pad, 'swids':imgs_wids, 'img' : real_img, 'label':real_labels,'img_path':'img_path', 'idx':'indexes', 'wcl':index}
        #HERE wcl returns the index itself
        return item

class TextCollator(object):
    def __init__(self):
        self.resolution = 16

    def __call__ (self, batch):

        img_path = [item['img_path'] for item in batch]
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        simgs =  torch.stack([item['simg'] for item in batch], 0)
        wcls =  torch.Tensor([item['wcl'] for item in batch])
        swids =  torch.Tensor([item['swids'] for item in batch])
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)], dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)
        item = {'img': imgs, 'img_path':img_path, 'idx':indexes, 'simg': simgs, 'swids': swids, 'wcl':wcls}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        if 'z' in batch[0].keys():
            z = torch.stack([item['z'] for item in batch])
            item['z'] = z
        return item
##FOR VALIDATION DATA
class TextDatasetval():

    def __init__(self, base_path = '/content/IAM-32.pickle', num_examples = 15, target_transform=None):

        self.NUM_EXAMPLES = num_examples
        #base_path = DATASET_PATHS
        file_to_store = open(base_path, "rb")
        self.IMG_DATA = pickle.load(file_to_store)['test']
        self.IMG_DATA  = dict(list( self.IMG_DATA.items()))#[NUM_WRITERS:])
        if 'None' in self.IMG_DATA.keys():
            del self.IMG_DATA['None']
        self.author_id = list(self.IMG_DATA.keys())

        self.transform = club_transforms(grayscale=True)
        self.target_transform = target_transform

        self.collate_fn = TextCollator()


    def __len__(self):
        return len(self.author_id)

    def __getitem__(self, index):

        NUM_SAMPLES = self.NUM_EXAMPLES

        author_id = self.author_id[index]

        self.IMG_DATA_AUTHOR = self.IMG_DATA[author_id]
        random_idxs = np.random.choice(len(self.IMG_DATA_AUTHOR), NUM_SAMPLES, replace = True)

        rand_id_real = np.random.choice(len(self.IMG_DATA_AUTHOR))
        real_img = self.transform(self.IMG_DATA_AUTHOR[rand_id_real]['img'].convert('L'))
        real_labels = self.IMG_DATA_AUTHOR[rand_id_real]['label'].encode()


        imgs = [np.array(self.IMG_DATA_AUTHOR[idx]['img'].convert('L')) for idx in random_idxs]
        labels = [self.IMG_DATA_AUTHOR[idx]['label'].encode() for idx in random_idxs]

        max_width = 192 #[img.shape[1] for img in imgs]

        imgs_pad = []
        imgs_wids = []

        for img in imgs:

            img = 255 - img
            img_height, img_width = img.shape[0], img.shape[1]
            outImg = np.zeros(( img_height, max_width), dtype='float32')
            outImg[:, :img_width] = img[:, :max_width]

            img = 255 - outImg

            imgs_pad.append(self.transform((Image.fromarray(img))))
            imgs_wids.append(img_width)

        imgs_pad = torch.cat(imgs_pad, 0)


        item = {'simg': imgs_pad, 'swids':imgs_wids, 'img' : real_img, 'label':real_labels,'img_path':'img_path', 'idx':'indexes', 'wcl':index}



        return item
