import json
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
import re
input_size=[224,224]
transform_train = transforms.Compose([
            transforms.Resize(input_size, interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class image_dataset_train(Dataset):
    def __init__(self, image_root,train_txt,preprocess):  
        with open(train_txt, "r") as f:
            self.files=[]
            for data in f.readlines():
                data = data.strip("\n")
                self.files.append(data)     
        self.transform = preprocess
        self.image_root=image_root
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_path = self.image_root+"/"+self.files[index]+".jpg"
        image = Image.open(img_path).convert('RGB') 
        image = self.transform(image)

        return image
"""
class image_dataset_test(Dataset):
    def __init__(self, image_root,test_txt,caption_txt):  
        with open(test_txt, "r") as f:
            self.files=[]
            for data in f.readlines():
                data = data.strip("\n")
                self.files.append(data)     
        self.transform = transform_train
        self.image_root=image_root
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        img_path = self.real_image_root+"/"+self.files[index]
        image = Image.open(img_path).convert('RGB') 
        image = self.transform(image)

        return image
"""


def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


