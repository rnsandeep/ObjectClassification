# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
import copy, cv2
import sys, shutil, pickle
from sklearn.metrics import classification_report, confusion_matrix
from os import listdir
from os.path import isfile, join
import torch.nn.functional as F
import scipy.ndimage
import matplotlib.pyplot as plt

mypath = sys.argv[1]

images = [os.path.join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

mean = [0.4616, 0.4006, 0.3602]
std = [0.2287, 0.2160, 0.2085]

crop_size = 224

resize_size = 224


data_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
      ])

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #"cpu"
print("device found:", device)

def load_model(path):
    model = torch.load(path)
    return model

def load_inputs_outputs(dataloaders):
    phase = 'val'
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
    return inputs, labels

def convert_to_numpy(x):
    return x.data.cpu().numpy()


def load_tensor_inputs(path, data_transforms):
    loader = data_transforms['test']
    image = Image.open(path[0])
    return loader(image)

def cv2_to_pil(img):
        """Returns a handle to the decoded JPEG image Tensor"""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        return im_pil

def eval_model(model, image):
    model.eval()  
    inputs = image 
    inputs = cv2_to_pil(inputs)
    inputs = data_transform(inputs)
    image_shape = inputs.size()
    inputs = inputs.reshape((1, 3, image_shape[1], image_shape[2])).to(device)
    with torch.no_grad():
       outputs = model(inputs)
    outputs = F.softmax(outputs, dim=1)
    cls = torch.argmax(outputs).item()
    return cls #classes[cls]


def eval_all(images, model):
    cls_dict = dict()
    for image in images:
        try:  
          print(image)
          image_np = cv2.imread(image)
          cls = eval_model(model, image_np)
          cls_dict[image] = cls
        except Exception as ex:
            print(ex)
            print("reading failed for :", image)
    return cls_dict    
        
    
def load_resnet18_model(model_path, num_classes):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes)
    model = model_ft.to(device)
    checkpoint = load_model(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def copyImages(cls_dict,output_dir):
    for key in cls_dict:
        output_path = os.path.join(output_dir, cls_dict[key])
#        print(output_path)
        if not os.path.exists(output_path):
             os.makedirs(output_path)
             shutil.copy(key, output_path)
        else:
              shutil.copy(key, output_path)


if __name__=="__main__":
    model_path = sys.argv[2]
    num_classes = int(sys.argv[3])
    output_dir = sys.argv[4]
    if not  os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(model_path)
    model = load_resnet18_model(model_path, num_classes)    

    since = time.time()
    cls_dict = eval_all(images, model)
    count  = 0
    for image in cls_dict:
        if cls_dict[image] == 'NOSE-PIGM':
            count = count +1
    print(count, len(cls_dict.keys()))
              
    last = time.time()
    total_time = last-since
    print("total time taken to process;", total_time, "per image:", total_time*1.0/len(cls_dict.keys()))
    #copyImages(cls_dict, output_dir)
    #pickle.dump([accuracy, label, output],open(os.path.join(output_dir, os.path.basename(model_path)[:-8]+'_'+str(crop_size)+'_'+str(resize_size)+'_accuracy.pkl'),'wb'))
