# -*- coding: utf-8 -*-

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
import copy, cv2
import sys, shutil, pickle
from sklearn.metrics import classification_report, confusion_matrix
#from time import time
# Data augmentation and normalization for training
# Just normalization for validation


def datatransforms(mean, std, crop_size, resize_size):
    data_transforms = {
      'train': transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize( mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05] ) #mean, std) #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
        transforms.Resize(424),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05]) #mean, std)#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'test': transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05]) #mean, std)#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),

    }
    return data_transforms


data_dir = sys.argv[1]

mean_file =  sys.argv[3]

#mean_std = np.load(mean_file)

mean = torch.tensor([0.4616, 0.4006, 0.3602])
std = torch.tensor([0.2287, 0.2160, 0.2085])
#mean = list(mean_std[0].data.cpu().numpy())
#std = list(mean_std[1].data.cpu().numpy())

crop_size = int(sys.argv[4])

resize_size = int(sys.argv[5])
data_transforms = datatransforms( mean, std, crop_size, resize_size)


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in [ 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=4)
              for x in [ 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['test'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print("device found:", device)

# Get a batch of training data
#inputs, classes = next(iter(dataloaders['t']))


def load_model(path):
    model = torch.load(path)
    return model

def load_inputs_outputs(dataloaders):
    phase = 'test'
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

def eval_model(model, dataloaders):
    model.eval()   # Set model to evaluate mode
    phase = 'test'
    running_corrects = 0
    output = []
    label = []
    for inputs, labels, paths in dataloaders[phase]:
    #    print(paths)
        inputs  = load_tensor_inputs(paths, data_transforms)#.to(device)
        inputs = inputs.reshape((1,3,299,299)).to(device)  
        labels = labels.to(device)
        outputs = model(inputs)
    #    print(outputs)
        _, outputs = torch.max(outputs, 1)
    #    print(outputs)
    #    print(labels)
        outputs_np = convert_to_numpy(outputs)
        labels_np = convert_to_numpy(labels)
        labels_np[np.where(labels_np==2)] = 1
        output += (list(outputs_np))
        label += (list(labels_np))
        running_corrects += np.sum(outputs_np == labels_np)
    accuracy = running_corrects*1.0/dataset_sizes['test']
#    print(classification_report(list(labels_np), list(outputs_np)))
#    print(confusion_matrix(label, output))
#    print("correct:", running_corrects, "total:",  dataset_sizes['test'])
    return accuracy, label, output

    
def load_inception_model(model_path, num_classes):
    model_ft = models.inception_v3(pretrained=True)
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes)
    model = model_ft.to(device)
    checkpoint = load_model(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__=="__main__":
    model_path = sys.argv[2]
    num_classes = int(sys.argv[6])
    output_dir = sys.argv[7]
    if not  os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(model_path)
    model = load_inception_model(model_path, num_classes)    
    
    since = time.time()
    accuracy, label, output = eval_model(model, dataloaders)
    last = time.time()
    total_time = last-since
    print("total time taken to process;", total_time, "per image:", total_time*1.0/len(output))
    pickle.dump([accuracy, label, output],open(os.path.join(output_dir, os.path.basename(model_path)[:-8]+'_'+str(crop_size)+'_'+str(resize_size)+'_accuracy.pkl'),'wb'))
