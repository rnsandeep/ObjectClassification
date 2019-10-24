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
import copy
import sys, shutil
import math
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def cyclical_lr(stepsize, min_lr=3e-2, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.
#
# .. Note ::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
#    and extract it to the current directory.

# Data augmentation and normalization for training
# Just normalization for validation


def datatransforms(mean_file, crop_size):
#`    mean_std = np.load(mean_file)
    mean = [0.485, 0.456, 0.406] #list(mean_std[0].data.cpu().numpy())
    std = [0.229, 0.224, 0.225] #list(mean_std[1].data.cpu().numpy())
    print("mean and standard deviation:",mean,std)
    data_transforms = {
      'train': transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation((-10,10)),
        transforms.ToTensor(),
        transforms.Normalize( mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05] ) #mean, std) #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05]) #mean, std)#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'test': transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std) #[0.00021798351, 0.00016647576, 0.00016200541], [5.786733e-05, 5.2953397e-05, 4.714992e-05]) #mean, std)#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),

    }
    return data_transforms


data_dir = sys.argv[1] 
output_dir = sys.argv[2]

mean_file =  sys.argv[3]
crop_size = int(sys.argv[4])

data_transforms = datatransforms( mean_file, crop_size)

if not  os.path.exists(output_dir):
    os.makedirs(output_dir)

BATCH_SIZE=32    

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']} #, 'val']}

weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             sampler=sampler, num_workers=16)
              for x in ['train']} #, 'val']}

#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
#                                              num_workers=16)
#              for x in ['train']} #, 'val']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train']} #, 'val']}

class_names = image_datasets['train'].classes

#print(image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device found:", device)


def load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger




def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:#, 'val', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            count = 0
            start = time.time()
            all_times = []
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if inputs.size()[0] ==1:
                    continue

                count = count +1  
                # zero the parameter gradients
                optimizer.zero_grad()
                cls_nums = [torch.sum(labels==i) for i in range(len(class_names))]
                w = torch.FloatTensor(cls_nums)
                w = w/torch.sum(w)
                class_weights = 1/torch.FloatTensor(w).to(device)
 
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                       outputs, aux = model(inputs)
                    else:
                       outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
#                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    loss = nn.CrossEntropyLoss(weight=class_weights)(outputs, labels)
#                    loss = criterion(outputs, labels) #, weight=class_weights)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                sample_time = time.time()-start   
                start = time.time()
                all_times.append(sample_time)
                sys.stdout.write('count: {:d}/{:d}, average time:{:f} \r' \
                             .format(count*BATCH_SIZE, len(dataloaders[phase])*BATCH_SIZE, np.mean(np.array(all_times))/BATCH_SIZE ))
                sys.stdout.flush()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        save_checkpoint({
            'epoch': epoch + 1,
            #'arch': args.arch,
            'state_dict': model_ft.state_dict(),
            'best_prec1': epoch_acc,
            'optimizer' : optimizer_ft.state_dict(),
        }, False, os.path.join(output_dir, str(epoch)+'_checkpoint.pth.tar'))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_acc, epoch


num_classes = int(sys.argv[5]) # no of classes
no_of_epochs = int(sys.argv[6]) # no of epochs to train


model_ft = models.inception_v3(pretrained=True)
num_ftrs = model_ft.AuxLogits.fc.in_features
model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,num_classes)

model_ft = model_ft.to(device)

# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 15-25 min on CPU. On GPU though, it takes less than a
# minute.
#
factor = 6
#end_lr = lr_max
end_lr = lr_max = 3*10e-3
#criterion = nn.CrossEntropyLoss()
criterion = ''
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=1.)
step_size = 4*len(dataloaders["train"]) #train_loader)
clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_ft, [clr])

model_ft, best_acc, epoch = train_model(model_ft, criterion, optimizer_ft, scheduler,
                       num_epochs=no_of_epochs)

