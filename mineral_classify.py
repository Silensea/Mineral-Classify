#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:16:59 2020

@author: silence
"""

## Lib Import
# pytorch version: 1.5.0+cu101

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import time
import os
import copy
import glob

## Deive Confirm
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

## Data Preprocess and Import
data_trans = {
    'train': transforms.Compose([
#        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(), # Random flip left or right
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
#        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# dataset path
data_dir = 'Data_all_256'

# build datasets
mineral_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_trans[x])
                                          for x in ['train', 'val']}

# build dataloader
dataloader = {x: torch.utils.data.DataLoader(mineral_datasets[x], 
                                             batch_size = 32, 
                                             shuffle = True, 
                                             num_workers = 4)
                                          for x in ['train', 'val']}

dataset_size = {x: len(mineral_datasets[x]) for x in ['train', 'val']}

labels = mineral_datasets['train'].classes

# batch image show (for tensor)
def imshow(tens, fig_title = None):
    tens = tens.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tens = std * tens + mean # normalization return
    tens = np.clip(tens, 0, 1)
    plt.imshow(tens)
    if fig_title is not None:
        plt.title(fig_title)
        

# get a batch
inputs, labels = next(iter(dataloader['train']))

# grid view of this batch
one_batch = torchvision.utils.make_grid(inputs)
plt.figure(figsize=(14, 7))
imshow(one_batch)

## Loss Curve Param
MAXEPOCH = 100
train_loss = np.zeros((1 , MAXEPOCH))
val_loss = np.zeros((1 , MAXEPOCH))
acc_train = np.zeros((1 , MAXEPOCH))
acc_valid = np.zeros((1 , MAXEPOCH))

## Train process build
def train_model(model, MAXEPOCH, optimizer, criterion, scheduler):
    start = time.time() # train start time
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(MAXEPOCH):
        print('Epoch {}/{} '.format(epoch + 1, MAXEPOCH ))
        
        for mode in ['train', 'val']: # modes of model
            if mode == 'train':
                model.train() # training mode
            else:
                model.eval() # evaluate mode
                
            iter_loss = 0
            iter_corr = 0
                
            # iteration in data
            for inputs, labels in dataloader[mode]:
                inputs = inputs.to(device)
                labels = labels.to(device) # data to device
                
                optimizer.zero_grad() # zero the computed gradient in optimizer
                
                # forward
                with torch.set_grad_enabled(mode == 'train'):
                    outputs = model(inputs) # train
                    _, preds = torch.max(outputs, 1) # get the predict result
                    loss = criterion(outputs, labels) # calculate loss 
                    
                    # backward
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                iter_loss += loss.item() * inputs.size(0)
                iter_corr += torch.sum(preds == labels.data)
            
            if mode == 'train':
                scheduler.step() # learning rate change
                
            epoch_loss = iter_loss / dataset_size[mode]           
            epoch_acc = iter_corr.double() / dataset_size[mode]
            
            print('{} Loss: {:.4f} Accuracy: {:.4f} \n'.format(mode, epoch_loss, epoch_acc))
            
            if mode == 'train':
                acc_train[0, epoch] += epoch_acc
                train_loss[0, epoch] += epoch_loss
            else:
                acc_valid[0, epoch] += epoch_acc
                val_loss[0, epoch] += epoch_loss                
            
            # deep copy the model to save the best weights
            if mode == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts =copy.deepcopy(model.state_dict())
                
    time_duration = time.time() - start
    print('trainintg finished in {:.0f} min {:.0f}sec '.format( time_duration // 60, time_duration % 60))
    print('Best validation accuracy: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    

## Loss Principle
criterion = nn.CrossEntropyLoss()

## Get the pretrained model
#model_ft = models.vgg19(pretrained = True).to(device)
model_ft = models.resnet18(pretrained = True).to(device)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)

## Optimizer 
optimizer_ft = optim.SGD(model_ft.parameters(), lr = 1e-3, momentum = 0.9)

## Scheduler 
# decrease lr every 10 epoch by a factor of 0.1
exp_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)

## Train
model_ft = train_model(model_ft, MAXEPOCH, optimizer_ft, criterion, exp_scheduler)


## Visualize results
def visualize_model(model, aim_num):
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, label_name) in enumerate(dataloader['val']):
            inputs = inputs.to(device)
            label_name = label_name.to(device)
            image_num_now = 0
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            plt.figure(figsize=(15, 15))            
            for j in range(inputs.size()[0]):
                image_num_now += 1
                
                ax = plt.subplot(aim_num // 2, 3, image_num_now)
#                ax.axis('off')
                plt.ylabel('Actual: {}'.format(labels[label_name[j]]))
                ax.set_title('predicted: {}'.format(labels[preds[j]]))
                imshow(inputs.cpu().data[j])
                
                if image_num_now == aim_num:
                    model.train(mode = model.training)
                    return
            model.train(mode = model.training)

visualize_model(model_ft, 9)

## Loss Curve Plot
plt.figure(figsize=(14, 7))
plt.subplot(2, 2, 1)
plt.plot(np.arange(0, MAXEPOCH), train_loss[0,0:MAXEPOCH])
plt.xlabel('epoch')
plt.ylabel('Train Loss')
plt.subplot(2, 2, 2)
plt.plot(np.arange(0, MAXEPOCH), val_loss[0,0:MAXEPOCH])
plt.xlabel('epoch')
plt.ylabel('Validation Loss')
plt.subplot(2, 2, 3)
plt.plot(np.arange(0, MAXEPOCH), acc_train[0,0:MAXEPOCH])
plt.xlabel('epoch')
plt.ylabel('Train Accuracy')
plt.subplot(2, 2, 4)
plt.plot(np.arange(0, MAXEPOCH), acc_valid[0,0:MAXEPOCH])
plt.xlabel('epoch')
plt.ylabel('Validation Accuracy')

## Weights Save
torch.save(model_ft.state_dict(),'./Model_weights/res18/res_weights_all_256.pth')

## Weights Load
model_ft.load_state_dict(torch.load('./Model_weights/res18/res_weights.pth'))

## Test
flag =0;
test_label=[];
for test_path in glob.glob("/media/silence/Silensea/深度学习/Final_Homework/矿物分类/Data_other3_256/*.png"):
    # get label name
    (filepath, tempfilename) = os.path.split(test_path)
    (filename, extension) = os.path.splitext(tempfilename)
    
    
    test_trans = transforms.Compose([
            transforms.Resize(192),# resize the image
#            transforms.CenterCrop(192),
            
            transforms.ToTensor(), # image => tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_image = Image.open(test_path).convert('RGB')    
    test_image = test_trans(test_image).to(device, torch.float)
    test_image = test_image.reshape(1, 3, 192, 256)
    if flag == 0:
        temp = test_image
        flag = 1
        test_label.append(filename)
    else:
        temp = torch.cat([temp, test_image], dim = 0)
        test_label.append(filename)

test_output = model_ft(temp)
_, preds = torch.max(test_output, 1)

image_num_now = 0
plt.figure(figsize=(15, 15))            
for j in range(0,temp.shape[0]):
    image_num_now += 1
    
    ax = plt.subplot(9 // 2, 3, image_num_now)
    plt.ylabel('Actual: {}'.format(test_label[j]))
    ax.set_title('predicted: {}'.format(labels[preds[j]]))
    imshow(temp[j,:,:,:].reshape(-1,192,256))

## heatmap
result = test_output.detach().numpy()
result = pd.DataFrame(result)
result = result.rename(columns = pd.Series(labels), index = pd.Series(test_label))

fig, ax = plt.subplots(figsize = (9,9))
sns.heatmap(result, annot=True, vmax=3,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="rainbow")