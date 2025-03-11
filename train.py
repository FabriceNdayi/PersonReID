#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries
from __future__ import print_function
import os
import time
import random
import zipfile
from itertools import chain
import timm
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import torchvision.transforms as T
import torch.distributed as dist

#from apex.parallel import DistributedDataParallel as DDP
#from apex import amp
from LATransformer.model import ClassBlock, LATransformer
from LATransformer.utils import save_network, update_summary

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

print("Number of CUDA: ",torch.cuda.device_count())
print("device name: ",torch.cuda.get_device_name(0))
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#device = "cuda"
print("cuda: {}".format(torch.cuda.is_available()))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8, 0)

# ### Set Config Parameters

batch_size = 64
num_epochs = 1000
lr = 3e-4
gamma = 0.7
unfreeze_after=2
lr_decay=.8
lmbd = 8

def printsave(*a):
    file = open('test.txt','a')
    print(*a)
    print(*a,file=file)
    file.close()
def printsave2(*a):
    file = open('log2.txt','a')
    print(*a)
    print(*a,file=file)
    file.close()

#### add data augmentation for Corrupted dataset 

transform_train_list = [
    transforms.Resize((224,224), interpolation=3),
    #transforms.RandomRotation(5),
    #transforms.RandomGrayscale(0.1),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform_val_list = [
    transforms.Resize(size=(224,224),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
data_transforms = {
'train': transforms.Compose( transform_train_list ),
'val': transforms.Compose(transform_val_list),
}
image_datasets = {}
data_dir = "//220.69.157.21/Users/KMU/tensor/data/Market_occluded/"
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])
train_loader = DataLoader(dataset = image_datasets['train'], batch_size=batch_size, shuffle=True, drop_last = True )
valid_loader = DataLoader(dataset = image_datasets['val'], batch_size=batch_size, shuffle=True)
class_names = image_datasets['train'].classes
print("number of classes: ",len(class_names))

print("How many devices: ",torch.cuda.device_count())
# ## Load Model
# Load pre-trained ViT
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
#vit_base = nn.DataParallel(vit_base)
vit_base= vit_base.to(device)
vit_base.eval()

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def validate(model, loader, loss_fn):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    epoch_accuracy = 0
    epoch_loss = 0
    end = time.time()
    last_idx = len(loader) - 1

    with torch.no_grad():
        for input, target in tqdm(loader):

            input, target = input.to(device), target.to(device)  #input data and label
            
            output = model(input) # model output (prediction)
            
            score = 0.0
            sm = nn.Softmax(dim=1)
            for k, v in output.items():
                score += sm(output[k])
            _, preds = torch.max(score.data, 1) #predict output class with higher score information
            #print("predictions: ",score)

            loss = 0.0
            for k,v in output.items():
                loss += loss_fn(output[k], target)
            batch_time_m.update(time.time() - end)
            acc = (preds == target.data).float().mean()
            epoch_loss += loss/len(loader)
            epoch_accuracy += acc / len(loader)
            #add scalar
            writer.add_scalar("loss/val", epoch_loss, epoch )
            writer.add_scalar("accuracy/val", epoch_accuracy, epoch)
            
            print(f"Epoch : {epoch+1} - val_loss : {epoch_loss:.4f} - val_acc: {epoch_accuracy:.4f}", end="\r")
    print()    
    metrics = OrderedDict([('val_loss', epoch_loss.data.item()), ("val_accuracy", epoch_accuracy.data.item())])
    return metrics, epoch_loss.data.item()

cnt_print = 0
def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn,
        lr_scheduler=None, saver=None, output_dir='', 
        loss_scaler=None, model_ema=None, mixup_fn=None):

    global cnt_print
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    model.train()
    epoch_accuracy = 0
    epoch_loss = 0
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    running_loss = 0.0
    running_corrects = 0.0
    transform = T.ToPILImage()
    #print("Loader: ", len(loader))
    idx = 0
    #print("Data Loader", loader)
    
    for data, target in tqdm(loader):        
        data, target = data.to(device), target.to(device)  #input data and the target
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        output = model(data)  #model output
          
        score = 0.0    #attention score??
        output_predicts = []
        sm = nn.Softmax(dim=1)
        for k, v, in output.items():  #the k is between 1 ~ 14
            #print("This is the V value", v)
            score += sm(output[k])  #score for each row classifier
            #print("Output prediction: ",output[k])
        _, preds = torch.max(score.data, 1) #class prediction, indices of higher information score
        #print("predictions: ",preds)
        #output_predicts.append(preds) #doesn't make sense
        #print(output_predicts)
        #k,v are key and value then q is query
        loss = 0.0
        for k,v in output.items():
            # print("output[k].shape : ",output[k].shape)
            # print("target.shape : ",target.shape)
            loss += loss_fn(output[k], target)  #loss for all 14 classifier
        loss.backward()
        # print("Loss: ", loss)
        optimizer.step()
        batch_time_m.update(time.time() - end)
#         print(preds, target.data)
        acc = (preds == target.data).float().mean()
        #print("Accuracy: ", acc)
        
#         print(acc)
        epoch_loss += loss/len(loader)
        epoch_accuracy += acc / len(loader)
        
        # add scalar data
        writer.add_scalar("loss/train", epoch_loss, epoch )
        writer.add_scalar("accuracy/train", epoch_accuracy, epoch)


        print(
    f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}"
, end="\r")
    return OrderedDict([('train_loss', epoch_loss.data.item()), ("train_accuracy", epoch_accuracy.data.item())]), epoch_loss.data.item(), epoch_accuracy.data.item()

def freeze_all_blocks(model):
    frozen_blocks = 12
    for block in model.model.blocks[:frozen_blocks]:
        for param in block.parameters():
            param.requires_grad=False
    
def unfreeze_blocks(model, amount= 1):    
    for block in model.model.blocks[11-amount:]:
        for param in block.parameters():
            param.requires_grad=True
    return model

# ## TRAINING LOOP ##
# Create LA Transformer
model = LATransformer(vit_base, lmbd).to(device)
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(),weight_decay=5e-4, lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
freeze_all_blocks(model)
best_acc = 0.0
y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []
print("training...")
output_dir = ""
best_acc = 0
name = "la_with_lmbd_{}".format(lmbd)
val_loss_value =0
train_loss_value = 0
val_loss = []
train_loss = []
acc_arr = []
try:
    os.mkdir("model/" + name)
except:
    pass
output_dir = "model/" + name
unfrozen_blocks = 0

# Load the model checkpoint
#checkpoint = torch.load('weights/model_state_dict.pth')
#print("Checkpoint: ", checkpoint.keys())
#model.state_dict(checkpoint['state_dict'])

#start_epoch = checkpoint['epoch']

for epoch in range(num_epochs):
    if epoch%unfreeze_after==0:
        unfrozen_blocks += 1
        model = unfreeze_blocks(model, unfrozen_blocks)
        optimizer.param_groups[0]['lr'] *= lr_decay 
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Unfrozen Blocks: {}, Current lr: {}, Trainable Params: {}".format(unfrozen_blocks, 
                                                                             optimizer.param_groups[0]['lr'], 
                                                                             trainable_params))
    train_metrics, train_loss_value, acc_value = train_one_epoch(
        epoch, model, train_loader, optimizer, criterion,
        lr_scheduler=None, saver=None)
    eval_metrics, val_loss_value = validate(model, valid_loader, criterion)
    print("#########################################################")
    #print(type(train_loss_value))
    train_loss.append(train_metrics['train_loss'])
#     val_loss.append(round(val_loss_value,2))
    acc_arr.append(acc_value)
    # update summary
    update_summary(epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=True)
    # deep copy the model
    last_model_wts = model.state_dict()
    if eval_metrics['val_accuracy'] > best_acc:
        best_acc = eval_metrics['val_accuracy']
        save_network(model, epoch,name)
        print("SAVED!")

# fig=plt.figure(figsize=(10,10))
# ax1 = plt.subplot(2,1,1)
# x = list(range(len(train_loss)))
# y1 = train_loss
# ax1.plot(x,y1)
# ax1.set_title("train_loss")
# ax1.set_ylabel("train loss")
# ax2 = plt.subplot(2,1,2)
# # x2 = [i for i in range(len(acc_arr))]
# y2 = acc_arr
# ax2.plot(x,y2)
# ax2.set_title("acc_loss")
# ax2.set_ylabel("acc")
# plt.savefig('train_Erase_aff.png')