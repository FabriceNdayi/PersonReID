import timm
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
#from sklearn.model_selection import train_test_split

import os
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
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = "cuda"


print("Number of CUDA: ",torch.cuda.device_count())
print("device name: ",torch.cuda.get_device_name(0))
os.environ['CUDA_VISIBLE_DEVICES']='0'
#device = "cuda"
print("cuda: {}".format(torch.cuda.is_available()))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8, 0)

# weights initialization
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
        
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x
        
class LATransformer(nn.Module):
    def __init__(self, model, lmbd ):
        super(LATransformer, self).__init__()
        
        self.class_num = 751
        self.part = 14 # We cut the pool5 to sqrt(N) parts
        self.num_blocks = 12 #Encoder total block
        self.model = model
        self.model.head.requires_grad_ = False 
        self.cls_token = self.model.cls_token  #torch.Size([1, 1, 768])
        self.pos_embed = self.model.pos_embed  #torch.Size([1, 197, 768])I am frequently 192, 384, 768
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,768)) #AdaptiveAvgPool2d(output_size=(14, 768))
        self.dropout = nn.Dropout(p=0.5)
        self.lmbd = lmbd
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(768, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))
            

    def forward(self,x):
        x = self.model.patch_embed(x)  #B, 196, 768        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  #B, 1, 768 
        x = torch.cat((cls_tokens, x), dim=1) #B, 197, 768        
        x = self.model.pos_drop(x + self.pos_embed) #B, 197, 768 position + patch embedding
        
        # Feed forward through transformer blocks
        # use block for checking important features and pruning module
        # use threshold to prune the less import features
        # what is x and weights, use weights to make attention mask and the threshold
        # make a list for visualization

        # attention weights
        attention_weights = []
        layer_pruning_mask = []

        for i in range(self.num_blocks):
            x = self.model.blocks[i](x)  # block output and block weight attention
           
        x = self.model.norm(x) #B, 197, 768
        #### weight_attn = torch.Size([4, 12, 197, 197])
        # extract the cls token
        cls_token_out = x[:, 0].unsqueeze(1)  #B, 1, 768]
        # Average pool
        x = self.avgpool(x[:, 1:]) #B, 14, 768
        # Add global cls token to each local token (Globally Enhanced Local Tokens )
        for i in range(self.part):
            out = torch.mul(x[:, i, :], self.lmbd)  #B, 768
            x[:,i,:] = torch.div(torch.add(cls_token_out.squeeze(),out), 1 + self.lmbd)  #L as the averaged GELT
        # Locally aware network
        part = {}
        predict = {}
        for i in range(self.part):
            part[i] = x[:,i,:]
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])
        return predict
    
class LATransformerTest(nn.Module):
    def __init__(self, model, lmbd ):
        super(LATransformerTest, self).__init__()
        
        self.class_num = 751
        self.part = 14 # We cut the pool5 to sqrt(N) parts
        self.num_blocks = 12
        self.model = model
        self.model.head.requires_grad_ = False 
        self.cls_token = self.model.cls_token
        self.pos_embed = self.model.pos_embed
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,192))
        self.dropout = nn.Dropout(p=0.5)
        self.lmbd = lmbd
#         for i in range(self.part):
#             name = 'classifier'+str(i)
#             setattr(self, name, ClassBlock(768, self.class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        

    def forward(self,x):
        
        # Divide input image into patch embeddings and add position embeddings
        x = self.model.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) 
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.pos_embed)
        
        # Feed forward through transformer blocks
        for i in range(self.num_blocks):
            x = self.model.blocks[i](x)
        x = self.model.norm(x)
        
        # extract the cls token
        cls_token_out = x[:, 0].unsqueeze(1)
        
        # Average pool
        x = self.avgpool(x[:, 1:])
        
        # Add global cls token to each local token 
#         for i in range(self.part):
#             out = torch.mul(x[:, i, :], self.lmbd)
#             x[:,i,:] = torch.div(torch.add(cls_token_out.squeeze(),out), 1+self.lmbd)

        return x.to(device)