#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:40:22 2019

@author: riccardo
"""

import torch
import torch.nn as nn
import numpy as np
import warnings

PARAMS_3DEpiBios={    'Categories':3,
            'FilterSize':3, #was 5
            'FiltersNum':np.array([16, 32, 64, 64]),
            'ClassFilters':int(32), 
            'Depth':int(4),
            'Activation':nn.LeakyReLU, 
            'InblockSkip':False,
            'PoolShape':2,
            'BNorm':nn.BatchNorm3d,
            'Conv':nn.Conv3d,
            'Pool':nn.MaxPool3d,
            'Unpool':nn.MaxUnpool3d,
            'SurfaceLossWeight':0,
            'WDecay':0,
            'GenDiceWeight':1,
            'MaskWeight':1,
            'MaskOnly':False,
            'LabelsOnly':False
            }
PARAMS_2DEpiTarget={    'Categories':3,
            'FilterSize':(3,3,1), #was 5
            'FiltersNum':np.array([16, 32, 64, 64]),
            'ClassFilters':int(64), 
            'Depth':int(4),
            'Activation':nn.LeakyReLU, 
            'InblockSkip':False,
            'PoolShape':(2,2,1),
            'BNorm':nn.BatchNorm3d,
            'Conv':nn.Conv3d,
            'Pool':nn.MaxPool3d,
            'Unpool':nn.MaxUnpool3d,
            'SurfaceLossWeight':0,
            'MaskWeight':1,
            'WDecay':0,
            'GenDiceWeight':1,
            'MaskOnly':False,
            'LabelsOnly':False
            }

PARAMS_SKULLNET={    'Categories':None,
            'FilterSize':3, #was 5
            'FiltersNum':np.array([16, 32, 64, 64]),
            'ClassFilters':int(8), 
            'Depth':int(4),
            'Activation':nn.LeakyReLU, 
            'InblockSkip':False,
            'PoolShape':2,
            'BNorm':nn.BatchNorm3d,
            'Conv':nn.Conv3d,
            'Pool':nn.MaxPool3d,
            'Unpool':nn.MaxUnpool3d,
            'SurfaceLossWeight':1,
            'WDecay':0,
            'MaskWeight':1,
            'GenDiceWeight':1,
            'MaskOnly':True,
            'LabelsOnly':False
            }

PARAMS_SKULLNET_2D={    'Categories':None,
            'FilterSize':(3,3,1), #was 5
            'FiltersNum':np.array([16, 32, 64, 64]),
            'ClassFilters':int(8), 
            'Depth':int(4),
            'Activation':nn.LeakyReLU, 
            'InblockSkip':False,
            'PoolShape':(2,2,1),
            'BNorm':nn.BatchNorm3d,
            'Conv':nn.Conv3d,
            'Pool':nn.MaxPool3d,
            'Unpool':nn.MaxUnpool3d,
            'SurfaceLossWeight':1,
            'WDecay':0,
            'GenDiceWeight':1,
            'MaskWeight':1,
            'MaskOnly':True,
            'LabelsOnly':False
            }
            
EPS=1e-10 # log offset to avoid log(0)

torch.set_default_tensor_type('torch.cuda.FloatTensor') # t
torch.backends.cudnn.benchmark = True
#Fix parameters

def FindPad(FilterSize):
    """
    Returns appropriate padding based on filter size
    """
    A=(np.array(FilterSize)-1)/2
    if type(FilterSize)==tuple:
        return tuple(A.astype(int))
    else:
        return int(A)
    
# One convolution step

class OneConv(nn.Module):
    """
    Performs one single convolution: activation of previous layer, batchnorm,
    convolution
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """


    def __init__(self,FilterIn,FilterNum, FilterSize,PAR):
        super(OneConv,self).__init__()
        # One activation - normalization - convolution step
        if FilterIn == 1:
            self.activate= lambda x: x
            self.norm= lambda x: x
        else:
            self.activate=PAR['Activation']()
            self.norm=PAR['BNorm'](int(FilterIn), eps=1e-05, momentum=0.1, affine=True)
        self.conv=PAR['Conv'](int(FilterIn),int(FilterNum),FilterSize,padding=FindPad(FilterSize) )
        
    def forward(self,layer):
        act=self.activate(layer)
        normalize=self.norm(act)
        convolve=self.conv(normalize)
        return convolve

# Bottleneck layer

class Bottleneck(nn.Module):
    """
    Bottleneck layer
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """


    def __init__(self,FilterIn,FilterNum,FilterSize,PAR):
        super(Bottleneck,self).__init__()
        self.norm=PAR['BNorm'](int(FilterIn), eps=1e-05, momentum=0.1, affine=True)
        self.conv=PAR['Conv'](int(FilterIn),int(FilterNum),FilterSize,padding=FindPad(FilterSize) )
        
    def forward(self,layer):
        normalize=self.norm(layer)
        convolve=self.conv(normalize)
        return convolve


# The type of convolution block will be chosen according to what is indicated
# in the parameters dictionary

class SkipConvBlock(nn.Module):
    """
    One full convolution block
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """

    def __init__(self,FilterIn,FilterNum,FilterSize,PAR):
        super(SkipConvBlock,self).__init__()
        self.conv1=OneConv(int(FilterIn),int(FilterNum),FilterSize=FilterSize,PAR=PAR)
        self.conv2=OneConv(int(FilterIn+FilterNum),int(FilterNum),FilterSize=FilterSize,PAR=PAR)
        self.conv3=OneConv(int(FilterIn+FilterNum*2),int(FilterNum),1,PAR=PAR)
        
    def forward(self,BlockInput):
        first=self.conv1(BlockInput)
        fconv=torch.cat((first,BlockInput),1)
        
        second=self.conv2(fconv)
        sconv=torch.cat((first,second,BlockInput),1)
        BlockOut=self.conv3(sconv)
        
        return BlockOut
    
class NoSkipConvBlock(nn.Module):
    """
    One full convolution block
    FilterIn is the number of input channels, FilterNum output channels,
    filters are of size FilterSize
    """

    def __init__(self,FilterIn,FilterNum,FilterSize,PAR):
        super(NoSkipConvBlock,self).__init__()
        self.conv1=OneConv(int(FilterIn),int(FilterNum),FilterSize=FilterSize,PAR=PAR)
        self.conv2=OneConv(int(FilterNum),int(FilterNum),FilterSize=FilterSize,PAR=PAR)
        self.conv3=OneConv(int(FilterNum),int(FilterNum),1,PAR=PAR)
        
    def forward(self,BlockInput):
        first=self.conv1(BlockInput)
        
        second=self.conv2(first)
        BlockOut=self.conv3(second)
        
        return BlockOut


class SkullNet(nn.Module):
    
    def __init__(self,PARAMS):
        super(SkullNet,self).__init__()
        self.PARAMS=PARAMS
        warnings.filterwarnings('ignore', '.*when mode=trilinear is changed.*')
        if PARAMS['InblockSkip']:
            ConvBlock=SkipConvBlock
            self.skipper=True
        else:
            ConvBlock=NoSkipConvBlock
            self.skipper=False
        self.layers=nn.ModuleDict()
        self.layers['Dense_Down'+str(0)]=ConvBlock(1,PARAMS['FiltersNum'][0],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        self.layers['Pool'+str(0)]=PARAMS['Conv'](PARAMS['FiltersNum'][0], PARAMS['FiltersNum'][0], PARAMS['PoolShape'],stride=PARAMS['PoolShape'])
        
        for i in range(1,PARAMS['Depth']):
            self.layers['Dense_Down'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i-1],PARAMS['FiltersNum'][i],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            self.layers['Pool'+str(i)]=PARAMS['Conv'](PARAMS['FiltersNum'][i], PARAMS['FiltersNum'][i], 3)
        
        self.layers['Bneck']=Bottleneck(PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        

        self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][-1]+PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-2],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        for i in reversed(range(1,PARAMS['Depth']-1)):
            

            
            self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i]+PARAMS['FiltersNum'][i],PARAMS['FiltersNum'][i-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            
            
        
        self.layers['Dense_Up'+str(0)]=ConvBlock(PARAMS['FiltersNum'][0]+PARAMS['FiltersNum'][0],PARAMS['ClassFilters'],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        

        self.layers['BinaryMask'] = PARAMS['Conv'](PARAMS['ClassFilters'],1,1)
        self.sigmoid=nn.Sigmoid()
            
            
    def forward(self,MRI):
        
        dense={}
        dense[0] = self.layers['Dense_Down'+str(0)](MRI)
        dense[1] = self.layers['Pool'+str(0)](dense[0])
        
        for i in range(1,self.PARAMS['Depth']):
            
            dense[i] = self.layers['Dense_Down'+str(i)](dense[i])
            dense[i+1] = self.layers['Pool'+str(i)](dense[i])
        
        BotNeck = self.layers['Bneck'](dense[i+1])
        
        Updense={}
        Unpool={}
        
        Unpool[i] = torch.nn.functional.interpolate(BotNeck,size=dense[i].size()[2:],mode='trilinear')
        cat=torch.cat([Unpool[i],dense[i]],dim=1)
        Updense[i] = self.layers['Dense_Up'+str(i)](cat)
        
        for i in reversed(range(self.PARAMS['Depth']-1)):
            
            Unpool[i]=torch.nn.functional.interpolate(Updense[i+1],size=dense[i].size()[2:],mode='trilinear')
            cat=torch.cat([Unpool[i],dense[i]],dim=1)
            Updense[i]=self.layers['Dense_Up'+str(i)](cat)
            
        MonoClass=self.layers['BinaryMask'](Updense[0])
        
        Mask=self.sigmoid(MonoClass)
        
        return Mask, []

class MUnet(nn.Module):
    """
    Network definition, without framing connections. 
    Returns (Mask,Classes)
    Generated based on parameters
    """
    
    def __init__(self,PARAMS):
        super(MUnet,self).__init__()
        self.PARAMS=PARAMS
        if PARAMS['InblockSkip']:
            ConvBlock=SkipConvBlock
            self.skipper=True
        else:
            ConvBlock=NoSkipConvBlock
            self.skipper=False
        self.layers=nn.ModuleDict()
        self.layers['Dense_Down'+str(0)]=ConvBlock(1,PARAMS['FiltersNum'][0],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        self.layers['Pool'+str(0)]=PARAMS['Pool'](PARAMS['PoolShape'],return_indices=True) 
        
        for i in range(1,PARAMS['Depth']):
            self.layers['Dense_Down'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i-1],PARAMS['FiltersNum'][i],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            self.layers['Pool'+str(i)]=PARAMS['Pool'](PARAMS['PoolShape'],return_indices=True) 
        
        self.layers['Bneck']=Bottleneck(PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        self.layers['Up'+str(i)]=PARAMS['Unpool'](PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][-1]+PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-2],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        for i in reversed(range(1,PARAMS['Depth']-1)):
            
            self.layers['Up'+str(i)]=PARAMS['Unpool'](PARAMS['PoolShape'])
            
            self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i]*2,PARAMS['FiltersNum'][i-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            
            
        self.layers['Up'+str(0)]=PARAMS['Unpool'](PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(0)]=ConvBlock(PARAMS['FiltersNum'][0]*2,PARAMS['ClassFilters'],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        

        self.layers['Classifier']=PARAMS['Conv'](PARAMS['ClassFilters'],PARAMS['Categories'],1) #classifier layer
        self.layers['BinaryMask']=PARAMS['Conv'](PARAMS['ClassFilters'],1,1) #binary mask classifier
        self.softmax=nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()
        #self.sigmoid1=nn.Sigmoid()
            
            
    def forward(self,MRI):
        pools={}
        dense={}
        dense[0] = self.layers['Dense_Down'+str(0)](MRI)
        dense[1], pools[0] = self.layers['Pool'+str(0)](dense[0])
        
        for i in range(1,self.PARAMS['Depth']):
            dense[i] = self.layers['Dense_Down'+str(i)](dense[i])
            dense[i+1], pools[i] = self.layers['Pool'+str(i)](dense[i])
        
        BotNeck = self.layers['Bneck'](dense[i+1])
        
        Updense={}
        Unpool={}
        
        Unpool[i] = self.layers['Up'+str(i)](BotNeck,pools[i],output_size=dense[i].size())
        cat=torch.cat([Unpool[i],dense[i]],dim=1)
        Updense[i] = self.layers['Dense_Up'+str(i)](cat)
        
        for i in reversed(range(self.PARAMS['Depth']-1)):
            
            Unpool[i]=self.layers['Up'+str(i)](Updense[i+1],pools[i],output_size=dense[i].size())
            cat=torch.cat([Unpool[i],dense[i]],dim=1)
            Updense[i]=self.layers['Dense_Up'+str(i)](cat)
            
        MultiClass=self.layers['Classifier'](Updense[0])
        MonoClass=self.layers['BinaryMask'](Updense[0])
        #MonoClassRef=self.layers['BinaryMask1'](Updense[0])
        
        Mask=self.sigmoid(MonoClass)
        
        
        return Mask, self.softmax(MultiClass)#, self.sigmoid1(MonoClassRef)

class HPonly(nn.Module):
    """
    Network definition, without framing connections. 
    Returns (Mask,Classes)
    Generated based on parameters
    """
    
    def __init__(self,PARAMS):
        super(HPonly,self).__init__()
        self.PARAMS=PARAMS
        if PARAMS['InblockSkip']:
            ConvBlock=SkipConvBlock
            self.skipper=True
        else:
            ConvBlock=NoSkipConvBlock
            self.skipper=False
        self.layers=nn.ModuleDict()
        self.layers['Dense_Down'+str(0)]=ConvBlock(1,PARAMS['FiltersNum'][0],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        self.layers['Pool'+str(0)]=PARAMS['Pool'](PARAMS['PoolShape'],return_indices=True) 
        
        for i in range(1,PARAMS['Depth']):
            self.layers['Dense_Down'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i-1],PARAMS['FiltersNum'][i],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            self.layers['Pool'+str(i)]=PARAMS['Pool'](PARAMS['PoolShape'],return_indices=True) 
        
        self.layers['Bneck']=Bottleneck(PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        self.layers['Up'+str(i)]=PARAMS['Unpool'](PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][-1]+PARAMS['FiltersNum'][-1],PARAMS['FiltersNum'][-2],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        
        for i in reversed(range(1,PARAMS['Depth']-1)):
            
            self.layers['Up'+str(i)]=PARAMS['Unpool'](PARAMS['PoolShape'])
            
            self.layers['Dense_Up'+str(i)]=ConvBlock(PARAMS['FiltersNum'][i]*2,PARAMS['FiltersNum'][i-1],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
            
            
        self.layers['Up'+str(0)]=PARAMS['Unpool'](PARAMS['PoolShape'])
        self.layers['Dense_Up'+str(0)]=ConvBlock(PARAMS['FiltersNum'][0]*2,PARAMS['ClassFilters'],FilterSize=PARAMS['FilterSize'],PAR=PARAMS)
        

        self.layers['Classifier']=PARAMS['Conv'](PARAMS['ClassFilters'],PARAMS['Categories'],1) #classifier layer

        self.softmax=nn.Softmax(dim=1)
            
            
    def forward(self,MRI):
        pools={}
        dense={}
        dense[0] = self.layers['Dense_Down'+str(0)](MRI)
        dense[1], pools[0] = self.layers['Pool'+str(0)](dense[0])
        
        for i in range(1,self.PARAMS['Depth']):
            dense[i] = self.layers['Dense_Down'+str(i)](dense[i])
            dense[i+1], pools[i] = self.layers['Pool'+str(i)](dense[i])
        
        BotNeck = self.layers['Bneck'](dense[i+1])
        
        Updense={}
        Unpool={}
        
        Unpool[i] = self.layers['Up'+str(i)](BotNeck,pools[i],output_size=dense[i].size())
        cat=torch.cat([Unpool[i],dense[i]],dim=1)
        Updense[i] = self.layers['Dense_Up'+str(i)](cat)
        
        for i in reversed(range(self.PARAMS['Depth']-1)):
            
            Unpool[i]=self.layers['Up'+str(i)](Updense[i+1],pools[i],output_size=dense[i].size())
            cat=torch.cat([Unpool[i],dense[i]],dim=1)
            Updense[i]=self.layers['Dense_Up'+str(i)](cat)
            
        MultiClass=self.layers['Classifier'](Updense[0])

        
        
        return  [], self.softmax(MultiClass)