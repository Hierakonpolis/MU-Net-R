#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:39:56 2020
.
@author: cat
"""

import helper as H
# import Helper2 as H2
import numpy as np
import torch, tqdm#,pickle
import torchvision
from network import MUnet, PARAMS_Actual2D
import os
# from radam import RAdam

# dataset=H.SkullStripDataset(HPCfolder, None,fakedata=FakeData)

# LabelVoxels=0
# TotalVoxels=0

# for i in range(len(dataset)):
#     sample=dataset[i]
#     _, mask= sample['MRI'], sample['Mask']
    
#     LabelVoxels+=mask.sum()
#     TotalVoxels+=np.prod(mask.shape)

# del dataset
# LabFreq=LabelVoxels/TotalVoxels

dataroot='/media/Olowoo/hippocampus/epitarget_all'
HPCfolder='/media/Olowoo/hippocampus'

them=['002','007','009','012','026','029']

nets={}
mnets={}
#PARAMS_2D_NoSkip['FiltersNum']=np.array([16, 32, 64])
#PARAMS_2D_NoSkip['Depth']=3

newpath=os.path.join(HPCfolder,'resultsET')

if not os.path.isdir(newpath): os.mkdir(newpath)
modelsfolder=os.path.join(HPCfolder, 'models')
saveprogress=os.path.join(modelsfolder,'latest.tar')
tensorboards=os.path.join(HPCfolder,'TBX')

decay=0.00
maxtime=20 *60*60 # hours
EPOCHS=20*60*12
patience=5000
testsize=0.15
Bsize=1

normalizator = H.Normalizer()
tensorize = H.ToTensor()
transforms = torchvision.transforms.Compose([ normalizator, tensorize])
dataset=H.Epitarget_inference(transforms)

for ii in range(6):
# for TSubj in them:
    ETModelSave=HPCfolder+'/models/ETmod'+str(ii)+'.pth'
    nets[ETModelSave]=[]
    mnets[ETModelSave]=[]
    Network=MUnet
    # name='HPU_dist'+TSubj
    
    data = torch.utils.data.DataLoader(dataset, batch_size=Bsize,shuffle=False,num_workers=23)
    
    # runfolder=tensorboards+name
    savefile=ETModelSave
    save=torch.load(savefile)
    PAR=PARAMS_Actual2D
    
    
    UN=Network(PAR).cuda()
    UN.load_state_dict(save['model_state_dict'])
    
    TotalTime=0
    epoch=0
    bestloss=0
    
    UN.eval()
    
    results={}
    names=[]
    
    for i, sample in tqdm.tqdm(enumerate(data),total=len(data)):
        #print(type(sample['Path'][0]))
        Mask, Labels = UN(sample['MRI'].cuda())
        sname=sample['Sample'][0]
        names.append(sname)
        nets[ETModelSave].append(Labels.detach().cpu().numpy())
        mnets[ETModelSave].append(Mask.detach().cpu().numpy())
        
logits=[]
mlogits=[]
for k in range(len(data)):
    a=0
    m=0
    for TSubj in nets:
        a+=nets[TSubj][k]
        m+=mnets[TSubj][k]
    logits.append(a/len(them))
    mlogits.append(m/len(them))

for ix in range(len(names)):
    Labels=logits[ix]
    Mask=mlogits[ix]
    sname=names[ix]
    Labels[np.where(Labels== np.amax(Labels,axis=1))] = 1
    Labels[Labels!=1]=0
    
    shape=Labels.shape[-3:]
    
    Mask[Mask<0.5]=0
    Mask[Mask!=0]=1
    Mask=Mask.reshape(shape)
    Mask=H.LargestComponent(Mask)
    Mask=H.FillHoles(Mask)
    
    
    Ipsi, Contra, BG = [ k.reshape(shape) for k in np.split(Labels,3,axis=1)]
    
    H.MakeNii(H.LargestComponent(Ipsi), sample['Path'][0], os.path.join(newpath,sname+'_Ipsi.nii.gz')) 
    H.MakeNii(H.LargestComponent(Contra), sample['Path'][0], os.path.join(newpath,sname+'_Contra.nii.gz')) 
    H.MakeNii(BG, sample['Path'][0], os.path.join(newpath,sname+'_Background.nii.gz'))
    H.MakeNii(Mask, sample['Path'][0], os.path.join(newpath,sname+'_Mask.nii.gz'))
