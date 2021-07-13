#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, torch, torchvision
import helper as H
import models as M
import network as N
from sklearn.model_selection import KFold
import numpy as np

# Make GT masks
HPCf='/media/Olowoo/hippocampus'
NewBMfolder=HPCf+'/newmasks'
ResultsFolder=HPCf+'/results'
vols=H.DatasetDict(HPCf)

for ID, data in vols.items():
    # path='/'+os.path.join(*data['Sparse Mask'].split('/')[:-1])
    # outpath=os.path.join(path,'wholebrain.nii')
    H.AltRefine(data['Sparse Mask'],data['Preliminary Mask'],iters=2)

#### ensemble train, ensamble test, ensemble eval for ss and hpc



scaler = H.Rescale(1.05,0.95,0.5,dim=3)
normalizator = H.Normalizer()
tensorize = H.ToTensor()
transforms = torchvision.transforms.Compose([scaler,normalizator,tensorize])
SkullStripDataset=H.SkullStripDataset(HPCf,transforms)
par=N.PARAMS_SKULLNET
datasplitter=KFold(6)
SSmodels=[]
for train_idxs, test_idxs  in datasplitter.split(range(len(SkullStripDataset))):
    # train_idxs, test_idxs = train_test_split(np.arange(len(SkullStripDataset)), test_size=0.15)

    trainloader = torch.utils.data.DataLoader(SkullStripDataset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(train_idxs),num_workers=23)
    testloader = torch.utils.data.DataLoader(SkullStripDataset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(test_idxs),num_workers=23)
    
    network=N.SkullNet
    SSsave=HPCf+'/models/PreSS'+str(test_idxs[0])+'.pth'
    

    SSModel=M.Segmentation(network,savefile=SSsave,parameters=par,device='cuda')
   

    # SSModel.train(trainloader,
    #             testloader,
    #             max_epochs=15000,
    #             patience=20,
    #             max_time=60*60,
    #             saveprogress='/media/Olowoo/hippocampus/models/PreSSp.pth',
    #             savebest=SSsave)
    SSmodels.append(SSsave)
    print(SSModel.opt['TestDices'][-1])


infdata=H.SSinferenceDataset(HPCf,transforms)
# transforms = torchvision.transforms.Compose([normalizator,tensorize])
inferences=[]
loader=0
# for sfile in SSmodels:
#     network=N.SkullNet
#     SSModel=M.Segmentation(network,savefile=sfile,parameters=par,device='cuda')
    
#     loader=torch.utils.data.DataLoader(infdata, batch_size=1,num_workers=23)
    
#     inferences.append(SSModel.inferece(loader))
    # '/media/Olowoo/hippocampus/newmasks',namepedix='mask'
    # from helper import MakeNii, LargestComponent, FillHoles
    
# for ID in inferences[0]:
#     temp=0
#     for prediction in inferences:
#         temp+=prediction[ID][0]/len(inferences)
#     temp[temp>=0.5]=1
#     temp[temp!=1]=0
#     temp=temp.reshape(temp.shape[-3:])
#     temp=H.LargestComponent(temp)
#     temp=H.FillHoles(temp)
    
#     filename=os.path.join(NewBMfolder,ID+'mask.nii.gz')
#     H.MakeNii(temp,prediction[ID][-1],filename)
    
    
        
    

del loader, infdata, SkullStripDataset, trainloader, testloader, inferences




EpiNet = N.MUnet
transforms = torchvision.transforms.Compose([scaler,normalizator,tensorize])
EpiBiosTrainingData = H.FullDataset(HPCf,newmasks=HPCf+'/newmasks',transform=transforms)
i=0
models=[]
for train_idxs, test_idxs  in datasplitter.split(range(len(EpiBiosTrainingData))):
    
    EBModelSave='/media/ramdisk/EBmod'+str(i)+'.pth'
    trainloader = torch.utils.data.DataLoader(EpiBiosTrainingData, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(train_idxs),num_workers=23)
    testloader = torch.utils.data.DataLoader(EpiBiosTrainingData, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(test_idxs),num_workers=23)
    EBModel=M.Segmentation(EpiNet,savefile=None,parameters=N.PARAMS_3DEpiBios,device='cuda')    
    EBModel.train(trainloader,
                testloader,
                max_epochs=15000,
                patience=20,
                max_time=60*60*2,
                saveprogress='/media/ramdisk/EBmodp.pth',
                savebest=EBModelSave)
    models.append(EBModelSave)
del trainloader, testloader, EpiBiosTrainingData

transforms = torchvision.transforms.Compose([normalizator,tensorize])
dataset=H.SSinferenceDataset(HPCf,transforms)
inferences=[]
loader=torch.utils.data.DataLoader(dataset, batch_size=1,num_workers=0)
for sfile in models:
    SSModel=M.Segmentation(EpiNet,savefile=sfile,parameters=N.PARAMS_3DEpiBios,device='cuda')
    inferences.append(SSModel.inferece(loader))

for ID in inferences[0]:
    temp=0
    templ=0
    for prediction in inferences:
        temp+=prediction[ID][0]/len(inferences)
        templ+=prediction[ID][1]/len(inferences)
    
    temp[temp>=0.5]=1
    temp[temp!=1]=0
    temp=temp.reshape(temp.shape[-3:])
    temp=H.LargestComponent(temp)
    temp=H.FillHoles(temp)
    
    filename=os.path.join(ResultsFolder,ID+'_Mask.nii.gz')
    H.MakeNii(temp,prediction[ID][-1],filename)
    
    
    templ [np.where(templ== np.amax(templ,axis=1))] = 1
    templ[templ!=1]=0
    
    for idx, roiname in enumerate(['_Ipsi','_Contra','_Background']):
        filename=os.path.join(ResultsFolder,ID+roiname+'.nii.gz')
        R=templ[:,idx,...]
        R=R.reshape(R.shape[-3:])
        R=H.LargestComponent(R).astype(float)
        R=H.FillHoles(R).astype(float)
        H.MakeNii(R,prediction[ID][-1],filename)
    
    i+=1 
# train_idxs, test_idxs = train_test_split(np.arange(len(EpiBiosTrainingData)), test_size=0.15)