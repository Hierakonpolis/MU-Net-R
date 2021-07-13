#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, torch, torchvision
import helper as H
import models as M
import network as N
from sklearn.model_selection import train_test_split, KFold
import numpy as np

# Make GT masks
HPCf='/media/Olowoo/hippocampus'
NewBMfolder=HPCf+'/newmasks'
ResultsFolder=HPCf+'/results'
ValFolder=HPCf+'/ValEB'
vols=H.DatasetDict(HPCf)

for ID, data in vols.items():
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
inferences=[]
loader=0
    
        
    

del loader, infdata, SkullStripDataset, trainloader, testloader, inferences


pholds=[['1012', '1017'],
['1036', '1031'],
['1091', '1099'],
['1045', '1024'],
['1096', '1035'],
['1008', '1038']]

EpiNet = N.MUnet
transforms = torchvision.transforms.Compose([scaler,normalizator,tensorize])
inf_transforms = torchvision.transforms.Compose([normalizator,tensorize])
EpiBiosTrainingData = H.FullDataset(HPCf,newmasks=HPCf+'/newmasks',transform=transforms)
EpiBiosInference = H.FullDataset(HPCf,newmasks=HPCf+'/newmasks',transform=inf_transforms)


def GetSplits(realindexes,toextract):
    idxs=[]
    for k in toextract:
        idxs.append(realindexes[k])
    
    return idxs

for outer_fold in pholds:
    
    ToTrainIndexes = EpiBiosTrainingData.IDindexes(outer_fold,notinlist=True)
    ToTestIndexes =  EpiBiosTrainingData.IDindexes(outer_fold,notinlist=False)
    
    i=0
    models=[]
    datasplitter=KFold(6,shuffle=True)
    
    for train_idxs, test_idxs  in datasplitter.split(range(len(ToTrainIndexes))):
        
        train_idxs = GetSplits(ToTrainIndexes,train_idxs)
        test_idxs = GetSplits(ToTrainIndexes,test_idxs)
        
        EBModelSave=HPCf+'/models/EBmodVal_'+str(i)+'.pth'
        i+=1
        trainloader = torch.utils.data.DataLoader(EpiBiosTrainingData, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(train_idxs),num_workers=23)
        testloader = torch.utils.data.DataLoader(EpiBiosTrainingData, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(test_idxs),num_workers=23)
        
        HasConverged=False
        EBModel=0
        while not HasConverged:
            del EBModel
            EBModel=M.Segmentation(EpiNet,savefile=None,parameters=N.PARAMS_3DEpiBios,device='cuda')
            HasConverged=EBModel.train(trainloader,
                                        testloader,
                                        max_epochs=250,
                                        patience=10,
                                        max_time=60*60*2,
                                        saveprogress='/media/ramdisk/EBmodp.pth',
                                        savebest=EBModelSave)
        
        models.append(EBModelSave)
        
    LastTest = torch.utils.data.DataLoader(EpiBiosInference, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(ToTestIndexes),num_workers=23)
        
    
    transforms = torchvision.transforms.Compose([normalizator,tensorize])
    dataset=H.SSinferenceDataset(HPCf,transforms)
    inferences=[]
    loader=LastTest
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
        
        filename=os.path.join(ValFolder,ID+'_Mask.nii.gz')
        H.MakeNii(temp,prediction[ID][-1],filename)
        
        
        templ [np.where(templ== np.amax(templ,axis=1))] = 1
        templ[templ!=1]=0
        
        for idx, roiname in enumerate(['_Ipsi','_Contra','_Background']):
            filename=os.path.join(ValFolder,ID+roiname+'.nii.gz')
            R=templ[:,idx,...]
            R=R.reshape(R.shape[-3:])
            R=H.LargestComponent(R).astype(float)
            R=H.FillHoles(R).astype(float)
            H.MakeNii(R,prediction[ID][-1],filename)
        
        
