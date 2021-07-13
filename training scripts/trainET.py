#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:43:25 2019

@author: riccardo
"""

import helper as H
import numpy as np
import torch, tqdm
import torchvision
from network import PARAMS_Actual2D, MUnet
from models import DiceLoss, GeneralizedDice
import time
import os
from radam import RAdam
from sklearn.model_selection import KFold

dataroot='/media/Olowoo/hippocampus/Epitarget_HC_lining'
HPCfolder='/media/Olowoo/hippocampus'


SaveModels=True

if SaveModels:
    modelsfolder=os.path.join(HPCfolder, 'models')
    if not os.path.isdir(modelsfolder): os.mkdir(modelsfolder)
    saveprogress=os.path.join(modelsfolder,'latest.tar')
    tensorboards=os.path.join(HPCfolder,'TBX')
    if not os.path.isdir(tensorboards): os.mkdir(tensorboards)

decay=0.00
maxtime=24 *60*60 # hours
EPOCHS=20*60*60
patience=400
testsize=0.15
Bsize=1
BoundaryWeight=2
Network=MUnet
datasplitter=KFold(6) 

SSmodels=[]

PAR=PARAMS_Actual2D

L=torch.nn.L1Loss()

GD=GeneralizedDice()

Lossf= lambda Ytrue, Ypred :  DiceLoss(Ytrue[0], Ypred[0])  + GD(Ypred[1],Ytrue[1]) #DiceLoss(Ytrue[1], Ypred[1])#+ L(Ytrue[1],Ypred[1]) #+MonoLoss(Ytrue[0],Ypred[0],BoundaryWeight,W1[0]) + CateLoss(Ytrue[1],Ypred[1],BoundaryWeight,W1[1:],categories=PAR['Categories']) 

    
##############################################################################
trainparams={
        # 'name':name,
        'maxtime':maxtime,
        'EPOCHS':EPOCHS,
        'PARAMS':PAR,
        'decay':decay,
        'testsize':testsize,
        'patience':patience
        }



scaler = H.Rescale(1.02,0.95,0.5,dim=3) 
normalizator = H.Normalizer()
tensorize = H.ToTensor()
transforms = torchvision.transforms.Compose([scaler, normalizator, tensorize])


dataset=H.EpiTargetPreSS(dataroot,transform=transforms,ManualOnly=False,ArtificialMask=True)

ii=0
for train_idx, test_idx  in datasplitter.split(range(len(dataset))):
    print('Train',[dataset[k]['Sample'] for k in train_idx])
    print('Train',[dataset[k]['Sample'] for k in train_idx])
    
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=Bsize,sampler=torch.utils.data.SubsetRandomSampler(train_idx),num_workers=23)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=Bsize,sampler=torch.utils.data.SubsetRandomSampler(test_idx),num_workers=23)
    ETModelSave=HPCfolder+'/models/ETmod'+str(ii)+'.pth'
    ii+=1
    if SaveModels:savefile=ETModelSave
    
    BestDice=0
    BestStep=0
    dicesstd=0
    step=0
    UN=Network(PAR).cuda()
    
    TotalTime=0
    epoch=0
    bestloss=np.inf
    optimizer=RAdam(UN.parameters(),weight_decay=decay)
    
    UN.train()
    
    time_check=time.time()
    
    while TotalTime < maxtime and (step - BestStep < patience) and epoch < EPOCHS:
        dice=[]
        
        
        for i, sample in tqdm.tqdm(enumerate(trainloader),total=len(trainloader)):
            step+=1
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            
            try:
                out = UN(sample['MRI'].cuda())
            except:
                print(sample['Sample'])
                raise
            true=(sample['Mask'].cuda(),sample['Labels'].cuda())
            loss = Lossf(true,out)
            
            loss.backward()
            optimizer.step()
            
            dice.append(H.FullDice(true,out))
            
        print('Training set running mean dices: ')
        print(np.nanmean(dice))
    
    
        
        
        
        
        epoch+=1
        UN.eval()
        dices=[]
        losses=[]
        TotalTime+=time.time()-time_check
        time_check=time.time()
        for i, sample in enumerate(testloader):
            with torch.no_grad():
                torch.cuda.empty_cache()
                out = UN(sample['MRI'].cuda())
                true=(sample['Mask'].cuda(),sample['Labels'].cuda())
                loss = Lossf(true,out)
                dices.append(H.FullDice(true,out))
                losses.append( float(loss))
        
        inftime=(time.time()-time_check)/len(testloader)
        print('Inference time: '+str(inftime)+' per volume')
        
        dices=np.array(dices)
        losses=np.array(losses)
        candidate_loss=np.mean(losses)
        longdices=np.mean(dices,axis=0)
        longstd=np.std(dices,axis=0)
        dicesstd=np.nanstd(dices.mean(axis=1))
        dices=np.nanmean(dices)
        candidate_dice=longdices[1]+longdices[2]
        candidate_dice/=2
        
        if SaveModels: torch.save({
                'epoch': epoch,
                'model_state_dict': UN.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'step': step,
                'bestloss':bestloss,
                'TotalTime':TotalTime,
                'trainparams':trainparams
                }, saveprogress)
        UN.train()
    
        
        if candidate_dice>BestDice: 
            bestloss=candidate_loss
            BestDice=candidate_dice
            BestStep=step
            BestLSTD=longstd
            bestROId=longdices
            bestepoch=epoch
            bestdice_loss=dices
            bestSTD=dicesstd
            if SaveModels: torch.save({
                    'epoch': epoch,
                    'model_state_dict': UN.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'step': step,
                    'bestloss':bestloss,
                    'dice_scores':dices,
                    'BestStep':BestStep,
                    'TotalTime':TotalTime,
                    'trainparams':trainparams
                    }, savefile)
            
        if BestStep==step: 
            bc=H.bcolors.OKGREEN
        else:
            bc=H.bcolors.OKBLUE
        print('Dev set dices:')
        print(bc+H.bcolors.BOLD+'{:.5f}'.format(dices)+' ± {:.5f}'.format(dicesstd)+H.bcolors.ENDC)
        print(longdices)
        print(longstd)
    try:
        print('Dice scores for lowest loss'+str(bestloss)+' at epoch'+str(bestepoch)+'and step ' +str(BestStep)+ ' :')
        print(H.bcolors.OKGREEN+H.bcolors.BOLD+'{:.5f}'.format(bestdice_loss)+' ± {:.5f}'.format(bestSTD)+H.bcolors.ENDC)
        print('Mean Dice score: '+str(np.mean(bestdice_loss)))
        print(bestROId)
        print(BestLSTD)
    except:
        print('I hate print statements')
