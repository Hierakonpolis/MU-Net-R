#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:43:25 2019

@author: riccardo
"""

import helper as H
# import Helper2 as H2
import numpy as np
import torch, tqdm#,pickle
import torchvision, sys
from network import PARAMS_Actual2D, MUnet
from models import DiceLoss, GeneralizedDice
import time
import os
from radam import RAdam
from sklearn.model_selection import KFold

dataroot='/media/Olowoo/hippocampus/Epitarget_HC_lining'
HPCfolder='/media/Olowoo/hippocampus'
outvalf='/media/Olowoo/hippocampus/ValET'

SaveModels=True

def split_indexes(indexes,folds):
    randomized=np.random.choice(indexes,size=len(indexes),replace=False)
    sets={}
    for k in range(folds):
        sets[k]=[]
    
    i=0
    for k in randomized:
        sets[i].append(k)
        i+=1
        if i==folds: i=0
    return list(sets.values())

def collate(indlist,whichistest):
    train=[]
    test=[]
    
    for k in range(len(indlist)):
        if k == whichistest:
            test.append(indlist[k])
        else:
            train.append(indlist[k])
    return train, test



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
Bsize=1
BoundaryWeight=2
Network=MUnet#HPonly
datasplitter=KFold(6) 
# Given the default order in the dataset this is enough to split subjects
# according to animal identity
# this might be different on another system

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
        'patience':patience
        }



scaler = H.Rescale(1.02,0.95,0.5,dim=3) 
normalizator = H.Normalizer()
tensorize = H.ToTensor()
transforms = torchvision.transforms.Compose([scaler, normalizator, tensorize])

dataset=H.EpiTargetPreSS(dataroot,transform=transforms,ManualOnly=False,ArtificialMask=True)


for train_idx, test_idx  in datasplitter.split(range(len(dataset))):
    innerfolds=split_indexes(train_idx, 6)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=Bsize,sampler=torch.utils.data.SubsetRandomSampler(test_idx),num_workers=23)
    TEST_ID = dataset[test_idx[0]]['ID'].split('_')[0]
    ii=0
    modpaths=[]
    
    for inner in range(6):
        INtrain, INval = collate(train_idx,inner)
        
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=Bsize,sampler=torch.utils.data.SubsetRandomSampler(INtrain),num_workers=23)
        valloader = torch.utils.data.DataLoader(dataset, batch_size=Bsize,sampler=torch.utils.data.SubsetRandomSampler(INval),num_workers=23)
        
        ETModelSave=HPCfolder+'/models/ETmod_Val'+TEST_ID+'_'+str(ii)+'.pth'
        modpaths.append(ETModelSave)
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
    
        while (TotalTime < maxtime and (step - BestStep < patience) and epoch < EPOCHS) or BestDice < 0.5:
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
                true=(sample['Mask'].cuda(),sample['Labels'].cuda())#,sample['Ref'])
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
            for i, sample in enumerate(valloader):
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
            # distances=np.mean(distances)
            losses=np.array(losses)
            candidate_loss=np.mean(losses)
            longdices=np.mean(dices,axis=0)
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
            
        try:
            print('Dice scores for lowest loss'+str(bestloss)+' at epoch'+str(bestepoch)+'and step ' +str(BestStep)+ ' :')
            print(H.bcolors.OKGREEN+H.bcolors.BOLD+'{:.5f}'.format(bestdice_loss)+' ± {:.5f}'.format(bestSTD)+H.bcolors.ENDC)
            print('Mean Dice score: '+str(np.mean(bestdice_loss)))
            print(bestROId)
        except:
            print('I hate print statements')
    
    ## build outputs here
    predictions={}
    for modelpath in modpaths:
        with torch.no_grad():
            checkpoint=torch.load(modelpath)
            UN.load_state_dict(checkpoint['model_state_dict'])
            UN.eval()
            
            for i, sample in tqdm.tqdm(enumerate(testloader),desc='Inference '+modelpath,total=len(testloader)):
                    
                torch.cuda.empty_cache()
                mask, labels = UN(sample['MRI'].cuda())
                ID = sample['ID'][0]
                mask = mask.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                
                if ID not in predictions:
                    predictions[ID]=[]
                    
                predictions[ID].append([mask,labels,sample['Path'][0]])
    for ID, results in predictions.items():
        M=0
        LAB=0
        
        
        for k in results:
            M+=k[0]/len(results)
            LAB+=k[1]/len(results)
        
        Mout = os.path.join(outvalf,ID+'_Mask.nii.gz')
        IPSout = os.path.join(outvalf,ID+'_Ipsi.nii.gz')
        CONout = os.path.join(outvalf,ID+'_Contra.nii.gz')
        
        M=M.reshape(M.shape[-3:])
        M[M>=0.5]=1
        M[M!=1]=0
        M=H.LargestComponent(M)
        M=H.FillHoles(M)
        
        LAB [np.where(LAB== np.amax(LAB,axis=1))] = 1
        LAB[LAB!=1]=0
        
        IPSI=LAB[0,0,:,:,:]
        CONTRA=LAB[0,1,:,:,:]
        
        IPSI=IPSI.reshape(IPSI.shape[-3:])        
        CONTRA=CONTRA.reshape(CONTRA.shape[-3:])
        
        IPSI=H.LargestComponent(IPSI)
        CONTRA=H.LargestComponent(CONTRA)
        
        IPSI=H.FillHoles(IPSI)
        CONTRA=H.FillHoles(CONTRA)
        
        # H.MakeNii(R,prediction[ID][-1],filename)
        assert results[0][2] == results[1][2]
        
        H.MakeNii(M,results[0][2],Mout)
        H.MakeNii(IPSI,results[0][2],IPSout)
        H.MakeNii(CONTRA,results[0][2],CONout)
        
        