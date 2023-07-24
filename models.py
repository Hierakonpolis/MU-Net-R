#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:17:03 2020

@author: cat
"""

from radam import RAdam
import numpy as np
import torch, tqdm, time, os


"""
Wrapper for our models, to streamline training, loading, and similar operations
"""

EPS=1e-10
smooth = 1

class GeneralizedDice():
    def __init__(self, classs=(0,1,2),sumdims=(2,3,4)):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = classs
        self.sumdims=sumdims

    def __call__(self, probs, target):
        
        pc = probs#[:, self.idc, ...].type(torch.cuda.FloatTensor)
        tc = target#[:, self.idc, ...].type(torch.cuda.FloatTensor)
        
        w = 1 / ((torch.einsum("bcdwh->bc", tc) + 1e-10) ** 2) 
        intersection = w *torch.einsum("bcdwh,bcdwh->bc", pc, tc)
        union = w * (torch.einsum("bcdwh->bc", pc) + torch.einsum("bcdwh->bc", tc))
        divided = 1 - 2 * (torch.einsum("bc->b", intersection) + 1e-10) / (torch.einsum("bc->b", union) + 1e-10)
        loss = divided.mean()
        return loss

def FocalTversky(y_true, y_pred, alpha=.7, gamma=.75):
    # another potentially useful loss. was not in the original, but it should improve things
    true_pos = torch.sum(y_true * y_pred)
    false_neg = torch.sum(y_true * (1-y_pred))
    false_pos = torch.sum((1-y_true)*y_pred)
    tversky = (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

    return torch.pow(1 - tversky, gamma)

class SurfaceLoss():
    def __init__(self, classs=(0,1)):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = classs

    def __call__(self, probs, ground_truth):

        pc = probs[:, self.idc, ...]#.type(torch.cuda.FloatTensor)
        dc = ground_truth[:, self.idc, ...]#.type(torch.cuda.FloatTensor)

        multipled = torch.einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss

def Dice(labels,Ypred):
    
    labels [np.where(labels == np.amax(labels,axis=1))] = 1
    labels[labels!=1]=0
    
    dice=2*(np.sum(labels*Ypred,(0,2,3,4))+1)/(np.sum((labels+Ypred),(0,2,3,4))+1)
    
    return dice

def MonoDice(labels,Ypred):
    Ypred[Ypred>=0.5]=1
    Ypred[Ypred!=1]=0
    dice=2*(np.sum(labels*Ypred,(0,2,3,4))+1)/(np.sum((labels+Ypred),(0,2,3,4))+1)
    
    return dice

def DiceLoss(Ytrue,Ypred):
    '''
    Returns binary cross entropy + dice loss for one 3D volume, normalized
    W0: added weight on region border
    W1: base class weight for Binary Cross Entropy, should depend on frequency
    '''

    DICE = -torch.div( torch.sum(torch.mul(torch.mul(Ytrue,Ypred),2)), torch.sum(torch.mul(Ypred,Ypred)) + torch.sum(torch.mul(Ytrue,Ytrue))+1)
    
    return DICE

GD=GeneralizedDice()

class Segmentation():
    
    def __init__(self,network,savefile=None,parameters=None,testset=None,device='cuda'):
        
        if savefile and os.path.isfile(savefile):
            self.load(savefile,network)
        else:
            self.opt={}
            
            self.opt['PAR']=parameters
            self.opt['device']=device
            self.opt['testset']=testset
            self.opt['Epoch']=0
            self.opt['TrainingLoss']=[]
            self.opt['TestDices']=[]
            self.opt['TestLoss']=[]
            self.opt['TotalTime']=0
            self.opt['BestLoss']=np.inf
            self.opt['BestLossEpoch']=0
            self.opt['BestDice']=0
            
            self.network=network(self.opt['PAR']).to(self.opt['device'])
            self.optimizer=RAdam(self.network.parameters(),weight_decay=self.opt['PAR']['WDecay'])
            
    
    def save(self,path):
        
        torch.save({'opt':self.opt,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    path)
        
    def load(self,path,network):
        checkpoint=torch.load(path)
        self.opt=checkpoint['opt']
        
        self.network=network(self.opt['PAR']).to(self.opt['device'])
        self.optimizer=RAdam(self.network.parameters(),weight_decay=self.opt['PAR']['WDecay'])
        
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Loaded model',path)
    
    def setnames(self,mask,labels):
        self.opt['MaskName'] = mask
        self.opt['LabelNames'] = labels
    
    def getnames(self):
        try:
            return self.opt['MaskName'], self.opt['LabelNames']
        except:
            return 'Mask', ['Ipsi','Contra','Background']
        
    def train_one_epoch(self,dataloader):
        
        self.network.train()
        
        losses=[]
        
        for sample in tqdm.tqdm(dataloader,total=len(dataloader),desc='Training...'):
            
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            
            
            out = self.network(sample['MRI'].to(self.opt['device']))
            
            true=(sample['Mask'].to(self.opt['device']),sample['Labels'].to(self.opt['device']))
            
            loss = self.loss(out,true)
            
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        
        self.opt['Epoch']+=1
        self.opt['TrainingLoss'].append(np.mean(losses))
        print(flush=True)
        return np.mean(losses)
    
    def test(self,dataloader):
        losses=[]
        dices=[]
        self.network.eval()
        
        with torch.no_grad():
            for sample in tqdm.tqdm(dataloader,total=len(dataloader),desc='Testing...'):
                torch.cuda.empty_cache()
                
                out = self.network(sample['MRI'].to(self.opt['device']))
                true=(sample['Mask'].to(self.opt['device']),sample['Labels'].to(self.opt['device']))
            
                loss = self.loss(out,true)
                losses.append(float(loss.detach().cpu()))
                if not self.opt['PAR']['MaskOnly']: 
                    dl = Dice(sample['Labels'].detach().cpu().numpy(), out[1].detach().cpu().numpy())
                else:
                    dl = []
                if not self.opt['PAR']['LabelsOnly']: 
                    dm = MonoDice(sample['Mask'].detach().cpu().numpy(), out[0].detach().cpu().numpy())
                else:
                    dm = []
                dices.append(list(dm) + list(dl))

        self.opt['TestLoss'].append((self.opt['Epoch'],np.mean(losses)))
        self.opt['TestDices'].append((self.opt['Epoch'],dices))
        dices=np.array(dices).mean(axis=0)
        print(flush=True)
        print('Test set Dices:',dices,flush=True)

        return np.mean(losses), dices
    
    def train(self,
              train_dataloader,
              test_dataloader,
              max_epochs,
              patience,
              max_time,
              saveprogress,
              savebest,
              LossMax=0.2,#above this threshold, do not trigger early stopping
              mindice=0.8,
              ): #below this mean dice threshold, do not trigger early stopping. Includes mask and background
        testloss=0
        testdice=0
        while \
                (
                        self.opt['Epoch']<max_epochs and
                        self.opt['TotalTime']<max_time and
                        (self.opt['Epoch']-self.opt['BestLossEpoch'])<patience
                ) or \
                        testloss>LossMax or \
                        testdice<mindice:
            start=time.time()
            print('Epoch',self.opt['Epoch'],flush=True)
            trainloss = self.train_one_epoch(train_dataloader)
            print('Training set running mean loss: ',trainloss,flush=True)
            
            testloss, dice = self.test(test_dataloader)
            print('Test set loss: ',testloss,flush=True)
            self.opt['TotalTime'] += time.time()-start
            
            if testloss < self.opt['BestLoss']: 
                self.opt['BestLoss']= testloss
                self.opt['BestLossEpoch']=self.opt['Epoch']
                self.save(savebest)
            
            testdice =  np.mean(dice)
            if testdice > self.opt['BestDice']:
                savedice = savebest.rstrip('.pth')+'_dice.pth'
                self.opt['BestDice'] = np.mean(dice)
                self.save(savedice)
            
            if saveprogress: self.save(saveprogress)
            if self.opt['Epoch']>max_epochs and (testloss>LossMax):  # or testdice<mindice):
                return False
        print('Best dice',self.opt['BestDice'])
        return True
            
    def loss(self,out,true):
        
        Tmask,Tlabels=true
        mask,labels=out
        
        trues = []
        infers = []
        if not self.opt['PAR']['LabelsOnly']:
            Mlossa=self.opt['PAR']['MaskWeight']*DiceLoss(Tmask,mask) 
            trues += [Tmask]
            infers += [mask]
        else:
            Mlossa = 0
        
        Mloss=Mlossa#+Mlossb
        
        if not self.opt['PAR']['MaskOnly']:
            trues += [Tlabels]
            infers += [labels]
            Lloss = self.opt['PAR']['GenDiceWeight']*GD(labels,Tlabels)
        else:
            Lloss=0
        a = Mloss+Lloss
        return a
    
    def skull_inference(self,inputloader):
        masks={}
        for i, sample in tqdm.tqdm(enumerate(inputloader),total=len(inputloader),desc='Inference...'):
            mask=self.network(sample['MRI'].to(self.opt['device']))[0]
            mask=mask.detach().cpu().numpy()
            
            masks[sample['Sample'][0]]=(mask,sample['Path'][0])
            
        return masks
    
    
    def inferece(self,inputloader):
        predictions={}
        
        self.network.eval()
        
        with torch.no_grad():
            for i, sample in tqdm.tqdm(enumerate(inputloader),total=len(inputloader),desc='Inference...'):
                torch.cuda.empty_cache()
                
                mask,labels = self.network(sample['MRI'].to(self.opt['device']))
                
                if type(mask) is not list: mask=mask.detach().cpu().numpy()
                if len(labels)>0:
                    labels=labels.detach().cpu().numpy()
                
                predictions[sample['Sample'][0]]=(mask,labels,sample['Path'][0])
                
        
        return predictions