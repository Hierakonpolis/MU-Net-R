#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:56:54 2021

@author: cat
"""
import argparse, pathlib, os, torch, re, shutil
from torchvision.transforms import Compose
import helper as H
import models as M
import network as N
import numpy as np
from sklearn.model_selection import KFold

#%% Parse input arguments
desc='Train an ensamble of neural networks and use it to segment MRI volumes'

parser = argparse.ArgumentParser(description=desc)

parser.add_argument('dataroot', metavar = 'Datafolder', type=pathlib.Path,
                    help = 'Root folder of the dataset, for training or labeling. Each sample should be located in a separate folder, and for training mask files should be located in the same folder. Files should have the same names in each folder.')

parser.add_argument('mriname', metavar = 'MRIname',type=str,
                    help = 'Name of the MRI file for each sample.')

parser.add_argument('modelname', type = str,
                    help = 'Define a name of the model, to load or save. Do not add a file extension.')

parser.add_argument('--savefolder', type = pathlib.Path, default = os.getcwd(),
                    help = 'Specify a directory where to save or from which to load the model, if none is specified the current working dir will be used.')

parser.add_argument('--train', action='store_true', default = False,
                    help = 'Train an ensamble of network.')

parser.add_argument('--mask', help = 'Name of brain mask file, optional.',
                    type = str, default = None)

parser.add_argument('--labels', nargs = '*', default = None, type = str,
                    help = 'Name of ROI mask files, listed after the argument, e.g. --labels file1.nii, file1.nii.')

parser.add_argument('--foldfile', default = None, type = str,
                    help = 'If you want to manually assign each sample to a specific fold, specify here the name of a text file in each folder indicating the specific fold it was assigned to, with a number starting from 0. Example: --foldfile fold.txt, where fold.txt contains the number 2. Mark each fold with an integer.')

parser.add_argument('--folds', default = 6, type = int,
                    help = 'Number of folds for your dataset, to use as different validation sets for each model.')

parser.add_argument('--workers', default = 0, type = int,
                    help = 'Number of external parallel processes to spawn when loading data. Provide any integer larger than zero to use parallel computing and speed up loading the data.')

parser.add_argument('--twoD', action = 'store_true',
                    help = 'Use 2D filters instead of defaulting to 3D filters. Can be useful for anisotropic data. The direction of anisotropy is assumed to be on the third axis of the nii volumes: filters and pooling will be (x,x,1).')

parser.add_argument('--maxtime',default = 300, type=int,
                    help = 'Maximum time to train each network in the ensemble, in minutes.')

parser.add_argument('--kernels', nargs = '*', default = [16, 32, 64, 64], type = int,
                    help = 'numer of kernels for convolutions in each block, shallowest to deepest e.g. --kernels 16 36 64 64')

parser.add_argument('--patience', default = 20, type = int,
                    help = 'After spending these many epochs with no validation set improvements, stop training.')

parser.add_argument('--maxepochs', default = np.inf, type = float,
                    help = 'Maximum number of epochs spent training. By default set to infinity.')

parser.add_argument('--maskweight', default = 1, type = float,
                    help = 'Weight parameter for the brain mask. If you get stuck training and only learning the brain mask, make this smaller. E.g. 0.01 or 0.001.')

parser.add_argument('--startfold', default = 0, type = int,
                    help = 'Fold to start from when training, default = 0. Use if resuming.')

parser.add_argument('--max_loss', default = 0.2, type = float,
                    help = 'Do not accept early stopping above this value for the loss. May reboot training. May require trial and error. Default: 0.2')

parser.add_argument('--min_dice', default = 0.8, type = float,
                    help = 'Do not accept early stopping when the AVERAGE Dice score is below this value. May reboot training. May require trial and error. Default: 0.8')

parser.add_argument('--outpath', type=pathlib.Path, default=None,
                    help = 'Specify a different output directory. Otherwise, outputs will be placed in the same directory as inputs. Replicates the same folder structure.')

args = parser.parse_args()

def fixp(p):
    if 'LabelsOnly' not in p:
        p['LabelsOnly'] = False
        
    if 'MaskOnly' not in p:
        p['MaskOnly'] = False

#%% Dataset and transforms

normalizator = H.Normalizer()
tensorize = H.ToTensor()

if args.train:
    scaler = H.Rescale(1.02,0.95,0.5,dim=3)
    transforms = Compose([scaler,normalizator,tensorize])
else:
    transforms = Compose([normalizator,tensorize])

Dataset = H.GenericDataset(args.dataroot, args.mriname,
                          vol_list = args.labels, maskname = args.mask, training = args.train, 
                          transform = transforms,foldfile = args.foldfile)

#%% Select network and parameters

hasmasks = len(Dataset.masks) > 0
haslabels = len(Dataset.vols) > 0

if haslabels: 
    print('ROIs detected')
    

if hasmasks:
    print('Mask detected')
    if haslabels:
        p = N.PARAMS_3DEpiBios
        n = N.MUnet
    else:
        p = N.PARAMS_SKULLNET
        n = N.SkullNet
elif haslabels:
    n = N.HPonly
    p = N.PARAMS_3DEpiBios
    p['LabelsOnly'] = True
if args.twoD:
    p['FilterSize'] = (3,3,1)
    p['PoolShape'] = (2,2,1)

if hasmasks or haslabels: p['FiltersNum'] = args.kernels
if haslabels:
    p['Categories'] = len(args.labels)
#%% Training script


if args.train:
    p['FiltersNum'] = args.kernels
    print('Training')
    # Split the data in different folds, or use custom split
    assignments = np.array(Dataset.foldlist)
    
    p['MaskWeight'] = args.maskweight
    
    if len(assignments) != 0:
        uniques = H.unique_elements(assignments)
        folds = len(uniques)
        groups = {k:[] for k in uniques}
        for i, k in enumerate(assignments):
            groups[k].append(i)
        
        split = []
        for k in range(folds):
            train = []
            test = []
            for j in range(folds):
                if j == k:
                    test += list(groups.values())[j]
                else:
                    train += list(groups.values())[j]
        
            split.append([np.array(train),np.array(test)])
        
        
    else:
        folds = args.folds
        datasplitter=KFold(folds, shuffle = True, random_state = 93)
        split = datasplitter.split(range(len(Dataset)))
    
    # training loop over different folds
    
    for i, (train_idxs, test_idxs)  in enumerate(split):
        if i < args.startfold: continue
        converged = False
        while not converged:
            ModelSave = os.path.join(args.savefolder, args.modelname+'_'+str(i)+'.pth')
            trainloader = torch.utils.data.DataLoader(Dataset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(train_idxs),num_workers=args.workers)
            testloader = torch.utils.data.DataLoader(Dataset, batch_size=1, sampler=torch.utils.data.SubsetRandomSampler(test_idxs),num_workers=args.workers)
            
            Model=M.Segmentation(n,savefile=None,parameters=p,device='cuda')
            Model.setnames(args.mask,args.labels)
            
            converged = Model.train(trainloader,
                                    testloader,
                                    max_epochs=args.maxepochs,
                                    patience=args.patience,
                                    max_time=60*args.maxtime,
                                    saveprogress=False,
                                    savebest=ModelSave,
                                    LossMax=args.max_loss,
                                    mindice=args.min_dice)
#%% Inference
else:
    print('Inference')
    pattern = re.compile(args.modelname+'_[0-9]*.pth')
    
    models = []
    inferences = []
    for file in os.scandir(args.savefolder):
        if pattern.search(file.name): models.append(file.path)
    
    if len(models) == 0:
        raise NameError('No models found for inference named '+str(args.modelname)+' in '+str(args.savefolder))
    
    loader=torch.utils.data.DataLoader(Dataset, batch_size=1,num_workers=args.workers)
    for sfile in models:
        p=torch.load(sfile)['opt']['PAR']
        fixp(p)
        
        if p['MaskOnly']:
            n = N.SkullNet
        elif p['LabelsOnly']:
            n = N.HPonly
        else:
            n = N.MUnet
            
        Model=M.Segmentation(n,savefile=sfile,parameters=p,device='cuda')
        inferences.append(Model.inferece(loader))
    
    p = Model.opt['PAR']
    fixp(p)
    maskn, labeln = Model.getnames()
    
    if maskn is not None: maskn = maskn.split('.nii')[0]
    if labeln is not None: labeln = [k.split('.nii')[0] for k in labeln]
    
    for ID in inferences[0]:
        temp=0
        templ=0
        nameroot = ID.split('.nii')[0]
        if args.outpath is not None:
            nameroot = str(nameroot).replace(str(args.dataroot),str(args.outpath))
            if not os.path.isdir(nameroot):
                os.makedirs(os.path.dirname(nameroot))
        
        for prediction in inferences:
            if not p['LabelsOnly']: temp+=prediction[ID][0]/len(inferences)
            if not p['MaskOnly']: templ+=prediction[ID][1]/len(inferences)
        
        if not p['LabelsOnly']:
            temp[temp>=0.5]=1
            temp[temp!=1]=0
            temp=temp.reshape(temp.shape[-3:])
            temp=H.LargestComponent(temp)
            temp=H.FillHoles(temp)
            
            filename=nameroot+'_'+maskn+'.nii.gz'
            H.MakeNii(temp,prediction[ID][-1],filename)
        
        if not p['MaskOnly']:
            templ [np.where(templ== np.amax(templ,axis=1))] = 1
            templ[templ!=1]=0
            if 'Background' not in labeln:
                labeln += ['Background']
            for idx, roiname in enumerate(labeln):
                filename=nameroot+'_'+roiname+'.nii.gz'
                R=templ[:,idx,...]
                R=R.reshape(R.shape[-3:])
                R=H.LargestComponent(R, 2).astype(float)
                R=H.FillHoles(R).astype(float)
                H.MakeNii(R,prediction[ID][-1],filename)
        
