#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:23:43 2019

@author: riccardo
"""
import os
import torch
import nibabel as nib
import numpy as np
import copy, random
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import shift
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from skimage.measure import label as SkLabel
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_opening
from scipy.ndimage import binary_closing
from scipy import ndimage
import warnings
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from scipy.ndimage import distance_transform_edt as distance
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

def one_hot2dist(seg):
    C=len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


ker=np.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]])

ker=np.swapaxes(ker,0,-1)

def niload(path):
    nob=nib.load(path, keep_file_open=False)
    data=nob.get_fdata()
    new=np.copy(data)
    del nob
    del data
    return new

def MakeNii(Volume,ReferencePath,Output,reshape=True):
    '''
    Builds nii file in the same reference of the volume located at 
    ReferencePath, from PyTorchTensor, placing it at Output. Es:
        MakeNii(Mask,'mydata/ref.nii','output/Mask.nii')
    
    '''
    Reference=nib.load(ReferencePath)
    Affine=Reference.affine
    if reshape: Volume=Volume.reshape((Volume.shape[-3],Volume.shape[-2],Volume.shape[-1]))
    nii=nib.Nifti1Image(Volume.astype(float),Affine)
    nii.header['qoffset_x']=Reference.header['qoffset_x']
    nii.header['qoffset_y']=Reference.header['qoffset_y']
    nii.header['qoffset_z']=Reference.header['qoffset_z']
    nib.save(nii,Output)
    
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def LargestComponent(Mask, components=1):
    Divs=SkLabel(Mask)
    counts=np.zeros(np.max(Divs))
    taken = 0
    inds = []
    for i in range(len(counts)):
        counts[i]=np.sum(Mask[Divs==(i+1)])
    if len(counts)==0: return Mask
    while taken < components:
        inds.append(np.argmax(counts))
        counts[inds[-1]] = -1
        taken +=1
    NewMask = np.zeros_like(Mask)
    for i in inds:
        NewMask[Divs == i+1] = 1
    # Mask[Divs!=ind]=0
    return NewMask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def TCudaCheck():
    '''
    Just prints a bunch of data on what CUDA is doing with the memory
    '''
    div=1000000
    print('Cuda is available: '+str(torch.cuda.is_available()))
    print('Max Memory Allocated: '+str(torch.cuda.max_memory_allocated(device=None)/div))
    print('Max Memory Cached: '+str(torch.cuda.max_memory_cached(device=None)/div))
    print('Memory Allocated: '+str(torch.cuda.memory_allocated(device=None)/div))
    print('Memory Cached: '+str(torch.cuda.memory_cached(device=None)/div))
    
def FillHoles(image):
    return binary_fill_holes(image).astype(int)
    
    
def MakeLabPic(Yout,roi_index,mri_slice):
    if mri_slice>=Yout.shape[4]:
        mri_slice=Yout.shape[4]-1
    Show=Yout.narrow(4,mri_slice,1)
    Show=Show.narrow(1,roi_index,1)
    return torch.reshape(Show,(Yout.shape[2],Yout.shape[3]))


def AltRefine(Maskfile, out,iters=1):
    ker1=np.swapaxes(ker,1,-1)
    M=niload(Maskfile)
    M=np.swapaxes(M,1,2)
    new=binary_closing(M,ker)
    new=binary_closing(new,ker1).astype(int)
    new=binary_opening(new,iterations=iters).astype(int)
    MakeNii(new, Maskfile, out)
    


def PPCloseAndHoles(Mask):
    Mask[Mask>=0.5]=1
    Mask[Mask!=1]=0
    for i in range(Mask.shape[2]):
        OO=Mask[:,:,i]
        OO=binary_closing(OO,ker)
        Mask[:,:,i]=FillHoles(OO)
    for i in range(Mask.shape[1]):
        OO=Mask[:,i,:]
        OO=binary_closing(OO,ker)
        Mask[:,i,:]=FillHoles(OO)
    for i in range(Mask.shape[0]):
        OO=Mask[i,:,:]
        OO=LargestComponent(OO)
        Mask[i,:,:]=FillHoles(OO)
    
        
    return Mask

def PPShift(volin,d1=0,d2=5,d3=5):
    
    a=volin
    b=np.zeros_like(a)
    for k in range(a.shape[0]-d1):
        b[k,:,:]=a[k+d1,:,:]
    c=np.zeros_like(a)
    for k in range(b.shape[1]-d2):
        c[:,k,:]=b[:,k+d2,:]
    d=np.zeros_like(c)
    for k in range(a.shape[2]-d3):
        d[:,:,k]=c[:,:,k+d3]
    
    return d


def PPRefine(volin,mri):
    
    d=volin
    d1=np.copy(d)
    
    for k in range(d1.shape[1]):
        d1[:,k,:]=binary_dilation(d1[:,k,:],iterations=10)
    
    mri[d1==0]=mri.min()
    mri=inverse_gaussian_gradient(mri)
    d=morphological_geodesic_active_contour(mri,5,init_level_set=d,smoothing=1)#,balloon=-0.1)
    
    return d
    

def ShiftNii(volin,volout,mrivol,d1=0,d2=5,d3=5):
    mri=niload(mrivol)
    a=niload(volin)
    
    d= PPRefine(PPShift(a,d1,d2,d3),mri)
    MakeNii(d,volin,volout)
    return d
        

def MaskFixer(Mask):
    
    OrigShape=Mask.shape
    Mask=Mask.reshape((OrigShape[-3],OrigShape[-2],OrigShape[-1]))
    for i in range(Mask.shape[2]):
        Mask[:,:,i]=FillHoles(binary_closing(Mask[:,:,i],ker))
    
    Mask=LargestComponent(Mask)
    Mask=Mask.reshape(OrigShape)
    
    return Mask



def NiftyzeLabels(samples,ReferencePath,outfolder,LabelNames=['Cortex','Hippocampus','Ventricles','Striatum','Background'],add='',correct=True):
    Mask=samples[0].detach().cpu().numpy()
    Labels=samples[1].detach().cpu().numpy()
    if correct:
        Mask, Labels = PPCloseAndHoles(Mask,Labels)
    
    for i in range(Labels.shape[1]):
        Vol=Labels[0,i,:,:,:]
        MakeNii(Vol,ReferencePath,outfolder+LabelNames[i]+add+'.nii.gz')
    MakeNii(Mask,ReferencePath,outfolder+'Mask'+add+'.nii.gz')


def OneDice(Ytrue,Ypred):
    Mask=Ypred.detach().cpu().numpy()
    Mask[Mask>=0.5]=1
    Mask[Mask!=1]=0
    
    Ytrue=Ytrue.cpu().numpy()
    
    dice=2*np.sum(Ytrue*Mask)/np.sum((Ytrue+Mask))
    
    return dice

def FullDice(Ytrue,Ypred):
    
    
    Mask=Ypred[0].detach().cpu().numpy()
    Mask[Mask>=0.5]=1
    Mask[Mask!=1]=0
    labels=Ypred[1].detach().cpu().numpy()
    labels [np.where(labels== np.amax(labels,axis=1))] = 1
    labels[labels!=1]=0
    Ypred=np.concatenate((Mask,labels),axis=1)
    Ytrue=np.concatenate((Ytrue[0].cpu().numpy(), Ytrue[1].cpu().numpy()),axis=1)
    
    
    dice=2*np.sum(Ytrue*Ypred,(0,2,3,4))/np.sum((Ytrue+Ypred),(0,2,3,4))
    return dice

def SimpleDice(x,y):
    
    labels=x.detach().cpu().numpy()
    labels [np.where(labels== np.amax(labels,axis=1))] = 1
    labels[labels!=1]=0
    Ypred=labels
    Ytrue=y.detach().cpu().numpy()
    
    
    dice=2*np.sum(Ytrue*Ypred,(0,2,3,4))/np.sum((Ytrue+Ypred),(0,2,3,4))
    return dice

def HPCDice(Ytrue,Ypred):
    
    Ytrue=Ytrue.detach().cpu().numpy()
    labels=Ypred.detach().cpu().numpy()
    labels [np.where(labels== np.amax(labels,axis=1))] = 1
    labels[labels!=1]=0
    Ypred=labels
    
    
    dice=2*np.sum(Ytrue*Ypred,(0,2,3,4))/np.sum((Ytrue+Ypred),(0,2,3,4))
    return dice


def MaskDice(Ytrue,Ypred,correct=False,extra_mask_fix=False):
    Mask=Ypred[0].detach().cpu().numpy()
    Mask[Mask>=0.5]=1
    Mask[Mask!=1]=0
    Ytrue=Ytrue.narrow(1,0,1).cpu().numpy()
    
    if correct:
        Mask, Ypred = PPCloseAndHoles(Mask,Ypred)
    
    if extra_mask_fix:
        Mask=MaskFixer(Mask)
    
    
    dice=2*np.sum(Ytrue*Mask,(0,2,3,4))/np.sum(Ytrue+Mask,(0,2,3,4))
    return dice


def DiceTester(Ytrue,Ypred):

    Ytrue=Ytrue.cpu().numpy()
    Ypred=Ypred.cpu().numpy()
    
    
    dice=2*np.sum(Ytrue*Ypred,(1,2,3))/np.sum(Ytrue+Ypred,(1,2,3))
    return dice


def snr(folder,sli):
    if folder[-1]!='/':
        folder+='/'
    vol=nib.load(folder+'2dseq.nii').get_data()[:,:,sli]
    mask=nib.load(folder+'scan_brainmask.nii').get_data()[:,:,sli]
    noise=nib.load(folder+'snr.nii.gz').get_data()[:,:,sli]
    
    print('with std: '+str(np.nanmean(vol[mask==1])/np.nanstd(vol[noise==1])))
    print('with signal: '+str(np.nanmean(vol[mask==1])/np.nanmean(np.abs(vol[noise==1]))))
    print('with std/M: '+str(np.nanmean(vol[mask==1])/np.nanstd(vol[mask==0])))
    print('with signal/M: '+str(np.nanmean(vol[mask==1])/np.nanmean(np.abs(vol[mask==0]))))
    

def XYBox(Mask):
    S=Mask.shape
    LowX=0
    while np.sum(Mask[LowX,:,:])==0:
        LowX+=1
    
    LowY=0
    while np.sum(Mask[:,LowY,:])==0:
        LowY+=1
    
    
    HighX=S[0]-1
    while np.sum(Mask[HighX,:,:])==0:
        HighX-=1
    
    HighY=S[1]-1
    while np.sum(Mask[:,HighY,:])==0:
        HighY-=1
        
    return LowX, HighX, LowY, HighY

def optinfile(addto,key,fileobj,filename,animal):
    if filename in fileobj.name:
        if animal not in addto.keys(): addto[animal]={}
        addto[animal][key]=fileobj.path
        return True
    else:
        return False

def DatasetDict(HPCfolder):
    cohorts=['C1','C2','C3']
    segmentation={}
    
    for animal in os.scandir(os.path.join(HPCfolder,'segmentations')):
        if os.path.isdir(animal): 
            name=animal.name
            for file in os.scandir(animal):
                if optinfile(segmentation, 'Sparse Mask', file, '_mask_wholebrain.nii',name):
                    segmentation[name]['Preliminary Mask']=  os.path.join(animal.path,'Prel_wholebrain.nii')
                
    for cohort in cohorts:
        folder=os.path.join(HPCfolder,cohort)
        for file in os.scandir(folder):
            for name in [animal for animal in segmentation.keys() if (animal in file.name)]:
                segmentation[name]['MRI']=file.path
    
    return segmentation

exclusions=['MGRE_anatomy_1029_d9_breathingMovement.nii',
            'MGRE_anatomy_1161_5mo_Rot.nii',
            'MGRE_anatomy_1156_5mo_Rot_Craniectomy.nii',
            'MGRE_anatomy_1153_5mo_Rot.nii',
            'MGRE_anatomy_1098_2d_CONTRA injuryExcluded.nii',
            'MGRE_anatomy_1029_d9_breathingMovement.nii',
            'MGRE_anatomy_1028_5mo_Rot.nii'
            ]

AllAnimals=[1019,1028,1036,1038,1043,1046,1090,1095,1099,1103,1105,1138,1139,1142,1145,1149,1150,1152,1153,1154,1156,1158,1159,1017,1035,1045,1085,1091,1096,1107,1143,1146,1155,1161,1008,1012,1024,1029,1031,1084,1104,1140,1144]
AllAnimals=[str(k) for k in AllAnimals]


def FullDatasetDict(HPCfolder,exclude=exclusions,newm='/home/riccardo/Dataplace/hippocampus/newmasks'):
    cohorts=['C1','C2','C3']
    segmentation={}
    
    for animal in os.scandir(os.path.join(HPCfolder,'segmentations')):
        if os.path.isdir(animal):
            name=animal.name
            for file, label in zip( ['scan_HCcontra.nii', 'scan_HCipsi.nii', 'scan_refpoint.nii'],
                                   ['Contra','Ipsi','Ref']):
                if name not in segmentation: segmentation[name]={}
                
                a=os.path.join(animal.path,file)
                if os.path.isfile(a): 
                    segmentation[name][label]= a
                else:
                    pass# print(a+' missing '+a)
            a=os.path.join(newm,name+'mask.nii.gz')
            if os.path.isfile(a): 
                segmentation[name]['Mask']=a
            else:
                # print(a+' missing mask')
                pass
            
    for cohort in cohorts:
        folder=os.path.join(HPCfolder,cohort)
        for file in os.scandir(folder):
            for name in [animal for animal in segmentation.keys() if (animal in file.name)]:
                if file.name not in exclude:
                    segmentation[name]['MRI']=file.path
    todelete=[]
    for name in segmentation:
        if len(segmentation[name])<5: todelete.append(name) 
        
    for name in todelete: del segmentation[name]
    
    return segmentation


    
def InferenceDict(HPCfolder,exclude=exclusions):
    cohorts=['C1','C2','C3']
    segmentation={}
    seen={}
    
    for cohort in cohorts:
        folder=os.path.join(HPCfolder,cohort)
        for file in os.scandir(folder):
            if ('MGRE_anatomy' in file.name or 'MGRE_Anatomy' in file.name) and file.name not in exclude:
                name=file.name.lstrip('MGRE_anatomy_').lstrip('MGRE_Anatomy_').rstrip('.nii')
                optinfile(segmentation,'MRI',file,'MGRE',name)
                try:
                    ID, tp = name.split('_')
                except ValueError:
                    print(name)
                if ID not in seen: seen[ID]=[]
                seen[ID].append(tp)
    for ID in AllAnimals:
        if ID not in seen:
            print('No volumes for ID',ID)
        else:
            if len(seen[ID])<4:
                print('For '+ID+' we only have',seen[ID])
            
    
    return segmentation, seen


class SkullStripDataset(Dataset):
    
    def __init__(self,HPCfolder,transform=None):
        self.transform=transform
        data=DatasetDict(HPCfolder)
        Masks=[]
        MasksDistances=[]
        sample=[]
        MRIs=[]
        paths=[]
        
        for key in data:
            SMfile=data[key]['Preliminary Mask']
            MRfile=data[key]['MRI']
            MR=nib.load(MRfile, keep_file_open=False).get_fdata()
            
            shape=[1]+list(MR.shape)
            MRIs.append(MR.reshape(shape))
            Masks.append(niload(SMfile).reshape(shape))
            Mdist=one_hot2dist(Masks[-1])
            MasksDistances.append(Mdist)
            
            sample.append(key)
            
            paths.append(MRfile)
        
        
        
        self.MRI=MRIs
        self.Masks=Masks
        self.sample=sample
        self.MaskDistances=MasksDistances
        self.path=paths
        
    def __len__(self):
        return len(self.MRI)
    
    def __getitem__(self,idx):
        
        sample = {'MRI': self.MRI[idx],
                  'Mask': self.Masks[idx],
                  'Sample':self.sample[idx],
                  'MDist':self.MaskDistances[idx],
                  'LDist':np.zeros(self.MRI[idx].shape),
                  'Path':self.path[idx],
                  'Labels':np.zeros(self.MRI[idx].shape)}
        
#         Transform
        
        if self.transform:
            sample = self.transform(sample)
            
        
        return sample

class SSinferenceDataset(Dataset):
    
    def __init__(self,HPCfolder,transform=None,fakedata=False,SuppressLoad=False):
        self.transform=transform
        data=InferenceDict(HPCfolder)[0]
        Mslices=[]
        sample=[]
        path=[]
        for key in data:
            MRfile=data[key]['MRI']
            if SuppressLoad:
                MR = np.zeros((1,1))
            else:
                MR=niload(MRfile)
            
            shape=[1]+list(MR.shape)
            Mslices.append(MR.reshape(shape))
            
            sample.append(key)
            path.append(MRfile)
        
        assert len(Mslices) == len(path)
        
        self.MRI=Mslices
        self.sample=sample
        self.path=path
        
    def __len__(self):
        return len(self.MRI)
    
    def __getitem__(self,idx):
        
        sample = {'MRI': self.MRI[idx],
                  'Sample':self.sample[idx],
                  'Path':self.path[idx]}
        
#         Transform
        
        if self.transform:
            sample = self.transform(sample)
            
        
        return sample
class Epitarget_inference(Dataset):
    def __init__(self,transform,path='/media/Olowoo/hippocampus/epitarget_all'):
        self.MRIs=[]
        self.Names=[]
        self.Paths=[]
        self.missing=[]
        self.transform=transform
        for f in ['Day2','Day7','Day21']:
            for group in os.scandir(os.path.join(path, f)):
                for item in os.scandir(group):
                    vol=os.path.join(item,'t2star_sumOverEchoes.nii')
                    if os.path.isfile(vol):
                        self.MRIs.append(vol)
                        self.Names.append(item.name)
                        self.Paths.append(vol)
                    else:
                        self.missing.append(item.path)
                        print('Missing',item.path)
    def __len__(self):
        return len(self.MRIs)
    
    def __getitem__(self,idx):
        MRI=niload(self.MRIs[idx])
        MRI=MRI.reshape([1]+list(MRI.shape))
                  
        sample={'MRI':MRI,
                'Sample':self.Names[idx],
                'Path':self.Paths[idx]}
        
        return self.transform(sample)
    
class InferenceDataset(Dataset):
  
    def __init__(self,HPCfolder,transform=None,exclude=exclusions,):
        self.transform=transform
        data=InferenceDict(HPCfolder,exclude)[0]
        MRIs=[]
        
        sample=[]
        path=[]
        for key in data:
            
            MRIs.append(niload(data[key]['MRI']))
            MRIs[-1]=MRIs[-1].reshape([1]+list(MRIs[-1].shape))
            
            sample.append(key)
            path.append(data[key]['MRI'])
        
        self.MRI=MRIs
        self.sample=sample
        self.path=path
        
    def __len__(self):
        return len(self.MRI)
    
    def __getitem__(self,idx):
        
        sample = {'MRI': self.MRI[idx],
                  'Sample':self.sample[idx],
                  'Path':self.path[idx]}
        
#         Transform
        
        if self.transform:
            sample = self.transform(sample)
            
        
        return sample

class FullDataset(Dataset):
    
    def __init__(self,HPCfolder,transform=None,exclude=exclusions,
                  newmasks='/home/riccardo/Dataplace/hippocampus/newmasks'):
        self.transform=transform
        data=FullDatasetDict(HPCfolder,exclude=exclude,newm=newmasks)
        MRIs=[]
        labels=[]
        Ref=[]
        Masks=[]
        
        sample=[]
        IDs=[]
        path=[]
        LabelsDistances=[]
        MasksDistances=[]
        for key in data:
            Ipsi=np.swapaxes(niload(data[key]['Ipsi']),1,2)
            Contra=np.swapaxes(niload(data[key]['Contra']),1,2)
            shape=[1]+list(Ipsi.shape)
            
            Labels=np.zeros(([3]+list(Ipsi.shape)))
            Labels[0,:,:,:]=Ipsi
            Labels[1,:,:,:]=Contra
            
            NBG=Ipsi+Contra
            BG=np.zeros_like(NBG)
            BG[NBG==0]=1
            Labels[2,:,:,:]=BG
            labels.append(Labels)
            Ldist=one_hot2dist(Labels)
            LabelsDistances.append(Ldist)
            
            MRIs.append(niload(data[key]['MRI']).reshape(shape))
            Ref.append(np.swapaxes(niload(data[key]['Ref']),1,2).reshape(shape))
            Masks.append(niload(data[key]['Mask']).reshape(shape))
            Mdist=one_hot2dist(Masks[-1])
            MasksDistances.append(Mdist)
            IDs.append(key.split('_')[0])
            
            sample.append(key)
            path.append(data[key]['MRI'])
        
        assert len(MRIs) == len(Masks)
        
        self.MRI=MRIs
        self.Masks=Masks
        self.Labels=labels
        self.Ref=Ref
        self.MasksDistances=MasksDistances
        self.LabelsDistances=LabelsDistances
        self.sample=sample
        self.IDs=IDs
        self.path=path
        
        self.uIDs={}
        
        for k in IDs:
            if k not in self.uIDs.keys():
                self.uIDs[k]=1
            else:
                self.uIDs[k]+=1
    
    def IDindexes(self,indexlist,notinlist=True):
        which=[]
        
        for j in range(len(self.IDs)):
            
            if self.IDs[j] in indexlist and (not notinlist): which.append(j)
            if (self.IDs[j] not in indexlist) and notinlist: which.append(j)
                
        return which
        
        
    def __len__(self):
        return len(self.MRI)
    
    def __getitem__(self,idx):
        
        sample = {'MRI': np.copy(self.MRI[idx]),
                  'Mask':np.copy(self.Masks[idx]),
                  'Labels':np.copy(self.Labels[idx]),
                   'Ref':np.copy(self.Ref[idx]),
                   'MDist':np.copy(self.MasksDistances[idx]),
                   'LDist':np.copy(self.LabelsDistances[idx]),
                   'Sample':self.sample[idx],
                   'ID':self.IDs[idx],
                    'Path':self.path[idx],
                   }
            
        
#         Transform
        
        if self.transform:
            sample = self.transform(sample)
            
        
        return sample

def EpiTargetPaths(root):
    data=[]
    
    for subj in os.scandir(root):
        if subj.name[0]=='0':
            data.append({})
            data[-1]['Sample']=subj.name
            data[-1]['Path']=str(subj.path)
            data[-1]['MRI']=os.path.join(subj,'t2star_sumOverEchoes.nii')
            data[-1]['Contra']=os.path.join(subj,'t2star_sumOverEchoes_mask_HCcontra.nii')
            data[-1]['Ipsi']=os.path.join(subj,'t2star_sumOverEchoes_mask_HCipsi.nii')
            data[-1]['Ref']=os.path.join(subj,'t2star_sumOverEchoes_mask_refpoint.nii')
            bm=os.path.join(subj,'t2star_sumOverEchoes_mask_whole_brain.nii')
            am=os.path.join(subj,'mask.nii.gz')
            if os.path.isfile(am):
                data[-1]['Mask']=am
            if os.path.isfile(bm):
                data[-1]['ManualMask']=bm
        
    return data

def unique_elements(coll):
    unq = []
    for k in coll:
        if k not in unq:
            unq.append(k) 
    return unq

class GenericDataset(Dataset):
    
    def __init__(self,rootfolder,mriname,vol_list=None,maskname=None,
                 training=True,transform=None,foldfile=None):
        self.transform = transform
        self.MRIs = []
        self.vols = []
        self.foldlist = []
        self.mask = maskname
        self.masks = []
        self.training = training
        self.labels = vol_list
        for dirpath, dirnames, filenames in os.walk(rootfolder):
            
            if mriname in filenames:
                if foldfile is not None: 
                    with open(os.path.join(dirpath,foldfile),'r') as a:
                        
                        self.foldlist.append(int(a.read().split('\n')[0]))
                
                путь = os.path.join(dirpath,mriname)
                
                self.MRIs.append(путь)
                if training and (vol_list is not None):
                    nvs = [os.path.join(dirpath,v) for v in vol_list]
                    for v in nvs: assert os.path.isfile(v)
                    self.vols.append(nvs)
                if training and (maskname is not None):
                    m = os.path.join(dirpath,maskname)
                    assert os.path.isfile(m), m
                    self.masks.append(m)
    
    def __len__(self):
        return len(self.MRIs)
    
    def __getitem__(self,idx):
        sample = {}
        mri = niload(self.MRIs[idx])
        mris = [1] + list(mri.shape)
        mri = mri.reshape(mris)
        sample['MRI'] = mri
        
        labels = []
        if self.labels is not None:
            for name, volpath in zip(self.labels,self.vols[idx]):
                V = niload(volpath)
                S = V.shape
                S = [1] + list(S)
                labels.append(V.reshape(S))
            NBG = np.concatenate(labels,axis=0).sum(axis=0,keepdims=True)
            BG = np.zeros_like(labels[0])
            BG[NBG==0]=1
            labels.append(BG)
            labels = np.concatenate(labels,axis=0)
            sample['Labels'] = labels
            
        if self.mask is not None:
            M = niload(self.masks[idx])
            S = [1] + list(M.shape)
            sample['Mask'] = M.reshape(S)
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.labels is None: sample['Labels'] = 1
        if self.mask is None: sample['Mask'] = 1
        sample['Path'] = self.MRIs[idx]
        sample['Sample'] = self.MRIs[idx]
        
        return sample



class EpiTargetPreSS(Dataset):
    
    def __init__(self,rootfolder,transform=None,ManualOnly=True,ArtificialMask=False):
        self.transform=transform
        data=EpiTargetPaths(rootfolder)
        self.dataset=[]
        
        for x in data:
            get = (not ManualOnly or (ManualOnly and 'ManualMask' in x)) or ArtificialMask
            if get:
                MRI=niload(x['MRI'])
                shape=list(MRI.shape)
                MRI=MRI.reshape([1]+shape)
                if ManualOnly:
                    brainmask=niload(x['ManualMask']).reshape([1]+shape)
                    Labels = brainmask
                    ref=np.array((0,0,0))
                elif ArtificialMask:
                    brainmask=niload(x['Mask']).reshape([1]+shape)
                    
                    ref=niload(x['Ref'])
                    xg, yg, zg = np.meshgrid(range(shape[0]),range(shape[1]),range(shape[2]),indexing='ij')
                    X=np.sum(xg*ref)/shape[0]
                    Y=np.sum(yg*ref)/shape[1]
                    Z=np.sum(zg*ref)/shape[2]
                    ref=np.array((X,Y,Z))
                    
                    ipsi=niload(x['Ipsi'])
                    contra=niload(x['Contra'])
                    
                    NBG=ipsi+contra
                    BG=np.zeros_like(NBG)
                    BG[NBG==0]=1
                    Labels=np.zeros([3]+shape)
                    Labels[0,:,:,:]=ipsi
                    Labels[1,:,:,:]=contra
                    Labels[2,:,:,:]=BG
                    
                    
                else:
                    brainmask=np.zeros_like(MRI)
                    Labels = brainmask
                    ref=np.array((0,0,0))
                name=x['Sample']
                path=x['MRI']
                #assert ref.sum()==1, str(ref.sum()) + x['Ref']
                            
                
                self.dataset.append({
                    'MRI':MRI,
                    'Sample':name,
                    'Mask':brainmask,
                    'ID':path.split('/')[-2],
                    'Path':path,
                    'Labels':Labels,
                    'Ref':ref,
                    'Folder':x['Path']
                    })
                
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        
        sample = copy.deepcopy(self.dataset[idx])
        
#         Transform
        
        if self.transform:
            sample = self.transform(sample)
            
        
        return sample
        
    
allentries=['MRI','Mask','Ipsi','Contra','Ref','Labels','MDist','LDist']

class ShiftSlice():
    '''
    Shifts image by random amount in (-max_voxels, max_voxels) with
    probability=probability, only on x,y plane
    '''
    def __init__(self,max_voxels,probability):
        assert isinstance(max_voxels,(int))
        self.maxshift=max_voxels
        self.probability=probability
        
    def __call__(self,sample):
        if float(np.random.random(1))<=self.probability:
            
            if len(sample['MRI'].shape) ==3:
                delta=(0,np.random.uniform(-self.maxshift,self.maxshift),   np.random.uniform(-self.maxshift,self.maxshift))
            else:
                delta=(0,np.random.uniform(-self.maxshift,self.maxshift),   np.random.uniform(-self.maxshift,self.maxshift),  0)
            
            sample['MRI']=shift(sample['MRI'],delta,order=3)
            for k in allentries:
                if k in sample and k != 'MRI':  sample[k]=shift(sample[k],delta,order=0)
            
        
        return sample
    
class Normalizer():
    
    def __call__(self,sample):
        
        mean=np.nanmean(sample['MRI'])
        std=np.nanstd(sample['MRI'])
    
        sample['MRI']=(sample['MRI']-mean)/(np.sqrt(std))
        return sample
    
class RotSlice():
    '''
    Randomly rotates of an angle in [-MaxAngle, MaxAngle]
    with probability=probability
    else, the data is left unchanged
    always rotates on the x,y plane, z is bad
    rehsape: if using batches, must be False
    '''
    def __init__(self,MaxAngle,probability,reshape=False):

        self.MaxAngle=MaxAngle
        self.probability=probability
        self.reshape=reshape
    def __call__(self,sample):
        if float(np.random.random(1))<= self.probability:
            
            Ang=float(np.random.uniform(-self.MaxAngle,self.MaxAngle))
            
            RotMe = lambda samp,spline: rotate(samp,Ang,(1,2),reshape=self.reshape,order=spline)
            sample['MRI']=RotMe(sample['MRI'],3)
            for k in allentries:
                if k in sample and k != 'MRI':  sample[k]=RotMe(sample[k],0)
                
        return sample


def CropZoom2D(vol,factor,order):
    shape=vol.shape
    zoomed=zoom(vol,factor,order=order)
    
    
    diff=np.abs(np.array(zoomed.shape)-np.array(shape))
    if np.mean(factor)>1:
        A=zoomed[0, diff[1]//2:(shape[1]+diff[1]//2) , diff[2]//2:(shape[2]+diff[2]//2) ]
    
    else:
        BG=np.mean(vol[0,:,0])
        A=np.ones_like(vol)
        A*=BG
        A[0, diff[1]//2:(zoomed.shape[1]+diff[1]//2) , diff[2]//2:(zoomed.shape[2]+diff[2]//2) ]=zoomed
    
    return A.reshape(shape)

def CropZoom3D(vol,factor,order):
    shape=vol.shape
    zoomed=zoom(vol,factor,order=order)
    
    # s0=vol.shape[0]
    diff=np.abs(np.array(zoomed.shape)-np.array(shape))
    if np.mean(factor)>1:
        A=zoomed[:, diff[1]//2:(shape[1]+diff[1]//2) , diff[2]//2:(shape[2]+diff[2]//2) , diff[3]//2:(shape[3]+diff[3]//2) ]
    
    else:
        BG=np.mean(vol[0,:,0,0])
        A=np.ones_like(vol)*1.
        A*=BG
        A[:, diff[1]//2:(zoomed.shape[1]+diff[1]//2) , diff[2]//2:(zoomed.shape[2]+diff[2]//2),  diff[3]//2:(zoomed.shape[3]+diff[3]//2) ]=zoomed
    
    return A.reshape(shape)
    
    
class Rescale():
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, MaxScale,MinScale,probability,dim=3):
        
        self.MinScale = MinScale
        self.span = MaxScale-MinScale
        self.probability=probability
        
        if dim==3:
            self.zoom=CropZoom3D
        elif dim==2:
            self.zoom=CropZoom2D

    def __call__(self, sample):
        
        scaleFactor=self.MinScale + self.span * random.random()
        if len(sample['MRI'].shape) ==3:
            scaleFactor=(1,scaleFactor,scaleFactor)
        else:
            scaleFactor=(1,scaleFactor,scaleFactor,scaleFactor)
        
        if np.random.rand()<self.probability:
            
            sample['MRI']=self.zoom(sample['MRI'],scaleFactor,order=3)
            for k in allentries:
                if k != 'Ref':
                    # print(k)
                    if k in sample and k != 'MRI':  
                        sample[k]=self.zoom(sample[k],scaleFactor,order=0)
        
        
        
        return sample
    
def OneTensor(vol,device='cuda'):
    return torch.from_numpy(vol).float()
    
    
class ToTensor():
    
    def __call__(self,sample,device='cuda'):
        
        
        for k in allentries:
                if k in sample: sample[k] = OneTensor(sample[k],device)
        return sample

