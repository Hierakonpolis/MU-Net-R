# MU-Net-R

**[Automatic hippocampus segmentation in rat brain MRI after traumatic brain injury](let this be a link to the preprint)**

This convolutional neural network is designed to perform skull-stripping and region segmentation on mouse brain MRI. The conda environment we used is specified in the .yml file included in this repository.

## Training
To use this script the data is assumed to be organized in a series of folders, one for each sample, sharing the same root folder. Each folder needs to contain a file for the MRI scan and one for each ROI you want segmented, as a binary mask in the same space as the MRI. Files need to have the same name in each folder. Do not forget to set the `--train` option for training. Defaults are as set in our paper, check the usage section below for how to change them if needed. The number of blocks on the encoder and decoder branches can simply be changed by changing the lenght of the `--kernels` option: 8 16 32 64 64 64 64 mean you will train a network with 7 blocks on each branch, 16 32 means there will only be two blocks.

## Training scripts and models
These folders contain the scripts we used for the experiments in the paper and the files containing the parameters for the models we trained. They are here for transparency purposes and they are not necessary in practice for using this release. To use this method you will most likely have to retrain the network on your own data.

## Usage
```
python3 segmenter.py [-h] [--savefolder SAVEFOLDER] [--train] [--mask MASK] [--labels [LABELS [LABELS ...]]] [--foldfile FOLDFILE] [--folds FOLDS] [--workers WORKERS] [--twoD] [--maxtime MAXTIME] [--kernels [KERNELS [KERNELS ...]]] [--patience PATIENCE] [--maxepochs MAXEPOCHS] [--maskweight MASKWEIGHT]
                    Datafolder MRIname modelname
Train an ensamble of neural networks and use it to segment MRI volumes

positional arguments:
  Datafolder            Root folder of the dataset, for training or labeling. Each sample should be located in a separate folder, and for training mask files should be located in the same folder. Files should have the same names in each folder.
  MRIname               Name of the MRI file for each sample.
  modelname             Define a name of the model, to load or save. Do not add a file extension.

optional arguments:
  -h, --help            show this help message and exit
  --savefolder SAVEFOLDER
                        Specify a directory where to save or from which to load the model, if none is specified the current working dir will be used.
  --train               Train an ensamble of network.
  --mask MASK           Name of brain mask file, optional.
  --labels [LABELS [LABELS ...]]
                        Name of ROI mask files, listed after the argument, e.g. --labels file1.nii, file1.nii.
  --foldfile FOLDFILE   If you want to manually assign each sample to a specific fold, specify here the name of a text file in each folder indicating the specific fold it was assigned to, with a number starting from 0. Example: --foldfile fold.txt, where fold.txt contains the number 2. Mark each fold with an
                        integer.
  --folds FOLDS         Number of folds for your dataset, to use as different validation sets for each model.
  --workers WORKERS     Number of external parallel processes to spawn when loading data. Provide any integer larger than zero to use parallel computing and speed up loading the data.
  --twoD                Use 2D filters instead of defaulting to 3D filters. Can be useful for anisotropic data. The direction of anisotropy is assumed to be on the third axis of the nii volumes: filters and pooling will be (x,x,1).
  --maxtime MAXTIME     Maximum time to train each network in the ensemble, in minutes.
  --kernels [KERNELS [KERNELS ...]]
                        numer of kernels for convolutions in each block, shallowest to deepest e.g. --kernels 16 36 64 64
  --patience PATIENCE   After spending these many epochs with no validation set improvements, stop training.
  --maxepochs MAXEPOCHS
                        Maximum number of epochs spent training. By default set to infinity.
  --maskweight MASKWEIGHT
                        Weight parameter for the brain mask. If you get stuck training and only learning the brain mask, make this smaller. E.g. 0.01 or 0.001.
```


