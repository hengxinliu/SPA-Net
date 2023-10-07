# A deep convolutional neural network for the automatic segmentation of glioblastoma brain tumor: Joint spatial pyramid module and attention mechanism network

This repository is the work of "_A deep convolutional neural network for the automatic segmentation of glioblastoma brain tumor: Joint spatial pyramid module and attention mechanism network_" based on **pytorch** implementation.The multimodal brain tumor dataset (BraTS 2019) could be acquired from [here](https://www.med.upenn.edu/cbica/brats-2019/).

## SPA-Net

<center>Architecture of  SPA-Net</center>
<div  align="center">  
 <img src="https://github.com/hengxinliu/ADHDC-Net/blob/main/fig/SPA-Net.jpg"
     align=center/>
</div>




## Requirements
* python 3.8
* pytorch 1.6.0
* nibabel
* pickle 
* imageio
* pyyaml

## Implementation

Download the BraTS2019 dataset and change the path:

```
experiments/PATH.yaml
```

### Data preprocess
Convert the .nii files as .pkl files. Normalization with zero-mean and unit variance . 

```
python preprocess.py
```

(Optional) Split the training set into k-fold for the **cross-validation** experiment.

```
python split.py
```

### Training

Sync bacth normalization is used so that a proper batch size is important to obtain a decent performance. Multiply gpus training with batch_size=8 is recommended.The experimental environment in this study was as follows: CPU Intel® Core i9-9900X 3.5GHZ, GPU GTX2080Ti (11GB) × 4, and a Ubuntu 16.04 operating system..
d
```
python train_all.py --gpu=0,1,2,3 --cfg=EMMNet_1 --batch_size=8
```


### Test

You could obtain the resutls as paper reported by running the following code:

```
python test.py --mode=1 --is_out=True --verbose=True --use_TTA=True --postprocess=True --snapshot=True --restore=model_last.pth --cfg=EMMNet_1 --gpu=0
```
Then make a submission to the online evaluation server.



## Acknowledge

1. [DMFNet](https://github.com/China-LiuXiaopeng/BraTS-DMFNet)
2. [BraTS2018-tumor-segmentation](https://github.com/ieee820/BraTS2018-tumor-segmentation)
3. [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
4. [HDC-Net](https://github.com/luozhengrong/HDC-Net)

