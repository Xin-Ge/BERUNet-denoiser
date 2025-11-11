
# Enhancing Border Learning for Better Image Denoising

 Xin Ge, Yu Zhu, Liping Qi, Yaoqi Hu, Jinqiu Sun and Yanning Zhang

[![paper](https://img.shields.io/badge/MDPI-Mathematics-blue.svg)](https://www.mdpi.com/2227-7390/13/7/1119)


<hr />

> **Abstract:** * Deep neural networks for image denoising typically follow an encoder–decoder model, with convolutional (Conv) layers as essential components. Conv layers apply zero padding at the borders of input data to maintain consistent output dimensions. However, zero padding introduces ring-like artifacts at the borders of output images, referred to as border effects, which negatively impact the network’s ability to learn effective features. In traditional methods, these border effects, associated with convolutional/deconvolutional operations, have been mitigated using patch-based techniques. Inspired by this observation, patch-wise denoising algorithms were explored to derive a CNN architecture that avoids border effects. Specifically, we extend the patch-wise autoencoder to learn image mappings through patch extraction and patch-averaging operations, demonstrating that the patch-wise autoencoder is equivalent to a specific convolutional neural network (CNN) architecture, resulting in a novel residual block. This new residual block includes a mask that enhances the CNN’s ability to learn border features and eliminates border artifacts, referred to as the Border-Enhanced Residual Block (BERBlock) . By stacking BERBlocks, weconstructed a U-Net denoiser (BERUNet). Experiments on public datasets demonstrate that the proposed BERUNet achieves outstanding performance. The proposed network architecture is built on rigorous mathematical derivations, making its working mechanism highly interpretable. The code and all pretrained models are publicly available.* 
<hr />

## Notice

This code is a **heavily modified version** based on Kai Zhang’s [KAIR framework](https://github.com/cszn/KAIR).  
After the publication of our paper, we **systematically revised** the implementation and **retrained all model parameters**.  
As a result, the current test results may **differ slightly** from those reported in the paper:

- **Gaussian noise removal** performance is **better** than in the published results.  
- **Real-world noise removal** performance is **worse** than in the published results.

## Installation

See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run Restormer.


## Data and Pretrained Model Preparation

### Gaussian Noise Data

**Training Data:** Download from [Baidu Pan](https://github.com/cszn/KAIR), which includes BSD, WED, DIV2K, and Flickr2K datasets. After downloading, place them in the following directory:
|--trainsets 
   |--trainH

**Testing Data:** Download from [Baidu Pan](https://github.com/cszn/KAIR), which includes Set12, BSD68, Kodak24, McMaster, and Urban100 datasets. After downloading, organize them as follows:
|-- testsets
   |-- original
      |-- Set12
      |-- BSD68
      |-- Kodak24
      |-- McMaster
      |-- Urban100

### Real-World Noise Data

**Training Data:** Download the **SIDD Training Set** from [Baidu Pan](https://github.com/cszn/KAIR). After downloading, place them in the following directory:
|-- trainsets
   |-- SIDD

**Testing Data:** Download the **SIDD Validation Set** from [Baidu Pan](https://github.com/cszn/KAIR). After downloading, place them in the following directory:
|-- testsets
   |-- original
      |-- SIDD

### Pretrained Models
Download pretrained models from [Baidu Pan](https://github.com/cszn/KAIR). After downloading, place them in the following directory:
|-- model_zoo
   |-- BERUNet
