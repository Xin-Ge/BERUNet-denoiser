# Enhancing Border Learning for Better Image Denoising

 Xin Ge, Yu Zhu, Liping Qi, Yaoqi Hu, Jinqiu Sun and Yanning Zhang

Paper:[![paper](https://img.shields.io/badge/MDPI-Mathematics-blue.svg)](https://www.mdpi.com/2227-7390/13/7/1119)

<hr />

> **Abstract:** Deep neural networks for image denoising typically follow an encoder–decoder model, with convolutional (Conv) layers as essential components. Conv layers apply zero padding at the borders of input data to maintain consistent output dimensions. However, zero padding introduces ring-like artifacts at the borders of output images, referred to as border effects, which negatively impact the network’s ability to learn effective features. In traditional methods, these border effects, associated with convolutional/deconvolutional operations, have been mitigated using patch-based techniques. Inspired by this observation, patch-wise denoising algorithms were explored to derive a CNN architecture that avoids border effects. Specifically, we extend the patch-wise autoencoder to learn image mappings through patch extraction and patch-averaging operations, demonstrating that the patch-wise autoencoder is equivalent to a specific convolutional neural network (CNN) architecture, resulting in a novel residual block. This new residual block includes a mask that enhances the CNN’s ability to learn border features and eliminates border artifacts, referred to as the Border-Enhanced Residual Block (BERBlock) . By stacking BERBlocks, weconstructed a U-Net denoiser (BERUNet). Experiments on public datasets demonstrate that the proposed BERUNet achieves outstanding performance. The proposed network architecture is built on rigorous mathematical derivations, making its working mechanism highly interpretable. 
<hr />

## Notice

After the publication of our paper, we **systematically revised** the implementation and **retrained all model parameters**.  
As a result, the current test results may **differ slightly** from those reported in the paper:

- **Gaussian noise removal** performance is **better** than in the published results.  
- **Real-world noise removal** performance is **worse** than in the published results.

---
## Installation

Running **BERUNet** requires the following dependencies: **PyTorch**, **OpenCV**, **NumPy**, and **fvcore**.

---
## Data and Pretrained Model Preparation

### 🔹 Gaussian Noise Data

**Training Data:** Download from [Baidu Pan](https://github.com/cszn/KAIR) or [Google Drive](https://github.com/cszn/KAIR), which includes BSD, WED, DIV2K, and Flickr2K datasets. After downloading, place them in the following directory:
```
|--trainsets 
   |--trainH
```

**Testing Data:** Download from [Baidu Pan](https://pan.baidu.com/s/19St07LbHyUKf-25urcC7Bg?pwd=mahv) or [Google Drive](https://github.com/cszn/KAIR), which includes Set12, BSD68, Kodak24, McMaster, and Urban100 datasets. After downloading, organize them as follows:
```
|-- testsets
   |-- original
      |-- Set12
      |-- BSD68
      |-- Kodak24
      |-- McMaster
      |-- Urban100
```
### 🔹 Real-World Noise Data

**Training Data:** Download the **SIDD Training Set** from [Baidu Pan](https://pan.baidu.com/s/164cI2wbqwBVc7YaSKrLn4A?pwd=7nzd) or [Google Drive](https://github.com/cszn/KAIR). After downloading, place them in the following directory:
```
|-- trainsets
   |-- SIDD
      |-- noisy
      |-- groundtruth
```
**Testing Data:** Download the **SIDD Validation Set** from [Baidu Pan](https://pan.baidu.com/s/1JSyCpjLaZBWvHUCFlpeRyA?pwd=uvhn) or [Google Drive](https://github.com/cszn/KAIR). After downloading, place them in the following directory:
```
|-- testsets
   |-- original
      |-- SIDD
         |-- noisy
         |-- groundtruth
```
### 🔹 Pretrained Models
Download pretrained models from [Baidu Pan](https://pan.baidu.com/s/1nuzIsg1lNqmQ3Q3TKe3HPw?pwd=8ppt) or [Google Drive](https://drive.google.com/file/d/1xqiHJjwvATTQc0WjkrfE7MM0oLqm7dPk/view?usp=sharing). After downloading, place them in the following directory:
```
|-- model_zoo
   |-- BERUNet
```

---
## Testing

After placing the testing data and pretrained models in the specified directories, run the following Python files. 
Before execution, make sure to configure the testing dataset, color mode, and noise level within each `.py` file.

### 🔹 Gaussian Noise — Non-Blind Denoising
```
python main_test_BERUNet_v2_Gaussian.py
```
### 🔹 Gaussian Noise — Blind Denoising (not reported in the paper)
```
python main_test_BERUNet_Blind_v2_Gaussian.py
```
### 🔹 Real-World Noise — Blind Denoising
```
python main_test_BERUNet_Blind_v2_Real.py
```

---
## Training and Evaluation

After placing the training data in the specified directories, modify the corresponding JSON file under the `options/` folder according to your hardware setup, then execute the following command lines:

### 🔹 Grayscale Gaussian Noise — Non-Blind Denoising
```
python main_train.py --opt options/train_BERUNet_v2_nb8_256_g020_64_Xavier_StepL2_1e4_50_gray.json
```
### 🔹 Color Gaussian Noise — Non-Blind Denoising
```
python main_train.py --opt options/train_BERUNet_v2_nb8_256_g020_64_Xavier_StepL2_1e4_50_color.json
```
### 🔹 Grayscale Gaussian Noise — Blind Denoising (not reported in the paper)
```
python main_train.py --opt options/train_BERUNet_Blind_v2_nb8_256_g020_64_Xavier_StepL2_1e4_50_gray.json
```
### 🔹 Color Gaussian Noise — Blind Denoising (not reported in the paper)
```
python main_train.py --opt options/train_BERUNet_Blind_v2_nb8_256_g020_64_Xavier_StepL2_1e4_50_color.json
```
### 🔹 Real-World Noise — Blind Denoising
```
python main_train.py --opt options/train_BERUNet_Blind_v2_nb8_256_g020_Xavier_CosCycL1_1e4_real.json
```

💡 **Tip:**  
When running the above commands on a remote **Linux server**, it is recommended to use **tmux** to start a virtual session to prevent training from being interrupted.

---
## Citation
If you use BERUNet, please consider citing:

    @article{ge2025enhancing,
        title={Enhancing Border Learning for Better Image Denoising},
        author={Ge, Xin and Zhu, Yu and Qi, Liping and Hu, Yaoqi and Sun, Jinqiu and Zhang, Yanning},
        journal={Mathematics},
        volume={13},
        number={7},
        pages={1119},
        year={2025},
        publisher={MDPI AG}
    }

---
## Contact
Should you have any question, please contact gxin@mail.nwpu.edu.cn

**Acknowledgment:** This code is based on the [KAIR](https://github.com/cszn/KAIR) framework. 

## Our Related Works
- Enhancing the Noise Robustness of Sparse-form Patches for Image Denoising, KBS 2025. [Paper](https://www.sciencedirect.com/science/article/pii/S0950705125012717) | [Code](https://github.com/Xin-Ge/Feature-Patch-Denoiser)
