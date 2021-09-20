# XpressBio Net (XB-Net)

## Introduction
This is a place-holder repository for future publication of the network and weights.
More contents will be available soon.

For information or to report bugs, please email [Tianqi Guo](https://www.linkedin.com/in/tianqi-guo-purdue/) via tqguo246@gmail.com

### Tasks to complete:
- [x] Have a cup of coffee

### For CTC participation: 
- [ ] Scripts for image pre-processing and tertiary training targets
- [ ] Network implementation 
- [ ] Scripts for training 6 data configurations
- [ ] Scripts for inference and tracking

### For XB-Net manuscript: 
- [ ] Implementation of training schemes
- [ ] Configurable training settings
- [ ] Implementation of different backbones
- [ ] Integration of individual modules


## Dependencies
- Platform: Centos 7.7
- Python version: 3.7
- Cuda version: 11.0.3
- PyTorch version: 1.7.0
- Additional dependencies: imageio(2.8.0), numpy(1.18.4), scipy(1.4.1), pillow(8.1.0), libtiff(4.2.0)

A full list of the conda enviroment packages can be found in `full_env.txt`, as output by `conda list --json`.

## To use current repo
### Image preprocessing and generate tertiary training targets
- Download and unzip datasets from CTC website [Link](http://celltrackingchallenge.net/2d-datasets/)
- Place unzipped training sets under `datasets\raw\[dataset_name]`, and it should have the following folder structure (e.g. DIC-C2DH-HeLa)
  ```
  datasets
  ├──raw
  │   └── DIC-C2DH-HeLa   (Name of the dataset)
  │       ├── 01          (Raw images for video sequence 01)
  │       ├── 01_ST       (ST annotations for video sequence 01)
  │       ├── 01_GT       (GT annotations for video sequence 01)
  │       ├── 02          (Raw images for video sequence 02)
  │       ├── 02_ST       (ST annotations for video sequence 02)
  │       └── 02_GT       (GT annotations for video sequence 02)
  └──pre
  ```    
- Under current dir, run python scripts for 2D or 3D datasets by `datasets\pre\[dataset_name]`
  ```
  python imgPre_2D_GT+ST.py DIC-C2DH-HeLa
  ```
- Normalized cell images and tertiary training targets will be written under
  ```
  datasets
  ├──raw  
  └──pre\GT+ST                  (For data configuration GT+ST)                            
      └── DIC-C2DH-HeLa         (Name of the dataset)
          ├── 01                (Frames with ST annotations as the training set: normalized image\tertiary target pairs in npy)
          ├── 01_GT\SEG         (Original instance masks in tiff format for performance evaluation on the training set)
          ├── 01_RES            (Placeholder folder to write evaluation result masks)
          ├── 02                (Frames with GT annotations as the validation set: normalized image\tertiary target pairs in npy)
          ├── 02_GT\SEG         (Original instance masks in tiff format for performance evaluation on the validation set)
          ├── 02_RES            (Placeholder folder to write evaluation result masks)
          └── sample_pairs.png  (Randomly picked frames for visualization)
  ```  
- Examples for sample_pairs.png (DIC-C2DH-HeLa)
- **Note**: only scripts for GT+ST data configuration is included. Others can be obtained with minor modifications.

## References:
1. Tianqi Guo, Yin Wang, Luis Solorio, and Jan P. Allebach (2021). Training a universal instance segmentation network for live cell images of various cell types and imaging modalities. (Manuscript in preparation)
2. Guo, T., Ardekani, A. M., & Vlachos, P. P. (2019). Microscale, scanning defocusing volumetric particle-tracking velocimetry. Experiments in Fluids, 60(6), 89. [Link](https://link.springer.com/article/10.1007/s00348-019-2731-4)
3. Jun, B. H., Guo, T., Libring, S., Chanda, M. K., Paez, J. S., Shinde, A., ... & Solorio, L. (2020). Fibronectin-expressing mesenchymal tumor cells promote breast cancer metastasis. Cancers, 12(9), 2553. [Link](https://doi.org/10.3390/cancers12092553)
4. Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham. [Link](https://arxiv.org/abs/1505.04597)
