# XpressBio Net (XB-Net)

## Introduction
This is a place-holder repository for future publication of the network and weights.
More contents will be available soon.

For information or to report bugs, please email [Tianqi Guo](https://www.linkedin.com/in/tianqi-guo-purdue/) via tqguo246@gmail.com

### Tasks to complete:
- [x] Have a cup of coffee

### For CTC participation: 
- [x] Scripts for image pre-processing and tertiary training targets
- [x] Network implementation 
- [x] Scripts for training
- [x] Scripts for inference and tracking

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
- Other packages: imageio(2.8.0), numpy(1.18.4), scipy(1.4.1), pillow(8.1.0), libtiff(4.2.0)
- Official evaluation tool: [download](http://celltrackingchallenge.net/evaluation-methodology/), unzip, place under `./eval` folder

A full list of the conda environment packages can be found in `full_env.txt`, as output by `conda list --json`.

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
- Under current dir, run python scripts for 2D or 3D datasets by 
  ```
  python imgPre_2D_GT+ST.py DIC-C2DH-HeLa
  python imgPre_3D_GT+ST.py Fluo-N3DH-CHO
  ```
- Normalized cell images and tertiary training targets will be written under `datasets\pre\GT+ST\[dataset_name]`
  ```
  datasets
  ├──raw  
  └──pre\GT+ST                 (For data configuration GT+ST)                            
      └── DIC-C2DH-HeLa        (Name of the dataset)
          ├── 01               (Frames with ST annotations as the training set: normalized image\tertiary target pairs in npy)
          ├── 01_GT\SEG        (Original instance masks in tiff format for performance evaluation on the training set)
          ├── 01_RES           (Placeholder folder to write evaluation result masks)
          ├── 02               (Frames with GT annotations as the validation set: normalized image\tertiary target pairs in npy)
          ├── 02_GT\SEG        (Original instance masks in tiff format for performance evaluation on the validation set)
          ├── 02_RES           (Placeholder folder to write evaluation result masks)
          └── sample_pairs.png (Randomly picked frames for visualization)
  ```  
- Examples for sample_pairs.png (DIC-C2DH-HeLa)
  ![sample_pairs](/pics/sample_pairs.png)
- **Note**: only scripts for GT+ST data configuration is included. Others can be obtained with minor modifications.

### Training models
- Under current directory, run command
  ```
  python -u train_model.py
  ```
- Current training script uses two datasets as example: `DIC-C2DH-HeLa` and `Fluo-N3DH-CHO`
- Successfull launching should result in printouts similar to
  ```
  Result DIR =  ./datasets/pre/GT+ST/train_demo
  Tesla V100-PCIE-32GB  
  Training 2 cases.
  case result dir: ./datasets/pre/GT+ST/train_demo/DIC-C2DH-HeLa
  Training set: # samples: 150 , Image & target sizes: (1, 272, 272) (256, 256) , # batches: 19 , batchsize: 8
  Validation set:  # samples: 18 , Image & target sizes: (1, 528, 528) (512, 512)

  case result dir: ./datasets/pre/GT+ST/train_demo/Fluo-N3DH-CHO
  Training set: # samples: 920 , Image & target sizes: (1, 272, 272) (256, 256) , # batches: 115 , batchsize: 8
  Validation set:  # samples: 43 , Image & target sizes: (1, 464, 528) (443, 512)

  Training settings: LR = 0.000100, weights = (1,10,5)
  Start training from Epoch 0 towards Epoch 26000.
  Best seg score so far: 0.0000 at Epoch -1.

  Current RAM usage is 0002.72 MB; Peak was 0010.48 MB
  Current GPU usage is 0046.19 MB; Peak was 0046.19 MB

  [09/25 00:08:23] Itr: 00002, Time: ( 0:00:01.218137 , 0.61), GPU: (0232.76, 4383.96) MB, Loss: (1.2933, 0.0000), LR: 1.00E-04, Valid SEG: ( 0.0000 0.0000 ). Mean: 0.0000, Best: 0.0000 @ Itr 2.
  [09/25 00:10:15] Itr: 00260, Time: ( 0:01:20.623416 , 0.31), GPU: (0232.76, 4524.44) MB, Loss: (0.4767, 0.0000), LR: 1.00E-04, 
  [09/25 00:11:35] Itr: 00520, Time: ( 0:02:40.380476 , 0.31), GPU: (0232.76, 4524.44) MB, Loss: (0.4166, 0.0000), LR: 1.00E-04, 
  ```
- Training results are under the folder `./datasets/pre/GT+ST/train_demo`
  - `checkpoint.pth`: training states for optimizer, network, etc. used for resuming training.
  - `loss_history.npy`: loss history, validation IoU information, etc.
  - `trained_model.pth`: best model with the highest averaged IoU across datasets, used for later inferencing.

### Training models
- Under current directory, run command
  ```
  python -u test_model.py path_to_weights path_to_imgs is_volume obj_min_size hole_min_size search_radius
  ```
- Arguments:
  - path_to_weights: trained weights in `*.pth` format
  - path_to_imgs: raw test images in `*.tif` format
  - is_volume: if current dataset is 3D (1 for yes, 0 for no)
  - obj_min_size: objects smaller than this value will be ignored in tracking
  - hole_min_size: holes within objects that are smaller will be filled 
  - search_radius: maximum distance to search for matching cells in the next frame
- Command example:
  ```
  python -u test_model.py ./trained_models/trained_model.pth ./dataset/raw/DIC-C2DH-HeLa/01 0 500 100 50
  ```
 - Resulting masks in `*tiff` format will be writted under `./test_results/` directory.
 
## References:
1. Tianqi Guo, Yin Wang, Luis Solorio, and Jan P. Allebach (2021). Training a universal instance segmentation network for live cell images of various cell types and imaging modalities. (Manuscript in preparation)
2. Guo, T., Ardekani, A. M., & Vlachos, P. P. (2019). Microscale, scanning defocusing volumetric particle-tracking velocimetry. Experiments in Fluids, 60(6), 89. [Link](https://link.springer.com/article/10.1007/s00348-019-2731-4)
3. Jun, B. H., Guo, T., Libring, S., Chanda, M. K., Paez, J. S., Shinde, A., ... & Solorio, L. (2020). Fibronectin-expressing mesenchymal tumor cells promote breast cancer metastasis. Cancers, 12(9), 2553. [Link](https://doi.org/10.3390/cancers12092553)
4. Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham. [Link](https://arxiv.org/abs/1505.04597)
