import os 
import sys
import math
import glob
import torch
import shutil
import imageio
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
from scipy.ndimage import measurements
from scipy import ndimage as ndi
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from XBNet import *

def main():
    
    weight_path = sys.argv[1]
    input_data_path = sys.argv[2]
    is_volume = (int(sys.argv[3])==1)
    obj_min_size = int(sys.argv[4])
    hole_min_size = int(sys.argv[5])
    search_radius = int(sys.argv[5])
    
    formats = '/mask%0.4d.tif'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = "./test_results/"
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir) 
 
    model = XBNet().to(device)
    model.load_state_dict(torch.load(weight_path))
    model.train()
    
    img_list = sorted(glob.glob(os.path.join(input_data_path,'*.tif')))
    N_testing = len(img_list)
    
    intensity_min = np.inf
    intensity_max = 0
    
    for image_path in img_list:
        if is_volume:
            img = np.asarray(imageio.volread(image_path), dtype=np.double)
        else:
            img = np.asarray(imageio.imread(image_path), dtype=np.double)
        intensity_min = np.minimum(intensity_min, img.min())
        intensity_max = np.maximum(intensity_max, img.max())
    intensity_range = intensity_max - intensity_min
    
    if is_volume:
        _, h, w = img.shape
    else:
        h, w = img.shape
    H = (math.ceil(h/16)+1) * 16
    W = (math.ceil(w/16)+1) * 16
    pad_H_0 = (H - h)//2
    pad_H_1 = H - h - pad_H_0
    pad_W_0 = (W - w)//2
    pad_W_1 = W - w - pad_W_0
    pads = (pad_H_0, pad_H_1, pad_W_0, pad_W_1)

    segmented = []
    CoMs = []
    track_id = 0
    float_inf = float('inf')
    
    for t, image_path in enumerate(img_list):
        if is_volume:
            img = np.asarray(imageio.volread(image_path), dtype=np.double)
        else:
            img = np.asarray(imageio.imread(image_path), dtype=np.double)
        images_n = (img - intensity_min)/intensity_range
        
        if is_volume:
            cell_padded = []
            for images_n_i in images_n:
                imgs_testing = np.asarray(np.pad(images_n_i,((pad_H_0,pad_H_1), (pad_W_0,pad_W_1)),mode='symmetric'),dtype=np.double)
                X = torch.from_numpy(imgs_testing).float().to(device)
                prediction = model(X.view([1,1,H,W])).cpu().detach().numpy()
                cell_pred = np.logical_and(prediction[:,0,:,:] < prediction[:,2,:,:],prediction[:,1,:,:] < prediction[:,2,:,:])
                cell_padded_i = cell_pred[0,pad_H_0:-pad_H_1,pad_W_0:-pad_W_1]
                
                cell_padded_i = morphology.remove_small_objects(cell_padded_i, min_size=obj_min_size, in_place=True)
                cell_padded_i = morphology.remove_small_holes(cell_padded_i, area_threshold=hole_min_size, in_place=True)
                
                cell_padded.append(cell_padded_i)
                
            cell_padded = np.asarray(cell_padded,dtype=np.double)
            
            seg_RS_i, n_cells = measurements.label(cell_padded)
            
            CoM = measurements.center_of_mass(cell_padded, seg_RS_i, range(1,n_cells+1))
            tracked = np.zeros((len(images_n), h,w),dtype='uint16')
            
            CoMs_new = []
            
            for i, z0, y0, x0, t0 in CoMs:
                dxdydz = [((xx-x0)**2+(yy-y0)**2+(zz-z0)**2)**0.5 for zz, yy, xx in CoM]
                if dxdydz:
                    j = np.argmin(dxdydz)
                    if dxdydz[j] < search_radius:
                        tracked[(seg_RS_i == (j+1))] = i
                        CoMs_new += [(i, CoM[j][0], CoM[j][1], CoM[j][2], t0)]
                        CoM[j] = (float_inf, float_inf, float_inf)
                        continue
                    
            for j, (z, y, x) in enumerate(CoM):
                if y != float_inf:
                    track_id += 1
                    tracked[(seg_RS_i == j+1)] = track_id
                    CoMs_new += [(track_id, z, y, x, t)]
            CoMs = CoMs_new
            
            imageio.volwrite(out_dir+formats%t, tracked)
            
        else:
            
            imgs_testing = np.asarray(np.pad(images_n,((pad_H_0,pad_H_1), (pad_W_0,pad_W_1)),mode='symmetric'),dtype=np.double)
            X = torch.from_numpy(imgs_testing).float().to(device)
            prediction = model(X.view([1,1,H,W])).cpu().detach().numpy()
            cell_pred = np.logical_and(prediction[:,0,:,:] < prediction[:,2,:,:],prediction[:,1,:,:] < prediction[:,2,:,:])
            cell_padded = cell_pred[0,pad_H_0:-pad_H_1,pad_W_0:-pad_W_1]
            
            cell_padded = morphology.remove_small_objects(cell_padded, min_size=obj_min_size, in_place=True)
            cell_padded = morphology.remove_small_holes(cell_padded, area_threshold=hole_min_size, in_place=True)
            
            seg_RS_i, n_cells = measurements.label(cell_padded)
            CoM = measurements.center_of_mass(cell_padded, seg_RS_i, range(1,n_cells+1))
            
            tracked = np.zeros((h,w),dtype='uint16')
            
            CoMs_new = []
            
            for i, y0, x0, t0 in CoMs:
                dxdy = [((xx-x0)**2+(yy-y0)**2)**0.5 for yy, xx in CoM]
                j = np.argmin(dxdy)
                if dxdy[j] < search_radius:
                    tracked[(seg_RS_i == (j+1))] = i
                    CoMs_new += [(i, CoM[j][0], CoM[j][1], t0)]
                    CoM[j] = (float_inf, float_inf)
                    
            for j, (y, x) in enumerate(CoM):
                if y != float_inf:
                    track_id += 1
                    tracked[(seg_RS_i == j+1)] = track_id
                    CoMs_new += [(track_id, y, x, t)]
            CoMs = CoMs_new
                
            im = Image.fromarray(tracked)
            im.save(out_dir+formats%t, compression="tiff_lzw", save_all=True)
    
if(__name__ == "__main__"):
    main()
    print('Processing completed!')