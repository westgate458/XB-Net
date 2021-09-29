'''
Author:
Tianqi Guo 
Yin Wang
EISL-A @ Purdue University - School of Electrical and Computer Engineering
Do not use for commercial purposes. All rights reserved.
Contact:
guo246@purdue.edu
'''

import os
import math
import glob
import shutil
import random
import imageio

import tracemalloc
import numpy as np

from scipy.ndimage import measurements

import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader

import shlex
from datetime import datetime
from subprocess import Popen, PIPE

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

def dimension_matching(cell_pred, y):
    
    _, h, w = y.shape
    _, H, W = cell_pred.shape
    
    if H > h:
        pad_H_0 = (H - h)//2
        pad_H_1 = H - h - pad_H_0
        pad_W_0 = (W - w)//2
        pad_W_1 = W - w - pad_W_0
        cell_padded = cell_pred[:,pad_H_0:-pad_H_1,pad_W_0:-pad_W_1]
    elif H < h:
        pad_H_0 = (h - H)//2
        pad_H_1 = h - H - pad_H_0
        pad_W_0 = (w - W)//2
        pad_W_1 = w - W - pad_W_0
        cell_padded = np.pad(cell_pred, ((0,0),(pad_H_0,pad_H_1), (pad_W_0,pad_W_1)), 'constant', constant_values=0)
    
    return cell_padded

@torch.no_grad()
def eval_performance_test(model, dataset_loader, device, result_dir, set_id):
    
    case_dir = os.path.join(result_dir)
    
    RS_dir = os.path.join(case_dir,set_id+'_RES')
    
    if os.path.exists(RS_dir):
        shutil.rmtree(RS_dir)
    os.makedirs(RS_dir,exist_ok=True)
    
    counter = 0
    
    for X, y in dataset_loader:
        
        pred = model(X.to(device))
        prediction = pred.cpu().detach().numpy()  # [N, 2, H, W]   
        cell_pred = np.logical_and(prediction[:,0,:,:] < prediction[:,2,:,:],prediction[:,1,:,:] < prediction[:,2,:,:])
        cell_padded = dimension_matching(cell_pred, y)
        
        for cell_padded_i in cell_padded:
            if 'Fluo-C3DH-A549' in result_dir:
                seg_RS_i = cell_padded_i
            else:
                seg_RS_i, _ = measurements.label(cell_padded_i)
            imageio.imwrite(os.path.join(RS_dir,'mask%0.3d.tif'%counter), np.asarray(seg_RS_i,dtype=np.uint16))
            counter += 1
    
    cmd = "./eval/Linux/SEGMeasure "+case_dir+" "+set_id+" 3"
    process = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    out, err = process.communicate()
    
    #print(cmd, out, err)
    
    exitcode = process.returncode
    s = str(out, 'utf-8')
    
    file = open(os.path.join(RS_dir,'SEG_log.txt'), 'r') 
    Lines = file.readlines() 
      
    scores = []
    for line in Lines: 
        if 'J' in line:
            s = line.strip().split()[1]
            score = float(''.join([c for c in s if c == '.' or c.isdigit()]))
            scores.append(score)
    file.close() 
    
    det_score = np.count_nonzero(scores)/len(scores)
    seg_score = np.mean(scores)
    #float(''.join([c for c in s if c == '.' or c.isdigit()]))
    return(det_score, seg_score, scores)
    
def show_usage():
    
    print('')
    current, peak = tracemalloc.get_traced_memory()
    print("Current RAM usage is %07.2f MB; Peak was %07.2f MB"%(current/10**6,peak/10**6))
    current, peak = torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
    print("Current GPU usage is %07.2f MB; Peak was %07.2f MB"%(current/10**6,peak/10**6))
    print('')
    
class LoadDataset(data_utils.Dataset):    

    def __init__(self, data_path, crop_win, data_aug, pad):
        self.dataset_list = sorted(glob.glob(os.path.join(data_path,'*.npy')))        
        self.crop_win = crop_win        
        self.data_aug = data_aug
        self.pad = pad   
        
    def __getitem__(self, index):
        img, lbl = np.load(self.dataset_list[index],allow_pickle=True)
        
        if (self.crop_win == 0):
            img_aug = img
            lbl_aug = lbl 
        else:
            _, r, c = img.shape     
            max_r = r - self.crop_win
            max_c = c - self.crop_win
            #print(img.shape,self.crop_win)
            
            r1 = random.randint(0, max_r)
            r2 = r1 + self.crop_win
            c1 = random.randint(0, max_c)
            c2 = c1 + self.crop_win
            
            img_cropped = img[:,r1:r2,c1:c2]
            lbl_cropped = lbl[r1:r2,c1:c2]                       
        
            if (self.data_aug == 0):
                img_aug = img_cropped
                lbl_aug = lbl_cropped    
            else:
                aug_type = random.randrange(6)
                if aug_type == 0:
                    img_aug = img_cropped
                    lbl_aug = lbl_cropped              
                elif aug_type == 1:
                    img_aug = np.flip(img_cropped,-1)
                    lbl_aug = np.flip(lbl_cropped,-1)
                elif aug_type == 2:
                    img_aug = np.flip(img_cropped,-2)
                    lbl_aug = np.flip(lbl_cropped,-2)
                elif aug_type == 3:
                    img_aug = np.rot90(img_cropped,1,(-1,-2))
                    lbl_aug = np.rot90(lbl_cropped,1,(-1,-2))
                elif aug_type == 4:
                    img_aug = np.rot90(img_cropped,2,(-1,-2))
                    lbl_aug = np.rot90(lbl_cropped,2,(-1,-2))
                elif aug_type == 5:
                    img_aug = np.rot90(img_cropped,3,(-1,-2))
                    lbl_aug = np.rot90(lbl_cropped,3,(-1,-2))             
        
        if (self.pad != None):
            img_aug = np.pad(img_aug,((0,0), (self.pad[0],self.pad[1]),(self.pad[2],self.pad[3])),mode='symmetric')
        else:
            _, h, w = img_aug.shape   
            H = (math.ceil(h/16)+1) * 16
            W = (math.ceil(w/16)+1) * 16
            pad_H_0 = (H - h)//2
            pad_H_1 = H - h - pad_H_0
            pad_W_0 = (W - w)//2
            pad_W_1 = W - w - pad_W_0
            img_aug = np.pad(img_aug,((0,0), (pad_H_0,pad_H_1),(pad_W_0,pad_W_1)),mode='symmetric')
            
        #print(type(img_aug), type(lbl_aug))
        return tuple([img_aug.copy(), lbl_aug.copy()]) 

    def __len__(self):
        return len(self.dataset_list)