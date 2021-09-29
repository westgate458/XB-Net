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
import sys
import math
import tracemalloc
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader

from defs import *
from XBNet import XBNet

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
        
tracemalloc.start()
show_usage()

if_resume = 0 #int(sys.argv[2])
epochs = 26000 #int(sys.argv[3])
test = 'train_demo'

data_dir = './datasets/pre/GT+ST/'

result_dir = os.path.join(data_dir,test)
os.makedirs(result_dir, exist_ok=True)
print('Result DIR = ', result_dir)

print(torch.cuda.get_device_name()) 
device = 'cuda'
print(sys.argv)

cases = ['DIC-C2DH-HeLa','Fluo-N3DH-CHO']


N_cases = len(cases)
print('Training %d cases.'%N_cases)

data_aug = 1

loaders_training = []
loaders_validation = []

margin = 16

model = XBNet()
model.train()

for i in range(N_cases):
    
    case = cases[i]
    crop_win = 256 
    batchSize = 8   
    
    h = w = crop_win
    H = (math.ceil(h/margin)+1) * margin
    W = (math.ceil(w/margin)+1) * margin
    pad_H_0 = (H - h)//2
    pad_H_1 = H - h - pad_H_0
    pad_W_0 = (W - w)//2
    pad_W_1 = W - w - pad_W_0
    pad_train = (pad_H_0, pad_H_1, pad_W_0, pad_W_1)
    pad_eval = None
    
    source_dir = os.path.join(data_dir,case)
    
    training_data_path = os.path.join(source_dir,'01')
    validation_data_path = os.path.join(source_dir,'02')
    
    RS_dir = os.path.join(result_dir,case)
    print('case result dir:', RS_dir)
    
    dataset_training = LoadDataset(training_data_path, crop_win, data_aug, pad_train)
    dataset_validation = LoadDataset(validation_data_path, 0, 0, pad_eval)
    
    loader_training = InfiniteDataLoader(dataset_training, batch_size=batchSize, shuffle=True, drop_last=False, pin_memory=True)
    loaders_training.append(loader_training)
    
    loader_validation = data_utils.DataLoader(dataset_validation, batch_size=1, shuffle=False, pin_memory=True)
    loaders_validation.append(loader_validation)    
        
    X1, y1 = dataset_training[0]
    print('Training set: # samples:', len(dataset_training), ', Image & target sizes:', X1.shape, y1.shape, ', # batches:',len(loader_training), ', batchsize:', batchSize)
    X3, y3 = dataset_validation[0]
    print('Validation set:  # samples:', len(dataset_validation),', Image & target sizes:', X3.shape, y3.shape)
    print()
    
LR = 1e-4
weights_arg = [1,10,5]

period_eval = 2600
period_print = 260

settings = (LR, )+tuple(weights_arg)
print('Training settings: LR = %f, weights = (%d,%d,%d)'%settings)

weights = torch.tensor(weights_arg).to(device).float()


if if_resume == 1:
        
    loss_history_CE,loss_history_JR,LR_history,evl_epochs,seg_history_validation,seg_score_validation_max,seg_score_max_epoch, time_lapsed = \
        np.load(os.path.join(result_dir,'loss_history.npy'), allow_pickle = True)   

    if isinstance(evl_epochs,np.ndarray):
        evl_epochs  = evl_epochs.tolist()
        seg_history_validation = seg_history_validation.tolist()
        loss_history_CE = loss_history_CE.tolist()
        loss_history_JR = loss_history_JR.tolist()
        LR_history = LR_history.tolist()

    starting_epoch = int(evl_epochs[-1])

    loss_history_CE = loss_history_CE[:starting_epoch]
    loss_history_JR = loss_history_JR[:starting_epoch]
    LR_history = LR_history[:starting_epoch]

    checkpoint = torch.load(os.path.join(result_dir,'checkpoint.pth'))
    #checkpoint = torch.load(os.path.join(result_dir,'checkpoint.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    
    optim = torch.optim.AdamW(model.parameters(),lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=True)
    optim.load_state_dict(checkpoint['optim'])
    
    del checkpoint
    torch.cuda.empty_cache()
    
else:
    
    time_lapsed = datetime.now()-datetime.now()
    
    starting_epoch = 0
    loss_history_CE = []
    loss_history_JR = []
    evl_epochs = [] 
    seg_history_validation = []
    LR_history = []
    seg_score_validation_max = 0
    seg_score_max_epoch = -1
    
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=True)

print('Start training from Epoch %d towards Epoch %d.'%(starting_epoch, epochs))
print('Best seg score so far: %0.4f at Epoch %d.'%(seg_score_validation_max, seg_score_max_epoch))

show_usage()    

np.save(os.path.join(result_dir,'loss_history.npy'), (loss_history_CE,loss_history_JR,LR_history,evl_epochs,seg_history_validation,seg_score_validation_max,seg_score_max_epoch, time_lapsed))

batch_i = starting_epoch

time_0 = datetime.now()

time_lapsed_i = datetime.now()-datetime.now()

while batch_i < epochs:
    
    optim.zero_grad()
    
    for loader_training in loaders_training:   
        
        batch_i += 1
        
        X, y = next(loader_training)
        prediction = model(X.to(device)) 

        _, h, w = y.shape
        _, _, H, W = prediction.shape
        pad_H_0 = (H - h)//2
        pad_H_1 = H - h - pad_H_0
        pad_W_0 = (W - w)//2
        pad_W_1 = W - w - pad_W_0
        prediction = prediction[:,:,pad_H_0:-pad_H_1,pad_W_0:-pad_W_1]

        L_CE = F.cross_entropy(prediction, y.long().to(device), weight=weights)
        loss = L_CE        
        loss_history_CE += [L_CE.data.cpu().numpy()]
        loss_history_JR += [0] 
        
        loss.backward()
        
        del X, y, prediction, L_CE, loss
        torch.cuda.empty_cache()
        LR_now = optim.param_groups[0]['lr']
        LR_history += [LR_now]  
        
    optim.step()
    
    if (batch_i%period_print == 0) or (batch_i==N_cases):
        now = datetime.now()
        current_time = now.strftime("[%m/%d %H:%M:%S]")
        
        past_time = now - time_0
        time_lapsed = time_lapsed + past_time
        time_lapsed_i = time_lapsed_i + past_time
        
        time_0 = now
        sec_per_itr = time_lapsed_i/float(batch_i-starting_epoch)
        
        print(current_time,'Itr: %0.5d,'%batch_i, 'Time: (', time_lapsed, ', %0.2f)'%sec_per_itr.total_seconds(), end=", ")
        
        current, peak = torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
        print("GPU: (%07.2f, %07.2f) MB, "%(current/10**6,peak/10**6), end="")
        print('Loss: (%0.4f, %0.4f), LR: %.2E, '%(loss_history_CE[-1],loss_history_JR[-1],LR_history[-1]), end="")
   
    
    if (batch_i%period_eval == 0) or (batch_i==N_cases):
        
        print('Valid SEG: ( ', end="")
            
        evl_epochs.append(batch_i)     
        
        seg_score_validation_epoch = []
        
        for case, loader_validation in zip(cases,loaders_validation):          
            _, seg_score_validation, _ = eval_performance_test(model, loader_validation, device, os.path.join(data_dir,case), '02')   
            print('%0.4f '%seg_score_validation, end="")
            seg_score_validation_epoch += [seg_score_validation]
                   
        seg_history_validation += [seg_score_validation_epoch]
        seg_score_validation = np.mean(seg_score_validation_epoch)
        print('). Mean: %0.4f, '%seg_score_validation, end="")
        
        if seg_score_validation >= seg_score_validation_max:
            seg_score_validation_max = seg_score_validation
            seg_score_max_epoch = batch_i
            torch.save(model.state_dict(), os.path.join(result_dir,'trained_model.pth'))
        print('Best: %0.4f @ Itr %d'%(seg_score_validation_max, seg_score_max_epoch), end=".")

        checkpoint = { 
            'epoch': batch_i,
            'model': model.state_dict(),
            'optim': optim.state_dict()
        }
        torch.save(checkpoint, os.path.join(result_dir,'checkpoint_temp.pth'))
        shutil.move(os.path.join(result_dir,'checkpoint_temp.pth'), os.path.join(result_dir,'checkpoint.pth'))
        np.save(os.path.join(result_dir,'loss_history.npy'), (loss_history_CE,loss_history_JR,LR_history,evl_epochs,seg_history_validation,seg_score_validation_max,seg_score_max_epoch, time_lapsed))
        
        time_0 = datetime.now()
        
    if (batch_i%period_print == 0) or (batch_i==N_cases):
        print('')
        
print('')

print('Total training time:', time_lapsed)

np.save(os.path.join(result_dir,'loss_history.npy'), (loss_history_CE,loss_history_JR,LR_history,evl_epochs,seg_history_validation,seg_score_validation_max,seg_score_max_epoch, time_lapsed))
show_usage()
tracemalloc.stop()  