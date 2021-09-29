'''
Author:
Tianqi Guo 
Yin Wang
EISL-A @ Purdue University - School of Electrical and Computer Engineering
Do not use for commercial purposes. All rights reserved.
Contact:
guo246@purdue.edu
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_GROUP = 8

class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPP, self).__init__()   
        ASPP_planes = inplanes//2
        self.aspp_paths = nn.ModuleList()
        kernel_sizes = [1, 3, 3, 3]
        dilations = [1, 6, 12, 18]
        paddings = [0] + dilations[1:]
        for i in range(4):
            self.aspp_paths.append(                
                nn.Sequential(
                    nn.Conv2d(inplanes, ASPP_planes, kernel_size=kernel_sizes[i], 
                              stride=1, padding=paddings[i], dilation=dilations[i], bias=False),
                    nn.GroupNorm(N_GROUP, ASPP_planes),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.aspp_paths.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(inplanes, ASPP_planes, 1, stride=1, bias=False),
                nn.GroupNorm(N_GROUP, ASPP_planes),
                nn.ReLU(inplace=True)
            )  
        )
        
        self.exit = nn.Sequential(
            nn.Conv2d(ASPP_planes*5, outplanes, 1, bias=False),
            nn.GroupNorm(N_GROUP, outplanes)
        )
        if inplanes != outplanes:
            self.skip = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),  
                nn.GroupNorm(N_GROUP, outplanes)
            )     
        else:
            self.skip = nn.Sequential()            
        
    def forward(self, x): 
        skip = self.skip(x) 
        xs = [path(x) for path in self.aspp_paths]
        xs[-1] = F.interpolate(xs[-1], size=x.shape[-2:], mode='bilinear', align_corners=True)             
        xs = torch.cat(xs, dim=1)        
        out = self.exit(xs)  
        out = out + skip
        return F.relu(out, inplace=True) 
    
class UpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            nn.GroupNorm(N_GROUP, out_size//2),
            nn.ReLU(inplace=True)
        )
        self.bridge = nn.Sequential(
            nn.Conv2d(out_size, out_size//2, kernel_size=1, bias=False),  
            nn.GroupNorm(N_GROUP, out_size//2),
            nn.ReLU(inplace=True)
        )
 
        nf = out_size
        gc = out_size//4
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(nf, gc, 3, 1, 1, bias=False),
            nn.GroupNorm(N_GROUP, gc),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=False),
            nn.GroupNorm(N_GROUP, gc),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=False),
            nn.GroupNorm(N_GROUP, gc),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=False),
            nn.GroupNorm(N_GROUP, gc),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=False),
            nn.GroupNorm(N_GROUP, nf),            
        )       
        
    def forward(self, x, bridge):        
        up = self.up(x)  
        bridge = self.bridge(bridge)
        x0 = torch.cat([up, bridge], 1)        
        
        x1 = self.conv1(x0)
        x2 = self.conv2(torch.cat((x0, x1), 1))
        x3 = self.conv3(torch.cat((x0, x1, x2), 1))
        x4 = self.conv4(torch.cat((x0, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x0, x1, x2, x3, x4), 1))
        x6 = x5 * 0.2 + x0
        return F.relu(x6, inplace=True)         
    
class DownBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(DownBlock, self).__init__() 
        
        nf = inplanes
        gc = inplanes//4
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(nf, gc, 3, 1, 1, bias=False),
            nn.GroupNorm(N_GROUP, gc),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=False),
            nn.GroupNorm(N_GROUP, gc),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=False),
            nn.GroupNorm(N_GROUP, gc),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=False),
            nn.GroupNorm(N_GROUP, gc),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=False),
            nn.GroupNorm(N_GROUP, nf)
        )          
        
        self.down = nn.Sequential( 
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(N_GROUP, planes),
            nn.ReLU(inplace=True)
        )                    

    def forward(self, x):
                
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = x5 * 0.2 + x
        
        return self.down(x6)

class XBNet(nn.Module):    
    def __init__(self):
        super(XBNet, self).__init__()        
        chns = 32        
        self.entry_flow = nn.Sequential(   
            nn.Conv2d(1, chns, 7, stride=1, padding=3, bias=False),
            nn.GroupNorm(N_GROUP, chns),
            nn.ReLU(inplace=True),
            nn.Conv2d(chns, chns, 3, stride=1, padding=1, bias=False),
            nn.GroupNorm(N_GROUP, chns),
            nn.ReLU(inplace=True)
        )         
        self.down_path = nn.ModuleList()
        for _ in range(4):
            self.down_path.append(DownBlock(chns, chns*2))  
            chns *= 2
        self.ASPP = ASPP(chns, chns)         
        self.up_path = nn.ModuleList()
        for _ in range(4):
            self.up_path.append(UpBlock(chns, chns//2))
            chns //= 2
        self.last = nn.Conv2d(chns, 3, kernel_size=1)
        
    def forward(self, x):        
        feature_maps = []        
        x = self.entry_flow(x)
        feature_maps += [x]         
        for i, down in enumerate(self.down_path):
            x = down(x) 
            if i < 3:                
                feature_maps += [x]                       
        x = self.ASPP(x)           
        for up, feature_map in zip(self.up_path, feature_maps[::-1]):   
            x = up(x, feature_map)        
        return self.last(x)  