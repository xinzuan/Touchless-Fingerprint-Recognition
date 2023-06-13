from state import SRType
from cv2 import dnn_superres

import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import time

class SuperResolution(object):
    def __init__(self,type=SRType.FSRCNN.value[0],path=''):
        
        self.sr = dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(path)
        self.type = type
        if self.type == SRType.FSRCNN.value[0]:

            self.sr.setModel(SRType.FSRCNN.value[1], 4)
        elif self.type == SRType.EDSR.value[0]:
            self.sr.setModel(SRType.EDSR.value[1], 4)
        super().__init__()
    
    def upsample_image(self,img):
        if self.type == SRType.EDSR.value[0]:
        
            # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img = self.sr.upsample(img)
            # img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            return img

        return self.sr.upsample(img)

class SRESGRAN(object):
    def __init__(self,path):
        self.model_path = path
        self.device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
        # device = torch.device('cuda')

        

        self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(self.model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        
        super().__init__()
    
    def upsample_image(self,img):
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        with torch.no_grad():
            output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # output = cv2.cvtColor(output,cv2.COLOR_RGB2GRAY)
        return output
