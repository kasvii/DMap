import os
import torch
import numpy as np
from tqdm import tqdm
import cv2
import sys

from lib.models import build_body_model
from .losses_gar import SMPLifyLoss

class TemporalSMPLify_GAR():
    
    def __init__(self, 
                 smpl=None,
                 lr=1e-2,
                 num_iters=5,
                 num_steps=50,
                 img_w=None,
                 img_h=None,
                 device=None
                 ):
        
        self.smpl = smpl
        self.lr = lr
        self.num_iters = num_iters
        self.num_steps = num_steps
        self.img_w = img_w
        self.img_h = img_h
        self.device = device
        
    def fit(self, init_pred, keypoints, mask_full=None, gar_top=None, gar_bottom=None, renderer_textured_soft=None, transform=None, trans_cams=None, faces=None, bbox=None, scale=1, **kwargs):
        
        def to_params(param):
            return torch.from_numpy(param).float().to(self.device).requires_grad_(True)
        
        pose = init_pred['pose']
        betas = torch.from_numpy(init_pred['betas']).float().unsqueeze(0).to(self.device).requires_grad_(True)
        # cam = init_pred['cam']
        keypoints = torch.from_numpy(keypoints).float().unsqueeze(0).to(self.device)
        
        BN = pose.shape[0]
        lr = self.lr
        
        params = [to_params(pose), betas]
        
        optimizer = torch.optim.Adam(
            params, 
            lr=lr * BN
        )
        
        loss_fn = SMPLifyLoss(init_pose=pose, device=self.device, **kwargs)
        
        closure = loss_fn.create_closure(optimizer,
                       self.smpl, 
                       params,
                       bbox,
                       keypoints,
                       mask_full,
                       renderer_textured_soft,
                       transform,
                       trans_cams, 
                       faces,
                       scale,
                       gar_top, 
                       gar_bottom,
                       **kwargs)
        
        for j in (j_bar := tqdm(range(self.num_steps), leave=False)):
            print(f"Iter: {j}")
            optimizer.zero_grad()
            loss = optimizer.step(closure)
            msg = f'Loss: {loss.item():.1f}'
            j_bar.set_postfix_str(msg)
        
        init_pred['pose'] = params[0].detach()
        init_pred['betas'] = params[1].detach()
        
        return init_pred