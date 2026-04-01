""" 
SSMamba: State Space Duality Model for Remote Physiological Measurement
"""
import sys
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
import math
from einops import rearrange
from mamba_ssm.modules.SA_Mamba2_simple import Mamba2Simple as SA_Mamba
from mamba_ssm.modules.CA_Mamba2_simple import Mamba2Simple as CA_Mamba
# from tensorboardX import SummaryWriter
import numpy as np

# from mamba_ssm import Mamba
from .fusion_module import LCA, TemporalLCA


# ====================attention mask========================
# class TemporalAttention(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64, num_heads=1):
#         super(TemporalAttention, self).__init__()
#         self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=0.1)
#         self.norm = nn.LayerNorm(input_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim)
#         )
#         self.norm2 = nn.LayerNorm(input_dim)

#     def forward(self, x):
#         # x: [D, N, input_dim]
#         attn_output, _ = self.attention(x, x, x)
#         x = self.norm(x + attn_output)
#         ffn_output = self.ffn(x)
#         x = self.norm2(x + ffn_output)
#         return x
    # =====================attention mask========================

class CAFusion_Stem(nn.Module):
    def __init__(self,apha=0.5,belta=0.5,dim=24, use_3d_diff=True):

        super(CAFusion_Stem, self).__init__()
        self.use_3d_diff = use_3d_diff
        self.stem11 = nn.Sequential(nn.Conv2d(3, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )
        
        if use_3d_diff:
            # 3D motion stem
            # Input: [N, 12, D, H, W]
            self.stem12_3d = nn.Sequential(
                nn.Conv3d(12, dim // 2, kernel_size=(5, 3, 3),
                          stride=(1, 2, 2), padding=(2, 1, 1)),
                nn.BatchNorm3d(dim // 2),
                nn.ReLU(inplace=True),
                # Optional: spatial pooling only
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), ceil_mode=False)
            )
        else:
            self.stem12 = nn.Sequential(nn.Conv2d(12, dim//2, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(dim//2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                )

        self.stem21 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem22 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        # self.alpha = apha
        # self.beta = belta
        self.alpha = nn.Parameter(torch.tensor(float(apha)))
        self.beta = nn.Parameter(torch.tensor(float(belta)))
# =======Learnable temporal convolution for difference computation========
        # self.temporal_conv = nn.Conv3d(
        #     in_channels=3, 
        #     out_channels=12, 
        #     kernel_size=(3, 1, 1),  # Temporal kernel size of 3, spatial 1x1
        #     stride=(1, 1, 1),
        #     padding=(1, 0, 0))  # Padding to maintain temporal dimension

        # Temporal attention mechanism
        # self.temporal_attention = TemporalAttention(input_dim=3*128*128)
        # self.proj = nn.Linear(3*128*128, 12*128*128)  # Project to 12 channels

#  ===================Mamba STEM========================
# Mamba blocks for processing original and difference features
        # self.stem11 = nn.Sequential(
        #     Mamba(d_model=3, d_state=16, d_conv=4, expand=2),  # Process [N*D, H*W, 3]
        #     nn.Linear(3, dim//2),  # Adjust channels
        #     nn.MaxPool2d(kernel_size=4, stride=4)  # Downsample H, W by 4
        # )
        
        # self.stem12 = nn.Sequential(
        #     Mamba(d_model=12, d_state=16, d_conv=4, expand=2),  # Process [N*D, H*W, 12]
        #     nn.Linear(12, dim//2),
        #     nn.MaxPool2d(kernel_size=4, stride=4)
        # )

        # self.stem21 = nn.Sequential(
        #     Mamba(d_model=dim//2, d_state=16, d_conv=4, expand=2),
        #     nn.Linear(dim//2, dim),
        #     nn.MaxPool2d(kernel_size=2, stride=2)  # Further downsample H, W by 2
        # )

        # self.stem22 = nn.Sequential(
        #     Mamba(d_model=dim//2, d_state=16, d_conv=4, expand=2),
        #     nn.Linear(dim//2, dim),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.dim = dim
        # self.early_pool = nn.MaxPool2d(kernel_size=4, stride=4)  # Downsample from 128 to 32
    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/8,W/8]
        """
        N, D, C, H, W = x.shape 
        x1 = torch.cat([x[:,:1,:,:,:],x[:,:1,:,:,:],x[:,:D-2,:,:,:]],1)
        x2 = torch.cat([x[:,:1,:,:,:],x[:,:D-1,:,:,:]],1)
        x3 = x
        x4 = torch.cat([x[:,1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x5 = torch.cat([x[:,2:,:,:,:],x[:,D-1:,:,:,:],x[:,D-1:,:,:,:]],1)

        # x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2) .view(N * D, 12, H, W))
        diff = torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2)  # [N,D,4C,H,W] -> [N,D,12,H,W]

        if self.use_3d_diff:
            # [N,D,12,H,W] -> [N,12,D,H,W]
            diff_3d = diff.permute(0, 2, 1, 3, 4).contiguous()
            # Conv3D branch
            x_diff_3d = self.stem12_3d(diff_3d)  # [N, dim//2, D, H', W']
            N_, C_, D_, H_, W_ = x_diff_3d.shape
            # flatten time for 2D fusion
            x_diff = x_diff_3d.permute(0, 2, 1, 3, 4).reshape(N * D_, C_, H_, W_)
        else:
            diff_2d = diff.view(N * D, 12, H, W)
            x_diff = self.stem12(diff_2d)   # [N*D, dim//2, H', W']

        x3 = x3.contiguous().view(N * D, C, H, W) 
        x = self.stem11(x3) 

        #fusion layer1
        x_path1 = self.alpha*x + self.beta*x_diff 
        x_path1 = self.stem21(x_path1) 
        #fusion layer2
        x_path2 = self.stem22(x_diff) 

        x = self.alpha*x_path1 + self.beta*x_path2

        return x
    # ==================temporal convolution fwd =========================
    # def forward(self, x):
    #     """Definition of Fusion_Stem.
    #     Args:
    #       x [N,D,C,H,W]
    #     Returns:
    #       fusion_x [N*D,C,H/8,W/8]
    #     """
    #     N, D, C, H, W = x.shape
    #     # print(f"Input shape: {x.shape}")  # [8, 160, 3, 128, 128]

    #     # Hand-crafted temporal differences
    #     x1 = torch.cat([x[:, :1, :, :, :], x[:, :1, :, :, :], x[:, :D-2, :, :, :]], dim=1)
    #     x2 = torch.cat([x[:, :1, :, :, :], x[:, :D-1, :, :, :]], dim=1)
    #     x3 = x
    #     x4 = torch.cat([x[:, 1:, :, :, :], x[:, D-1:, :, :, :]], dim=1)
    #     x5 = torch.cat([x[:, 2:, :, :, :], x[:, D-1:, :, :, :], x[:, D-1:, :, :, :]], dim=1)

    #     x_diff = torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], dim=2)  # [N, D, 12, H, W]
    #     x_diff = x_diff.contiguous().view(N * D, 12, H, W)
    #     # print(f"Before reshape for Mamba (stem12): {x_diff.shape}")  # [1280, 12, 128, 128]
    #     x_diff = x_diff.permute(0, 2, 3, 1).contiguous().view(N * D, H * W, 12)
    #     # print(f"Before Mamba (stem12): {x_diff.shape}")  # [1280, 16384, 12]
    #     x_diff = self.stem12[0](x_diff)  # Mamba: [1280, 16384, 12]
    #     # print(f"After Mamba (stem12): {x_diff.shape}")  # [1280, 16384, 12]
    #     x_diff = self.stem12[1](x_diff)  # Linear: [1280, 16384, self.dim // 2]
    #     # print(f"After stem12 linear: {x_diff.shape}")  # [1280, 16384, 8]
    #     x_diff = x_diff.view(N * D, H, W, self.dim // 2).permute(0, 3, 1, 2)  # [1280, 8, 128, 128]
    #     # print(f"After reshape for pooling (stem12): {x_diff.shape}")  # [1280, 8, 128, 128]
    #     x_diff = self.stem12[2](x_diff)  # MaxPool: [1280, 8, 32, 32]
    #     # print(f"After stem12: {x_diff.shape}")  # [1280, 8, 32, 32]

    #     # Process original frames with Mamba
    #     x3 = x3.contiguous().view(N * D, C, H, W)
    #     # print(f"x3 shape: {x3.shape}")  # [1280, 3, 128, 128]
    #     x = x3.permute(0, 2, 3, 1).contiguous().view(N * D, H * W, C)
    #     # print(f"Before Mamba (stem11): {x.shape}")  # [1280, 16384, 3]
    #     x = self.stem11[0](x)
    #     # print(f"After Mamba (stem11): {x.shape}")  # [1280, 16384, 3]
    #     x = self.stem11[1](x)  # Linear: [1280, 16384, self.dim // 2]
    #     # print(f"After stem11 linear: {x.shape}")  # [1280, 16384, 8]
    #     x = x.view(N * D, H, W, self.dim // 2).permute(0, 3, 1, 2)  # [1280, 8, 128, 128]
    #     # print(f"After reshape for pooling (stem11): {x.shape}")  # [1280, 8, 128, 128]
    #     x = self.stem11[2](x)  # MaxPool: [1280, 8, 32, 32]
    #     # print(f"After stem11: {x.shape}")  # [1280, 8, 32, 32]

    #     # Fusion layer 1
    #     x_path1 = self.alpha * x + self.beta * x_diff
    #     x_path1 = x_path1.reshape(N * D, 32 * 32, self.dim // 2)  # [1280, 1024, 8]
    #     # print(f"Before Mamba (stem21): {x_path1.shape}")  # [1280, 1024, 8]
    #     x_path1 = self.stem21[0](x_path1)
    #     # print(f"After Mamba (stem21): {x_path1.shape}")  # [1280, 1024, 8]
    #     x_path1 = self.stem21[1](x_path1)  # Linear: [1280, 1024, self.dim]
    #     x_path1 = x_path1.view(N * D, self.dim, 32, 32)  # [1280, 16, 32, 32]
    #     # print(f"After stem21 linear: {x_path1.shape}")  # [1280, 16, 32, 32]
    #     x_path1 = self.stem21[2](x_path1)  # MaxPool: [1280, 16, 16, 16]
    #     # print(f"After stem21: {x_path1.shape}")  # [1280, 16, 16, 16]

    #     # Fusion layer 2
    #     x_path2 = x_diff.reshape(N * D, 32 * 32, self.dim // 2)  # [1280, 1024, 8]
    #     # print(f"Before Mamba (stem22): {x_path2.shape}")  # [1280, 1024, 8]
    #     x_path2 = self.stem22[0](x_path2)
    #     # print(f"After Mamba (stem22): {x_path2.shape}")  # [1280, 1024, 8]
    #     x_path2 = self.stem22[1](x_path2)  # Linear: [1280, 1024, self.dim]
    #     x_path2 = x_path2.view(N * D, self.dim, 32, 32)  # [1280, 16, 32, 32]
    #     # print(f"After stem22 linear: {x_path2.shape}")  # [1280, 16, 32, 32]
    #     x_path2 = self.stem22[2](x_path2)  # MaxPool: [1280, 16, 16, 16]
    #     # print(f"After stem22: {x_path2.shape}")  # [1280, 16, 16, 16]

    #     x = self.alpha * x_path1 + self.beta * x_path2
    #     # print(f"Output shape: {x.shape}")  # [1280, 16, 16, 16]
    #     print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    #     return x
 

class SAFusion_Stem(nn.Module):
    def __init__(self,apha=0.5,belta=0.5,dim=24, use_3d_diff=False):
        super(SAFusion_Stem, self).__init__()

        self.use_3d_diff = use_3d_diff
        self.stem11 = nn.Sequential(nn.Conv2d(3, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )
        if use_3d_diff:
            # 3D motion stem
            # Input: [N, 12, D, H, W]
            self.stem12_3d = nn.Sequential(
                nn.Conv3d(12, dim // 2, kernel_size=(5, 3, 3),
                          stride=(1, 2, 2), padding=(2, 1, 1)),
                nn.BatchNorm3d(dim // 2),
                nn.ReLU(inplace=True),
                # Optional: spatial pooling only
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), ceil_mode=False)
            )   
        else:
            self.stem12 = nn.Sequential(nn.Conv2d(12, dim//2, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(dim//2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                )

        self.stem21 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem22 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        # self.alpha = apha
        # self.beta = belta
        self.alpha = nn.Parameter(torch.tensor(float(apha)))
        self.beta = nn.Parameter(torch.tensor(float(belta)))
# Learnable temporal convolution for difference computation
        # self.temporal_conv = nn.Conv3d(
        #     in_channels=3, 
        #     out_channels=12, 
        #     kernel_size=(3, 1, 1),  # Temporal kernel size of 3, spatial 1x1
        #     stride=(1, 1, 1),
        #     padding=(1, 0, 0))  # Padding to maintain temporal dimension
        

# Temporal attention mechanism===========================================================
        # self.temporal_attention = TemporalAttention(input_dim=3*128*128)
        # self.proj = nn.Linear(3*128*128, 12*128*128)  # Project to 12 channels
#  ====================Mamba STEM========================
# Mamba blocks for processing original and difference features
        # self.stem11 = nn.Sequential(
        #     Mamba(d_model=3, d_state=16, d_conv=4, expand=2),  # Process [N*D, H*W, 3]
        #     nn.Linear(3, dim//2),  # Adjust channels
        #     nn.MaxPool2d(kernel_size=4, stride=4)  # Downsample H, W by 4
        # )
        
        # self.stem12 = nn.Sequential(
        #     Mamba(d_model=12, d_state=16, d_conv=4, expand=2),  # Process [N*D, H*W, 12]
        #     nn.Linear(12, dim//2),
        #     nn.MaxPool2d(kernel_size=4, stride=4)
        # )

        # self.stem21 = nn.Sequential(
        #     Mamba(d_model=dim//2, d_state=16, d_conv=4, expand=2),
        #     nn.Linear(dim//2, dim),
        #     nn.MaxPool2d(kernel_size=2, stride=2)  # Further downsample H, W by 2
        # )

        # self.stem22 = nn.Sequential(
        #     Mamba(d_model=dim//2, d_state=16, d_conv=4, expand=2),
        #     nn.Linear(dim//2, dim),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.dim = dim
        # self.early_pool = nn.MaxPool2d(kernel_size=4, stride=4)  # Downsample from 128 to 32
    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/8,W/8]
        """
        # x = x[:, :, :3, :, :]  # Ensure only first 3 channels are used
        # print(f"Input shape: {x.shape}")  # [8, 160, 3, 128, 128]
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:,:1,:,:,:],x[:,:1,:,:,:],x[:,:D-2,:,:,:]],1)
        x2 = torch.cat([x[:,:1,:,:,:],x[:,:D-1,:,:,:]],1)
        x3 = x
        x4 = torch.cat([x[:,1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x5 = torch.cat([x[:,2:,:,:,:],x[:,D-1:,:,:,:],x[:,D-1:,:,:,:]],1)
        # print(f"x5 shape: {x5.shape}")  # [8, 160, 3, 128, 128]

        # x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2).view(N * D, 12, H, W))
        # import pdb; pdb.set_trace()
        # x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2).view(N * D, 12, H, W))
        diff = torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2)  # [N,D,4C,H,W] -> [N,D,12,H,W]
        # print(f"Diff shape: {diff.shape}")  # [8, 160, 12, 128, 128]

        if self.use_3d_diff:
            # [N,D,12,H,W] -> [N,12,D,H,W]
            diff_3d = diff.permute(0, 2, 1, 3, 4).contiguous()
            # Conv3D branch
            x_diff_3d = self.stem12_3d(diff_3d)  # [N, dim//2, D, H', W']
            N_, C_, D_, H_, W_ = x_diff_3d.shape
            # flatten time for 2D fusion
            x_diff = x_diff_3d.permute(0, 2, 1, 3, 4).reshape(N * D_, C_, H_, W_)
        else:
            diff_2d = diff.view(N * D, 12, H, W)
            # print(f"Diff 2D shape: {diff_2d.shape}")  # [1280, 12, 128, 128]
            x_diff = self.stem12(diff_2d)   # [N*D, dim//2, H', W']
            # print(f"x_diff shape after stem12: {x_diff.shape}")  # [1280, 8, 32, 32]

        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        #fusion layer1
        x_path1 = self.alpha*x + self.beta*x_diff
        x_path1 = self.stem21(x_path1)
        #fusion layer2
        x_path2 = self.stem22(x_diff)

        x = self.alpha*x_path1 + self.beta*x_path2

        return x
    # ====================== temporal convolution fwd =========================
    # def forward(self, x):
    #     """Definition of Fusion_Stem.
    #     Args:
    #       x [N,D,C,H,W]
    #     Returns:
    #       fusion_x [N*D,C,H/8,W/8]
    #     """
    #     N, D, C, H, W = x.shape
    #     # print(f"Input shape: {x.shape}")  # [8, 160, 3, 128, 128]

    #     # Hand-crafted temporal differences
    #     x1 = torch.cat([x[:, :1, :, :, :], x[:, :1, :, :, :], x[:, :D-2, :, :, :]], dim=1)
    #     x2 = torch.cat([x[:, :1, :, :, :], x[:, :D-1, :, :, :]], dim=1)
    #     x3 = x
    #     x4 = torch.cat([x[:, 1:, :, :, :], x[:, D-1:, :, :, :]], dim=1)
    #     x5 = torch.cat([x[:, 2:, :, :, :], x[:, D-1:, :, :, :], x[:, D-1:, :, :, :]], dim=1)

    #     x_diff = torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], dim=2)  # [N, D, 12, H, W]
    #     x_diff = x_diff.contiguous().view(N * D, 12, H, W)
    #     # print(f"Before reshape for Mamba (stem12): {x_diff.shape}")  # [1280, 12, 128, 128]
    #     x_diff = x_diff.permute(0, 2, 3, 1).contiguous().view(N * D, H * W, 12)
    #     # print(f"Before Mamba (stem12): {x_diff.shape}")  # [1280, 16384, 12]
    #     x_diff = self.stem12[0](x_diff)  # Mamba: [1280, 16384, 12]
    #     # print(f"After Mamba (stem12): {x_diff.shape}")  # [1280, 16384, 12]
    #     x_diff = self.stem12[1](x_diff)  # Linear: [1280, 16384, self.dim // 2]
    #     # print(f"After stem12 linear: {x_diff.shape}")  # [1280, 16384, 8]
    #     x_diff = x_diff.view(N * D, H, W, self.dim // 2).permute(0, 3, 1, 2)  # [1280, 8, 128, 128]
    #     # print(f"After reshape for pooling (stem12): {x_diff.shape}")  # [1280, 8, 128, 128]
    #     x_diff = self.stem12[2](x_diff)  # MaxPool: [1280, 8, 32, 32]
    #     # print(f"After stem12: {x_diff.shape}")  # [1280, 8, 32, 32]

    #     # Process original frames with Mamba
    #     x3 = x3.contiguous().view(N * D, C, H, W)
    #     # print(f"x3 shape: {x3.shape}")  # [1280, 3, 128, 128]
    #     x = x3.permute(0, 2, 3, 1).contiguous().view(N * D, H * W, C)
    #     # print(f"Before Mamba (stem11): {x.shape}")  # [1280, 16384, 3]
    #     x = self.stem11[0](x)
    #     # print(f"After Mamba (stem11): {x.shape}")  # [1280, 16384, 3]
    #     x = self.stem11[1](x)  # Linear: [1280, 16384, self.dim // 2]
    #     # print(f"After stem11 linear: {x.shape}")  # [1280, 16384, 8]
    #     x = x.view(N * D, H, W, self.dim // 2).permute(0, 3, 1, 2)  # [1280, 8, 128, 128]
    #     # print(f"After reshape for pooling (stem11): {x.shape}")  # [1280, 8, 128, 128]
    #     x = self.stem11[2](x)  # MaxPool: [1280, 8, 32, 32]
    #     # print(f"After stem11: {x.shape}")  # [1280, 8, 32, 32]

    #     # Fusion layer 1
    #     x_path1 = self.alpha * x + self.beta * x_diff
    #     x_path1 = x_path1.reshape(N * D, 32 * 32, self.dim // 2)  # [1280, 1024, 8]
    #     # print(f"Before Mamba (stem21): {x_path1.shape}")  # [1280, 1024, 8]
    #     x_path1 = self.stem21[0](x_path1)
    #     # print(f"After Mamba (stem21): {x_path1.shape}")  # [1280, 1024, 8]
    #     x_path1 = self.stem21[1](x_path1)  # Linear: [1280, 1024, self.dim]
    #     x_path1 = x_path1.view(N * D, self.dim, 32, 32)  # [1280, 16, 32, 32]
    #     # print(f"After stem21 linear: {x_path1.shape}")  # [1280, 16, 32, 32]
    #     x_path1 = self.stem21[2](x_path1)  # MaxPool: [1280, 16, 16, 16]
    #     # print(f"After stem21: {x_path1.shape}")  # [1280, 16, 16, 16]

    #     # Fusion layer 2
    #     x_path2 = x_diff.reshape(N * D, 32 * 32, self.dim // 2)  # [1280, 1024, 8]
    #     # print(f"Before Mamba (stem22): {x_path2.shape}")  # [1280, 1024, 8]
    #     x_path2 = self.stem22[0](x_path2)
    #     # print(f"After Mamba (stem22): {x_path2.shape}")  # [1280, 1024, 8]
    #     x_path2 = self.stem22[1](x_path2)  # Linear: [1280, 1024, self.dim]
    #     x_path2 = x_path2.view(N * D, self.dim, 32, 32)  # [1280, 16, 32, 32]
    #     # print(f"After stem22 linear: {x_path2.shape}")  # [1280, 16, 32, 32]
    #     x_path2 = self.stem22[2](x_path2)  # MaxPool: [1280, 16, 16, 16]
    #     # print(f"After stem22: {x_path2.shape}")  # [1280, 16, 16, 16]

    #     x = self.alpha * x_path1 + self.beta * x_path2
    #     # print(f"Output shape: {x.shape}")  # [1280, 16, 16, 16]
    #     print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    #     return x
    # ===================temporal attention below========================
    #  print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    # def forward(self, x):
    #     """Definition of Fusion_Stem.
    #     Args:
    #       x [N,D,C,H,W]
    #     Returns:
    #       fusion_x [N*D,C,H/8,W/8]
    #     """
    #     N, D, C, H, W = x.shape  # e.g., [16, 160, 3, 128, 128]

    #     # Flatten spatial dimensions for attention: [N, D, C*H*W]
    #     x_temp = x.view(N, D, -1)  # [16, 160, 3*128*128]
    #     x_temp = x_temp.permute(1, 0, 2)  # [D, N, C*H*W] for MultiheadAttention
        
    #     # Apply temporal attention
    #     x_temp = self.temporal_attention(x_temp)  # [D, N, C*H*W]
    #     x_temp = x_temp.permute(1, 0, 2)  # [N, D, C*H*W]
        
    #     # Project to 12 channels and reshape
    #     x_diff = self.proj(x_temp)  # [N, D, 12*H*W]
    #     x_diff = x_diff.view(N, D, 12, H, W)  # [16, 160, 12, 128, 128]
        
    #     # Reshape for stem12: [N*D, 12, H, W]
    #     x_diff = x_diff.contiguous().view(N * D, 12, H, W)  # [2560, 12, 128, 128]
    #     x_diff = self.stem12(x_diff)  # [2560, 8, 32, 32]

    #     # Process original frames
    #     x3 = x.contiguous().view(N * D, C, H, W)  # [2560, 3, 128, 128]
    #     x = self.stem11(x3)  # [2560, 8, 32, 32]

    #     # Fusion layer 1 using learnable alpha and beta
    #     x_path1 = self.alpha * x + self.beta * x_diff
    #     x_path1 = self.stem21(x_path1)  # [2560, 16, 16, 16]

    #     # Fusion layer 2
    #     x_path2 = self.stem22(x_diff)  # [2560, 16, 16, 16]
    #     x = self.alpha * x_path1 + self.beta * x_path2  # [2560, 16, 16, 16]

    #     return x
    

# ====================modification above=========================
class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()

        self.scale = 0.02
        self.dim = dim * mlp_ratio

        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim),
        )


    def forward(self, x):
        B, N, C = x.shape
  
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)

        x_fre = torch.fft.fft(x, dim=1, norm='ortho') # FFT on N dimension

        x_real = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.real, self.r) - \
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.i) + \
            self.rb
        )
        x_imag = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.r) + \
            torch.einsum('bnc,cc->bnc', x_fre.real, self.i) + \
            self.ib
        )

        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
        x = x.to(torch.float32)
        # x = x.real.to(torch.float32) # Safe, explicit, and removes the warning.
        # x = x.abs().to(torch.float32) #  Useful for FFT-based features, where magnitude is important.


        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
        return x


class CA_MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = CA_Mamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv, 
            expand=expand  
        )
    def forward(self, x,C_SA):
        B, N, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm,C_SA)    
        return x_mamba

class SA_MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = SA_Mamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv, 
            expand=expand  
        )
    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)    
        return x_mamba

class SAMambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = SA_Mamba(
            d_model=dim,  
            d_state=d_state,  
            d_conv=d_conv, 
            expand=expand  
        )
    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)    
        return x_mamba

class CABlock_CrossMamba(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = CA_MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim,mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)
        self.downsampleseq=nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3,stride=2, padding=1),
            nn.BatchNorm1d(64)
            # norm_layer(dim)
        )
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x,C_SA_list):
        B, D, C = x.size()  
        x_path = torch.zeros(x.size()).to("cuda:0") 
        C1= C_SA_list[0].permute(0,2,1)
        C1=C1.permute(0,2,1)
        x_o = self.drop_path(self.attn(x,C1)) 
        tt = D // 2
        for j in range(D//tt):
            C2=C_SA_list[1][j].permute(0,2,1)
            C2=C2.permute(0,2,1)
            x_div = self.attn(x[:,j * tt : (j + 1)* tt,:],C2) 
            x_path[:,j * tt : (j + 1)* tt,:] = x_div 
        x_o = x_o + self.drop_path(x_path) 
        tt = D // 4
        for j in range(D//tt):
            C3=C_SA_list[2][j].permute(0,2,1)
            C3=C3.permute(0,2,1)
            x_div = self.attn(x[:,j * tt : (j + 1)* tt,:],C3)
            x_path[:,j * tt : (j + 1)* tt,:] = x_div 
        x_o = x_o + self.drop_path(x_path) 
        tt = D // 8
        for j in range(D//tt):
            C4=C_SA_list[3][j].permute(0,2,1)
            C4=C4.permute(0,2,1)
            x_div = self.attn(x[:,j * tt : (j + 1)* tt,:],C4)
            x_path[:,j * tt : (j + 1)* tt,:] = x_div
        x_o = x_o + self.drop_path(x_path)
        x = x + self.drop_path(self.norm1(x_o)) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x

class SABlock_CrossMamba(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = SA_MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim,mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, D, C = x.size()  
        x_path= torch.zeros(x.size()).to("cuda:0") 
        C_SA=[]
        C_SA_0=[]
        C_SA_2=[]
        C_SA_4=[]
        C_SA_8=[]
        x_o,C_SA_0 =self.attn(x)
        x_o = self.drop_path(x_o) 
        tt = D // 2
        for j in range(D//tt):
            x_div,C_SA_2_j = self.attn(x[:,j * tt : (j + 1)* tt,:]) 
            C_SA_2.append(C_SA_2_j)
            x_path[:,j * tt : (j + 1)* tt,:] = x_div 
        x_o = x_o + self.drop_path(x_path) 
        tt = D // 4
        for j in range(D//tt):
            x_div,C_SA_4_j = self.attn(x[:,j * tt : (j + 1)* tt,:]) 
            C_SA_4.append(C_SA_4_j)
            x_path[:,j * tt : (j + 1)* tt,:] = x_div 
        x_o = self.drop_path(x_path) 
        tt = D // 8
        for j in range(D//tt):
            x_div,C_SA_8_j = self.attn(x[:,j * tt : (j + 1)* tt,:])
            C_SA_8.append(C_SA_8_j)
            x_path[:,j * tt : (j + 1)* tt,:] = x_div
        x_o = x_o + self.drop_path(x_path)

        x = x + self.drop_path(self.norm1(x_o)) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 

        C_SA=[C_SA_0,C_SA_2,C_SA_4,C_SA_8]

        return x,C_SA


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
# ==============Temporal transformer Mamba Model=========================
class TemporalModule(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=1, mlp_hidden_dim=None, use_transformer=True, dropout=0.1):
        """
        TemporalModule: Choose between Transformer encoder or MLP for temporal modeling.
        
        Args:
            dim: int, input feature dimension per frame (C*H*W)
            num_heads: int, number of attention heads for transformer
            num_layers: int, number of transformer encoder layers
            mlp_hidden_dim: int, hidden dimension for MLP
            use_transformer: bool, True -> Transformer, False -> MLP
            dropout: float, dropout rate
        """
        super().__init__()
        self.use_transformer = use_transformer

        if use_transformer:
            # Transformer encoder for temporal modeling
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim*4,
                dropout=dropout,
                activation='gelu',
                batch_first=True  # input [B, D, C]
            )
            self.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # Simple MLP along temporal dimension
            hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else dim
            self.temporal_model = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            )

    def forward(self, x):
        """
        x: [B, D, C] tensor
        returns: [B, D, C] tensor
        """
        return self.temporal_model(x)


class SSMamba(nn.Module):
    def __init__(self, 
                 depth=24, 
                 embed_dim_CA=64, 
                 embed_dim_SA=64, 
                 mlp_ratio=2,
                 drop_rate=0.,
                 drop_path_rate_CA=0.1,
                 drop_path_rate_SA=0.1,
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()

        ## CA Layers
        self.embed_dim_CA = embed_dim_CA

        self.Fusion_Stem_CA = CAFusion_Stem(dim=embed_dim_CA//4) # best results (3d_diff = True)
        self.attn_mask_CA = Attention_mask()

        self.stem3_CA = nn.Sequential(
            nn.Conv3d(embed_dim_CA//4, embed_dim_CA, kernel_size=(2, 5, 5), stride=(2, 1, 1),padding=(0,2,2)),
            nn.BatchNorm3d(embed_dim_CA),
        )

        dpr_CA = [x.item() for x in torch.linspace(0, drop_path_rate_CA, depth)]  # stochastic depth decay rule
        inter_dpr_CA = [0.0] + dpr_CA
        self.blocks_CrossCA = nn.ModuleList([CABlock_CrossMamba(
            dim = embed_dim_CA, 
            mlp_ratio = mlp_ratio,
            drop_path=inter_dpr_CA[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(depth)])

        ## SA Layers
        self.embed_dim_SA = embed_dim_SA

        self.Fusion_Stem_SA = SAFusion_Stem(dim=embed_dim_SA//4) # best results (3d_diff = False)
        self.attn_mask_SA = Attention_mask()
        self.stem3_SA = nn.Sequential(
            nn.Conv3d(embed_dim_SA//4, embed_dim_SA, kernel_size=(2, 5, 5), stride=(2, 1, 1),padding=(0,2,2)),
            nn.BatchNorm3d(embed_dim_SA),
        )

        dpr_SA = [x.item() for x in torch.linspace(0, drop_path_rate_SA, depth)]  # stochastic depth decay rule
        inter_dpr_SA = [0.0] + dpr_SA
        self.blocks_CrossSA = nn.ModuleList([SABlock_CrossMamba(
            dim = embed_dim_SA, 
            mlp_ratio = mlp_ratio,
            drop_path=inter_dpr_SA[i], 
            norm_layer=nn.LayerNorm,)
        for i in range(depth)])

        self.ConvBlockLast= nn.Conv1d(128, 1, kernel_size=1,stride=1, padding=0) # change 128 to 64

        self.upsample = nn.Upsample(scale_factor=2)
        self.opt = False # True

        # init
        self.apply(segm_init_weights)
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.upchannels=nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1, stride=1, padding=0).to('cuda')

        # Intensity Enhancement Cross Attention
        self.SA_lca = LCA(dim=embed_dim_SA, num_heads=8)
        self.CA_lca = LCA(dim=embed_dim_CA, num_heads=8)
        self.T_lca = TemporalLCA(dim=embed_dim_CA, num_heads=8)

        # gated fusion
        self.fusion_gate = nn.Sequential(nn.Linear(self.embed_dim_SA, self.embed_dim_CA),nn.Sigmoid())


        # self.temporal_module = TemporalModule(dim=x_SA.shape[-1], num_heads=8, num_layers=2, use_transformer=True)
    def forward(self, x):
        # x = x[:, :, 3:, :, :] # only use last 3 channels (RGB)
        B, D, C, H, W = x.shape 
        # print('Input x shape:',x.shape)  #[N, D, C, H, W]
        x_SA = x[:, :, :, :, :]  
        x_CA = x[:, :, :, :, :]  

        Bf, Df, Cf, Hf, Wf = x_SA.shape
        Bs, Ds, Cs, Hs, Ws = x_CA.shape

        #### SA Pathway
        # print('X_SA shape 1:',x_SA.shape)
        x_SA = self.Fusion_Stem_SA(x_SA)    #[N*D C H/8 W/8] 
        x_SA = x_SA.view(Bf,Df,self.embed_dim_SA//4,Hf//8,Wf//8).permute(0,2,1,3,4) 
        x_SA = self.stem3_SA(x_SA)

        x_CA = self.Fusion_Stem_CA(x_CA)    #[N*D C H/8 W/8] 
        x_CA = x_CA.view(Bs,Ds,self.embed_dim_CA//4,Hs//8,Ws//8).permute(0,2,1,3,4) 
        x_CA = self.stem3_CA(x_CA) 

        #---- LCA module ------- Intensity Enhancement Cross Attention
        Bl, Cl, Dl, Hl, Wl = x_SA.shape
        x_SA_lca = x_SA.permute(0,2,1,3,4).reshape(Bl*Dl, Cl, Hl, Wl)
        x_CA_lca = x_CA.permute(0,2,1,3,4).reshape(Bl*Dl, Cl, Hl, Wl)
        # x_CA_lca = self.CA_lca(x_CA_lca, x_SA_lca)
        x_SA_lca = self.SA_lca(x_SA_lca, x_CA_lca) # 8.64MAE
        # x_SA_lca = self.SA_lca(x_SA_lca, x_SA_lca)
        x_SA = x_SA_lca.view(Bl, Dl, Cl, Hl, Wl).permute(0,2,1,3,4)
        # x_CA = x_CA_lca.view(Bl, Dl, Cl, Hl, Wl).permute(0,2,1,3,4)
        

        mask_f = torch.sigmoid(x_SA) 
        # print('Mask f sigmoid:', mask_f.shape)
        mask_f = self.attn_mask_SA(mask_f) 
        # print('Mask-f atten:', mask_f.shape)
        x_SA = x_SA * mask_f 
        # print('X_SA*Mask:',x_SA.shape)

        x_SA = torch.mean(x_SA,4) 
        # print('X_SA torch m, 4:',x_SA.shape)
        x_SA = torch.mean(x_SA,3) 
        # print('X_SA torch m, 3:',x_SA.shape)
        x_SA = rearrange(x_SA, 'b c t -> b t c') 
        # print('X_SA rearrange:',x_SA.shape)

        #### CA Pathway

        # x_CA = self.Fusion_Stem_CA(x_CA)    #[N*D C H/8 W/8] 
        # x_CA = x_CA.view(Bs,Ds,self.embed_dim_CA//4,Hs//8,Ws//8).permute(0,2,1,3,4) 
        # x_CA = self.stem3_CA(x_CA) 

        mask_s = torch.sigmoid(x_CA) 
        mask_s = self.attn_mask_CA(mask_s) 
        x_CA = x_CA * mask_s 

        x_CA = torch.mean(x_CA,4) 
        x_CA = torch.mean(x_CA,3) 
        x_CA = rearrange(x_CA, 'b c t -> b t c') 

        # x_SA=self.T_lca(x_SA, x_CA)         # option 2

        C_SA_list=[]
        for blk in self.blocks_CrossSA:
            x_SA,C_SA_list_j= blk(x_SA) 
            C_SA_list.append(C_SA_list_j)
        C_len=len(C_SA_list)
        i=0
        for blk in self.blocks_CrossCA:
            if i<C_len:
                x_CA = blk(x_CA,C_SA_list[i]) 
        # print('X_SA CrossA:',x_SA.shape)
        # print('X_CA CrossA:',x_CA.shape)

        # x_SA=self.T_lca(x_SA, x_CA) #it reduce MAE to 9.0 , not good (option 1)
        # Finfal fusion of SA and TC features using gated fusion
        if self.opt == True:
            x_concat=torch.cat((x_SA,x_CA),dim=2) # (option 1)
        else:
        # gated fusion instead of direct concat
            gate = self.fusion_gate(x_SA)  # [B, D, embed_dim_CA]
            x_concat = gate * x_SA + (1 - gate) * x_CA  # [B, D, embed_dim_CA] (option 2)
            x_concat = x_CA * gate   # (option 2)
        # print('x_concat gated shape:',x_concat.shape)

        rPPG = x_concat.permute(0,2,1)
         
        # print('rPPG permute:',rPPG.shape) 
        rPPG = self.upsample(rPPG) 
        # print('rPPG upsample:',rPPG.shape)
        rPPG = self.ConvBlockLast(rPPG)    #[N, 1, D] lets change to 64
        # print('rPPG Conv1d:',rPPG.shape)
        rPPG = rPPG.squeeze(1) 
        # print('rPPG squeeze:',rPPG.shape)
        # print(f"Memory allocated2: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        return rPPG

    # def forward(self, x):
    #     print(f"Input shape: {x.shape}")  # [B, D, C, H, W] = [16, 160, 3, 128, 128]
        
    #     B, D, C, H, W = x.shape 
    #     print(f"Initial dimensions - B:{B}, D:{D}, C:{C}, H:{H}, W:{W}")
        
    #     # SA Pathway
    #     x_SA = x[:, :, :, :, :]  
    #     x_CA = x[:, :, :, :, :]  
    #     print(f"\nSA Pathway input shape: {x_SA.shape}")
    #     print(f"CA Pathway input shape: {x_CA.shape}")

    #     #### SA Pathway Processing ####
    #     print("\n=== Processing SA Pathway ===")
    #     x_SA = self.Fusion_Stem_SA(x_SA)
    #     print(f"After x_SA Fusion Stem: {x_SA.shape}")                  # [B*D, C, H//8, W//8]
        
    #     x_SA = x_SA.view(B, D, self.embed_dim_SA//4, H//8, W//8).permute(0,2,1,3,4) # 16B, 16C, 160D, 16H, 16W
    #     print(f"After SA view & permute: {x_SA.shape}")                 # [B, C, D, H//8, W//8] 16B, 16C, 160D, 16H, 16W
        
    #     x_SA = self.stem3_SA(x_SA)  # 16, 64, 80, 16, 16
    #     print(f"After x_SA stem3_SA (Conv3D): {x_SA.shape}")            # [B, embed_dim_SA, D//2, H//8, W//8]
        
    #     mask_f = torch.sigmoid(x_SA)
    #     mask_f = self.attn_mask_SA(mask_f)
    #     x_SA = x_SA * mask_f
    #     print(f"After SA attention masking: {x_SA.shape}")
        
    #     x_SA = torch.mean(x_SA, 4)
    #     print(f"After SA width pooling: {x_SA.shape}")  # [B, C, D//2, H//8] 16, 64, 80, 16])
        
    #     x_SA = torch.mean(x_SA, 3)
    #     print(f"After SA height pooling: {x_SA.shape}")  # [B, C, D//2] 16, 64, 80
        
    #     x_SA = rearrange(x_SA, 'b c t -> b t c')
    #     print(f"Final SA features before blocks: {x_SA.shape}")  # [B, D//2, embed_dim_SA], 16, 80, 64

    #     #### CA Pathway Processing ####
    #     print("\n=== Processing CA Pathway ===")
    #     x_CA = self.Fusion_Stem_CA(x_CA)
    #     print(f"After CA Fusion Stem: {x_CA.shape}")  # [B*D, C, H//8, W//8]
        
    #     x_CA = x_CA.view(B, D, self.embed_dim_CA//4, H//8, W//8).permute(0,2,1,3,4)
    #     print(f"After CA view & permute: {x_CA.shape}")  # [B, C, D, H//8, W//8]
        
    #     x_CA = self.stem3_CA(x_CA)
    #     print(f"After CA stem3_CA (Conv3D): {x_CA.shape}")  # [B, embed_dim_CA, D//2, H//8, W//8]
        
    #     mask_s = torch.sigmoid(x_CA)
    #     mask_s = self.attn_mask_CA(mask_s)
    #     x_CA = x_CA * mask_s
    #     print(f"After CA attention masking: {x_CA.shape}")
        
    #     x_CA = torch.mean(x_CA, 4)
    #     print(f"After CA width pooling: {x_CA.shape}")  # [B, C, D//2, H//8]
        
    #     x_CA = torch.mean(x_CA, 3)
    #     print(f"After CA height pooling: {x_CA.shape}")  # [B, C, D//2]
        
    #     x_CA = rearrange(x_CA, 'b c t -> b t c')
    #     print(f"Final CA features before blocks: {x_CA.shape}")  # [B, D//2, embed_dim_CA],16, 80, 64

    #     #### Cross Attention Processing ####
    #     print("\n=== Processing Cross Attention ===")
    #     C_SA_list = []
    #     for i, blk in enumerate(self.blocks_CrossSA):
    #         x_SA, C_SA_list_j = blk(x_SA)
    #         print(f"SA Block {i+1} output shape: {x_SA.shape}")
    #         C_SA_list.append(C_SA_list_j)
        
    #     print("\n=== Processing CA Blocks with SA Masks ===")
    #     for i, blk in enumerate(self.blocks_CrossCA):
    #         if i < len(C_SA_list):
    #             x_CA = blk(x_CA, C_SA_list[i])
    #             print(f"CA Block {i+1} output shape: {x_CA.shape}") # 16, 80, 64

    #     #### Final Fusion ####
    #     print("\n=== Final Fusion ===")
    #     x_concat = torch.cat((x_SA, x_CA), dim=2) # 16, 80, 128
    #     print(f"After concatenation: {x_concat.shape}")  # [B, D//2, embed_dim_SA+embed_dim_CA]
        
    #     rPPG = x_concat.permute(0, 2, 1) # [16, 128, 80]
    #     print(f"After permute: {rPPG.shape}")  # [B, embed_dim_total, D//2]
        
    #     rPPG = self.upsample(rPPG) # [16, 128, 160]
    #     print(f"After upsample: {rPPG.shape}")  # [B, embed_dim_total, D]
        
    #     rPPG = self.ConvBlockLast(rPPG) # [16, 1, 160]
    #     print(f"After final conv: {rPPG.shape}")  # [B, 1, D]
        
    #     rPPG = rPPG.squeeze(1) # [16, 160]
    #     print(f"Final output shape: {rPPG.shape}")  # [B, D]
        
    #     return rPPG

if __name__ == "__main__":
    # Test the TSCAN_Improved model
    model = SSMamba()
    test_input = torch.randn(4, 160, 3, 128, 128) # (batch_size=2, depth=20, channels=6, height=36, width=36)
    output = model(test_input)
    print("Output shape:", output.shape)  # Should be (2,)  