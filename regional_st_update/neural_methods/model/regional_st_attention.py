"""Opt-in regional spatiotemporal pulse extractor, 2026-11-26.

Uses this project's RegionalDINO waveform, mask and fusion contract with DINO
disabled. Local 3D convolutions precede spatial pooling; compact self-attention
or a dilated TCN mixes time independently for each region. All input timesteps
are retained. A running graph does not establish physiological accuracy.
"""
import torch
from torch import nn
from torch.nn import functional as F
from neural_methods.model.regional_dino import RegionalDINO


class TemporalAttentionBlock(nn.Module):
    """Pre-norm attention on [N,C,T], with local convolutions supplying order.

    No absolute phase/position generator is added. Attention dropout is zero;
    channel dropout shares its mask across time, as in the project baseline.
    This block is offline/noncausal and is not a pretrained image ViT.
    """
    def __init__(self, channels=48, heads=4, dropout=.1):
        super().__init__()
        if channels % heads:
            raise ValueError('channels must be divisible by heads')
        self.norm_attention=nn.LayerNorm(channels)
        self.attention=nn.MultiheadAttention(channels,heads,dropout=0.,batch_first=True)
        self.norm_ffn=nn.LayerNorm(channels)
        self.ffn=nn.Sequential(nn.Linear(channels,2*channels),nn.GELU(),nn.Linear(2*channels,channels))
        self.drop_attention=nn.Dropout1d(dropout)
        self.drop_ffn=nn.Dropout1d(dropout)

    def forward(self,x):
        z=self.norm_attention(x.transpose(1,2))
        mixed,_=self.attention(z,z,z,need_weights=False)
        x=x+self.drop_attention(mixed.transpose(1,2))
        refined=self.ffn(self.norm_ffn(x.transpose(1,2)))
        return x+self.drop_ffn(refined.transpose(1,2))


class RegionalSTAttention(RegionalDINO):
    """Continuous RGB-only waveform estimator using existing regional fusion.

    Input: raw [B,T,3,H,W], 0..255. Aux output matches RegionalDINO, so the
    existing loss can be evaluated without changing its definition. All nine
    regions use the same encoder; no regional information is mixed before the
    existing gate. Temporal length is never downsampled.

    spatial_size=12 matches the baseline's pre-encoder crop size. A size24
    comparison must include an equally resized baseline. temporal_backend='tcn'
    provides a six-block dilated control; 'attention' uses two local TCN blocks
    and three attention blocks. Their capacities are measured, not assumed equal.
    """
    def __init__(self,temporal_backend='attention',spatial_size=12,hidden_dim=48,
                 grid_size=3,context_dim=32,dropout=.1,fusion='learned'):
        if temporal_backend not in ('attention','tcn'):
            raise ValueError('temporal_backend must be attention or tcn')
        if isinstance(spatial_size,bool) or not isinstance(spatial_size,int) or spatial_size<4:
            raise ValueError('spatial_size must be an integer >=4')
        super().__init__(use_dino=False,fusion=fusion,grid_size=grid_size,
                         hidden_dim=hidden_dim,context_dim=context_dim,
                         num_tcn_blocks=2 if temporal_backend=='attention' else 6,
                         kernel_size=3,dropout=dropout)
        self.temporal_backend=temporal_backend
        self.spatial_size=spatial_size
        self.local_tcn_receptive_field=self.receptive_field
        self.temporal_support='full_observed_clip' if temporal_backend=='attention' else '253_frame_tcn_span_plus_3d_stem'
        self.pulse_stem=nn.Sequential(
            nn.Conv3d(6,16,3,stride=(1,2,2),padding=1,padding_mode='replicate'),
            nn.GELU(),
            nn.Conv3d(16,24,3,stride=(1,2,2),padding=1,padding_mode='replicate'),
            nn.GELU(),
            nn.Conv3d(24,24,3,padding=1,groups=24,padding_mode='replicate'),
            nn.GELU(),nn.Conv3d(24,24,1),nn.GELU(),
        )
        if temporal_backend=='attention':
            self.temporal_encoder=nn.Sequential(self.temporal_encoder,*[
                TemporalAttentionBlock(hidden_dim,4,dropout) for _ in range(3)
            ])

    def _local_features(self,rgb):
        batch,time,_,height,width=rgb.shape
        previous=torch.cat((rgb[:,:1],rgb[:,:-1]),dim=1)
        relative=(rgb-previous)/(rgb+previous+1e-3)
        crops,traces,qualities=[],[],[]
        for row in range(self.grid_size):
            top,bottom=row*height//self.grid_size,(row+1)*height//self.grid_size
            for col in range(self.grid_size):
                left,right=col*width//self.grid_size,(col+1)*width//self.grid_size
                color=rgb[:,:,:,top:bottom,left:right]
                delta=relative[:,:,:,top:bottom,left:right]
                trace=torch.cat((color.mean((-1,-2)),delta.mean((-1,-2))),dim=-1)
                traces.append(trace)
                qualities.append(torch.cat((trace,color.square().mean((-1,-2)),delta.square().mean((-1,-2))),dim=-1))
                patch=torch.cat((color,delta),dim=2).reshape(batch*time,6,bottom-top,right-left)
                patch=F.adaptive_avg_pool2d(patch,self.spatial_size)
                patch=patch.reshape(batch,time,6,self.spatial_size,self.spatial_size).permute(0,2,1,3,4)
                crops.append(patch)
        # Explicit axes: [B,R,C,T,h,w] -> [B*R,C,T,h,w].
        crops=torch.stack(crops,dim=1).reshape(batch*self.num_regions,6,time,self.spatial_size,self.spatial_size)
        encoded=self.pulse_stem(crops)
        # Spatial pooling follows all local spatiotemporal filtering.
        spatial=encoded.mean((-1,-2)).transpose(1,2).reshape(batch,self.num_regions,time,24)
        return spatial,torch.stack(traces,dim=1),torch.stack(qualities,dim=1)
