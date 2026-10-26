"""Human-supervised state policy and explicit known-state retrieval baseline."""
import hashlib
import numpy as np
import torch
from torch import nn

SCHEMA='level2_full_ram_bc_v55'


class DemonstrationActor(nn.Module):
    def __init__(self,classes):
        super().__init__();self.register_buffer('mean',torch.zeros(2048));self.register_buffer('scale',torch.ones(2048))
        self.net=nn.Sequential(nn.Linear(2048,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,classes))

    def forward(self,ram):return self.net((ram.float()/255.-self.mean)/self.scale.clamp_min(.02))


class KnownStatePolicy:
    """A nonparametric imitation policy; no sequence cursor or frame-based lookup."""
    def __init__(self,ram,masks):
        if ram.dtype!=np.uint8 or ram.ndim!=2 or ram.shape[1]!=2048 or masks.shape!=(len(ram),):raise ValueError('Invalid policy memory')
        self.lookup={}
        for row,mask in zip(ram,masks):
            key=hashlib.sha256(row.tobytes()).digest();value=int(mask)
            if not 0<=value<=255:raise ValueError('Invalid controller mask')
            if key in self.lookup and self.lookup[key]!=value:raise ValueError('Conflicting state action labels')
            self.lookup[key]=value

    def action(self,ram):
        if ram.shape!=(2048,) or ram.dtype!=np.uint8:raise ValueError('Invalid live RAM')
        return self.lookup.get(hashlib.sha256(ram.tobytes()).digest())
