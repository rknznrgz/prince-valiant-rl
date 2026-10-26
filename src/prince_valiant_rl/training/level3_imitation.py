"""Separate raw-RAM Level3 observation contract and feed-forward imitation actor."""
import numpy as np
import torch
from torch import nn

SCHEMA='level3_raw_ram_imitation_v1'
OBSERVATION_BYTES=2048


def observe_ram(ram):
    ram=np.asarray(ram)
    if ram.dtype!=np.uint8 or ram.shape!=(OBSERVATION_BYTES,):raise ValueError('Expected current native uint8 RAM[2048]')
    return ram.copy()


class Level3ImitationActor(nn.Module):
    def __init__(self,classes):
        super().__init__()
        if type(classes) is not int or not 2<=classes<=256:raise ValueError('Invalid raw controller vocabulary size')
        self.register_buffer('mean',torch.zeros(OBSERVATION_BYTES));self.register_buffer('scale',torch.ones(OBSERVATION_BYTES))
        self.net=nn.Sequential(nn.Linear(OBSERVATION_BYTES,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,classes))

    def forward(self,ram):
        if ram.ndim!=2 or ram.shape[1]!=OBSERVATION_BYTES:raise ValueError('Expected batched RAM[B,2048]')
        return self.net((ram.float()/255.-self.mean)/self.scale.clamp_min(.02))
