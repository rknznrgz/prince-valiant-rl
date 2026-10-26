"""Independent actor-critics with categorical embeddings and local terrain CNN."""
import hashlib
import numpy as np
import torch
from torch import nn

from .schema import FEATURE_KEYS,WORKERS  # Re-export existing public imports.

def tensor_observation(obs, batch=False):
    return {k:torch.as_tensor(np.asarray(obs[k]) if batch else np.asarray(obs[k])[None]) for k in FEATURE_KEYS}

class ActorCritic(nn.Module):
    def __init__(self, scalar_size, actions, architecture_version=2):
        super().__init__()
        self.codes=nn.Embedding(256,3)
        self.tiles=nn.Embedding(257,4); self.classes=nn.Embedding(257,4)
        self.terrain=nn.Sequential(nn.Conv2d(9,8,3,stride=2,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(8*6*9,48),nn.ReLU())
        self.objects=nn.Sequential(nn.Linear(31,32),nn.ReLU())
        self.shots=nn.Sequential(nn.Linear(13,16),nn.ReLU())
        self.triggers=nn.Sequential(nn.Linear(20,16),nn.ReLU())
        layers=[nn.Linear(scalar_size+27+48+64+32+32,128)]
        if architecture_version>=2:layers.append(nn.LayerNorm(128))
        layers.extend([nn.Tanh(),nn.Linear(128,64),nn.Tanh()])
        self.encoder=nn.Sequential(*layers)
        self.architecture_version=architecture_version
        self.actor=nn.Linear(64,actions); self.critic=nn.Linear(64,1)
        # No running normalization: scales in interface.py are fixed and versioned.
        self.register_buffer('interface_version',torch.tensor(2 if architecture_version>=2 else 1))
        nn.init.orthogonal_(self.actor.weight,gain=.01); nn.init.zeros_(self.actor.bias)

    def pooled(self, values, mask):
        values=values*mask[...,None]
        return torch.cat([values.mean(1),values.max(1).values],-1)

    def forward(self, o):
        terrain=torch.cat([self.tiles(o['tiles'].long()),self.classes(o['classes'].long()),o['terrain_valid'][...,None]],-1).permute(0,3,1,2)
        entities=torch.cat([o['entities'],self.codes(o['entity_codes'].long()).flatten(2)],-1)
        shots=torch.cat([o['projectiles'],self.codes(o['projectile_codes'].long()).flatten(2)],-1)
        triggers=torch.cat([o['triggers'],self.codes(o['trigger_codes'].long()).flatten(2)],-1)
        combined=torch.cat([o['scalars'],self.codes(o['categorical'].long()).flatten(1),self.terrain(terrain),
                            self.pooled(self.objects(entities),o['entities'][...,3]),
                            self.pooled(self.shots(shots),o['projectiles'][...,2]),
                            self.pooled(self.triggers(triggers),o['triggers'][...,4])],-1)
        h=self.encoder(combined)
        return self.actor(h),self.critic(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False, action_mask=None):
        logits,value=self(tensor_observation(obs))
        if action_mask is not None:
            mask=torch.as_tensor(action_mask,dtype=torch.bool)
            if mask.shape!=(self.actor.out_features,) or not mask.any():raise ValueError('Invalid action mask')
            logits=logits.masked_fill(~mask[None],-1e9)
        dist=torch.distributions.Categorical(logits=logits)
        action=logits.argmax(-1) if deterministic else dist.sample()
        return int(action.item()),float(dist.log_prob(action).item()),float(value.item())

def parameter_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def freeze(model):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
