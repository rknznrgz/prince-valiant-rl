"""Existing shared encoder as a CUDA comparison control, using only current state.

No archived checkpoint is modified. This adapter is explicitly NOT the separate
actor/critic upgrade; it permits matched data/BC/PPO budgets in the new runner.
"""
import torch
from torch import nn
from .models import ActorCritic,FEATURE_KEYS
from .temporal_models import window_tensors,masked_distribution

BASELINE_ARCHITECTURE='baseline_current_cuda_control_v1'

class CurrentPath(nn.Module):
    def __init__(self,legacy,index):super().__init__();self.legacy=legacy;self.index=index
    def forward(self,obs):
        x={k:obs[k][:,-1] for k in FEATURE_KEYS};output=self.legacy(x)[self.index]
        return output if self.index==0 else output[:,None]

class BaselineCudaControl(nn.Module):
    architecture=BASELINE_ARCHITECTURE
    def __init__(self,shapes,actions,config):
        super().__init__();self.shapes=shapes;self.actions=actions;self.config=dict(config)
        legacy=ActorCritic(shapes['scalars'][0],actions);self.actor=CurrentPath(legacy,0);self.critic=CurrentPath(legacy,1)
    def forward(self,obs):return self.actor(obs),self.critic(obs).squeeze(-1)
    @torch.no_grad()
    def act(self,obs,deterministic=False,action_mask=None):
        logits,value=self(window_tensors(obs,next(self.parameters()).device));dist=masked_distribution(logits,action_mask)
        action=dist.logits.argmax(-1) if deterministic else dist.sample()
        return int(action.item()),float(dist.log_prob(action).item()),float(value.item())
    def counts(self):
        legacy=self.actor.legacy
        return dict(unique_total=sum(p.numel() for p in self.parameters()),actor_head=sum(p.numel() for p in legacy.actor.parameters()),
            critic_head=sum(p.numel() for p in legacy.critic.parameters()),shared_encoder=True)

class ArchivedCudaPolicy(nn.Module):
    """Explicit evaluation adapter; no transfer into GRU parameters."""
    def __init__(self,legacy,device):super().__init__();self.legacy=legacy.to(device).eval();self.device=device
    @torch.no_grad()
    def act(self,obs,deterministic=False,action_mask=None):
        x={k:torch.as_tensor(obs[k][-1][None],device=self.device) for k in FEATURE_KEYS}
        logits,value=self.legacy(x);dist=masked_distribution(logits,action_mask);action=dist.logits.argmax(-1) if deterministic else dist.sample()
        return int(action.item()),float(dist.log_prob(action).item()),float(value.item())
