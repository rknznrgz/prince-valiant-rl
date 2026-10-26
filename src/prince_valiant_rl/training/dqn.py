"""Lossless fixed-window replay and masked Double DQN; no pixel observations."""
import copy
from collections import deque
import numpy as np
import torch
from torch import nn
from .temporal_models import WindowPath,WINDOW_KEYS,window_tensors

ARCHITECTURE='dueling_window_dqn_v1'

class DuelingQ(nn.Module):
    def __init__(self,shapes,actions,config):
        super().__init__();self.shapes=shapes;self.actions=actions;self.config=dict(config)
        self.encoder=WindowPath(shapes,config['gru_hidden'],config);self.encoder.head=nn.Identity()
        self.value=nn.Linear(config['gru_hidden'],1);self.advantage=nn.Linear(config['gru_hidden'],actions)
        nn.init.zeros_(self.value.weight);nn.init.zeros_(self.value.bias)
        nn.init.orthogonal_(self.advantage.weight,.01);nn.init.zeros_(self.advantage.bias)
    def forward(self,o):
        if any(v.device!=next(self.parameters()).device for v in o.values()):raise RuntimeError('Q input/device mismatch')
        z=self.encoder(o);a=self.advantage(z);return self.value(z)+a-a.mean(-1,keepdim=True)
    def transfer_encoder(self,actor):
        state={k:v for k,v in actor.state_dict().items() if not k.startswith('head.')}
        self.encoder.load_state_dict(state,strict=True)
        return 'BC actor encoder only; action-logit head explicitly excluded; new value/advantage heads'

def masked_argmax(q,mask):
    mask=torch.as_tensor(mask,device=q.device,dtype=torch.bool)
    if mask.ndim==1:mask=mask.expand_as(q)
    if q.shape!=mask.shape or not mask.any(-1).all():raise ValueError('Invalid Q action mask')
    return q.masked_fill(~mask,-torch.inf).argmax(-1)

@torch.no_grad()
def double_target(online,target,following,mask,rewards,discount,bootstrap):
    choice=masked_argmax(online(following),mask)
    next_q=target(following).gather(1,choice[:,None]).squeeze(1)
    return rewards+discount*bootstrap*next_q

def storage_dtype(k):
    if k in ('tiles','classes'):return np.uint16
    if k in ('categorical','entity_codes','projectile_codes','trigger_codes','terrain_valid','history_valid'):return np.uint8
    return np.float32

class WindowReplay:
    """Store BOTH exact conditioning windows, losslessly, including reset validity.

    No reconstruction via adjacent ring indices, and no fp16 quantization. PER
    is optional; new experiences start at the maximum current priority.
    """
    def __init__(self,capacity,example,actions=12,alpha=0.):
        self.capacity=capacity;self.size=0;self.position=0;self.alpha=alpha
        self.obs={};self.following={}
        for k in WINDOW_KEYS:
            dtype=storage_dtype(k);shape=(capacity,*example[k].shape)
            self.obs[k]=np.empty(shape,dtype);self.following[k]=np.empty(shape,dtype)
        self.actions=np.empty(capacity,np.int16);self.rewards=np.empty(capacity,np.float32)
        self.discounts=np.empty(capacity,np.float32);self.bootstrap=np.empty(capacity,bool)
        self.masks=np.empty((capacity,actions),bool);self.next_masks=self.masks.copy()
        self.duration=np.empty(capacity,np.int32);self.priorities=np.ones(capacity,np.float32)
        self.source=np.empty(capacity,np.int16);self.episode=np.empty(capacity,np.int64)
        self.total_inserted=0
    @property
    def nbytes(self):
        return sum(a.nbytes for a in [*self.obs.values(),*self.following.values(),self.actions,self.rewards,self.discounts,self.bootstrap,self.masks,self.next_masks,self.duration,self.priorities,self.source,self.episode])
    def add(self,row):
        p=self.position
        for k in WINDOW_KEYS:
            for original,destination in ((row['obs'][k],self.obs[k]),(row['following'][k],self.following[k])):
                cast=np.asarray(original,dtype=destination.dtype)
                if not np.array_equal(cast,original):raise ValueError('Lossy replay conversion: '+k)
                destination[p]=cast
        if not row['mask'][row['action']] or not np.any(row['next_mask']):raise ValueError('Masked replay action')
        for field,attr in (('action','actions'),('reward','rewards'),('discount','discounts'),('bootstrap','bootstrap'),('mask','masks'),('next_mask','next_masks'),('duration','duration'),('source','source'),('episode','episode')):
            getattr(self,attr)[p]=row[field]
        self.priorities[p]=max(1.,float(self.priorities[:self.size].max())) if self.size else 1.
        self.position=(p+1)%self.capacity;self.size=min(self.size+1,self.capacity);self.total_inserted+=1
    def sample(self,n,rng,beta=1.):
        if not self.size:raise ValueError('Empty replay')
        if self.alpha:
            probabilities=self.priorities[:self.size].astype(np.float64)**self.alpha;probabilities/=probabilities.sum()
            ix=rng.choice(self.size,n,p=probabilities);weights=(self.size*probabilities[ix])**(-beta);weights/=weights.max()
        else:ix=rng.integers(self.size,size=n);weights=np.ones(n,np.float32)
        return dict(obs={k:v[ix] for k,v in self.obs.items()},following={k:v[ix] for k,v in self.following.items()},
            actions=self.actions[ix],rewards=self.rewards[ix],discount=self.discounts[ix],bootstrap=self.bootstrap[ix],
            mask=self.masks[ix],next_mask=self.next_masks[ix],weights=weights.astype(np.float32),indices=ix)
    def priority(self,ix,td):
        # Repeated sampled indices get maximum error, not arbitrary last-write order.
        for i in np.unique(ix):self.priorities[i]=max(.0001,float(np.max(np.abs(td[ix==i]))))
    def state_dict(self):
        d=dict(self.__dict__)
        if self.size<self.capacity:
            d={k:({f:a[:self.size].copy() for f,a in v.items()} if k in ('obs','following') else v[:self.size].copy() if isinstance(v,np.ndarray) else v) for k,v in d.items()}
        return d
    def checkpoint_dict(self):
        # Torch storages stream independently. Pickling all numpy arrays using
        # protocol2 creates a giant temporary bytes/unicode copy on Windows.
        d=self.state_dict()
        return {k:({f:torch.from_numpy(a) for f,a in v.items()} if k in ('obs','following') else
            torch.from_numpy(v) if isinstance(v,np.ndarray) else v) for k,v in d.items()}
    def load_state_dict(self,d):
        if self.capacity!=d['capacity'] or self.alpha!=d['alpha']:raise ValueError('Replay configuration mismatch')
        for k,v in d.items():
            if k in ('obs','following'):
                for key,a in v.items():getattr(self,k)[key][:len(a)]=a.numpy() if isinstance(a,torch.Tensor) else a
            elif isinstance(v,(np.ndarray,torch.Tensor)):getattr(self,k)[:len(v)]=v.numpy() if isinstance(v,torch.Tensor) else v
            else:setattr(self,k,v)

class NStep:
    """Owner-local queues; flush on any episode boundary, bootstrap separately."""
    def __init__(self,n,gamma):
        if n<1:raise ValueError('Positive n-step required')
        self.n=n;self.gamma=gamma;self.queues={}
    def aggregate(self,q):
        first=q[0];r=0.;discount=1.;duration=0;last=first
        for row in list(q)[:self.n]:
            r+=discount*row['reward'];duration+=row['duration'];discount*=self.gamma**row['duration'];last=row
            if row['boundary']:break
        return dict(obs=first['obs'],following=last['following'],mask=first['mask'],next_mask=last['next_mask'],action=first['action'],
            reward=r,discount=discount,bootstrap=last['bootstrap'],duration=duration,source=first['source'],episode=first['episode'])
    def push(self,owner,row):
        q=self.queues.setdefault(owner,deque());q.append(row);out=[]
        if row['boundary']:
            while q:out.append(self.aggregate(q));q.popleft()
        elif len(q)>=self.n:out.append(self.aggregate(q));q.popleft()
        return out

def optimize_q(model,target,optimizer,replay,rng,device,batch=128,beta=.4,clip=10.,tau=.005,reward_scale=1.):
    data=replay.sample(batch,rng,beta);x=window_tensors(data['obs'],device,batch=True);n=window_tensors(data['following'],device,batch=True)
    actual=model(x).gather(1,torch.as_tensor(data['actions'],device=device,dtype=torch.long)[:,None]).squeeze(1)
    tensor=lambda k:torch.as_tensor(data[k],device=device,dtype=torch.float32)
    expected=double_target(model,target,n,data['next_mask'],tensor('rewards')*reward_scale,tensor('discount'),tensor('bootstrap'))
    td=expected-actual;loss=(nn.functional.smooth_l1_loss(actual,expected,reduction='none')*tensor('weights')).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Double DQN loss')
    optimizer.zero_grad(set_to_none=True);loss.backward();norm=float(nn.utils.clip_grad_norm_(model.parameters(),clip));optimizer.step()
    with torch.no_grad():
        for p,t in zip(model.parameters(),target.parameters()):t.lerp_(p,tau)
    error=td.detach().cpu().numpy();replay.priority(data['indices'],error)
    return dict(td_huber=float(loss.detach()),td_abs=float(np.mean(np.abs(error))),q_mean=float(actual.detach().mean()),target_mean=float(expected.mean()),gradient_norm=norm,
                terminal_fraction=float(1-np.mean(data['bootstrap'])),max_abs_q=float(actual.detach().abs().max()))

def load_q(path,device):
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p.get('architecture')!=ARCHITECTURE:raise ValueError('Not a DQN checkpoint')
    model=DuelingQ(p['shapes'],p['actions'],p['model_config']).to(device);model.load_state_dict(p['state_dict']);return model,p
