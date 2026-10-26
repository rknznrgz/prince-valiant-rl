"""Separate actor and critic encoders/GRUs. Hidden states reset for every window."""
import json
import numpy as np
import torch
from torch import nn
from .models import FEATURE_KEYS
from .temporal import offsets_checked

ARCHITECTURE='separate_window_gru_v1'
BRANCH_NORM_ARCHITECTURE='separate_window_gru_branch_norm_v2'
WINDOW_KEYS=(*FEATURE_KEYS,'history_valid')

def configure_precision(c):
    """Checkpoint-declared GPU arithmetic; preserve legacy defaults explicitly."""
    mode=c.get('network_precision','legacy_default')
    if mode not in ('legacy_default','strict_fp32'):raise ValueError('Unknown network precision mode')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=mode!='strict_fp32'
    return dict(mode=mode,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_precision=torch.get_float32_matmul_precision())


def cuda_device(name='cuda:0'):
    device=torch.device(name)
    if device.type!='cuda':raise RuntimeError('GPU training requires CUDA; CPU fallback is disabled')
    if not torch.cuda.is_available():raise RuntimeError(f'CUDA unavailable: torch={torch.__version__}, CUDA build={torch.version.cuda}')
    torch.cuda.set_device(device)
    print(json.dumps(dict(device=str(device),gpu=torch.cuda.get_device_name(device),torch=torch.__version__,cuda=torch.version.cuda)),flush=True)
    return device

def window_tensors(obs,device,batch=False):
    if 'history_valid' not in obs:raise ValueError('Temporal validity mask is required')
    return {k:torch.as_tensor(np.asarray(v) if batch else np.asarray(v)[None],device=device)
            for k,v in obs.items()}

def indexed_window_tensors(data,ids,keys,device):
    """Gather a minibatch one field at a time and transfer directly to model dtypes.

    This avoids building a second dictionary of NumPy minibatches before CUDA
    transfer. The source may be a read-only memmap; only the requested rows are
    touched. It preserves the existing collator's source dtypes exactly; the
    model applies its existing embedding/float conversions inside forward.
    """
    if 'history_valid' not in keys:raise ValueError('Temporal validity mask is required')
    out={}
    for key in keys:
        # Advanced indexing creates just this one host batch; transfer it
        # directly using the same dtype as the original production collator.
        out[key]=torch.as_tensor(np.asarray(data[key][ids]),device=device)
    return out

def infer_shapes(observation):
    """Infer per-frame shapes from one observation or a complete temporal window."""
    result={}
    for key,value in observation.items():
        if key=='history_valid':continue
        a=np.asarray(value)
        if key in ('scalars','categorical'):result[key]=tuple(a.shape[1:] if a.ndim==2 else a.shape)
        elif key in ('tiles','classes','terrain_valid'):result[key]=tuple(a.shape[1:] if a.ndim==3 else a.shape)
        else:result[key]=tuple(a.shape[1:] if a.ndim>=3 else a.shape)
    return result

def masked_distribution(logits,mask=None):
    if mask is not None:
        mask=torch.as_tensor(mask,device=logits.device,dtype=torch.bool)
        if mask.ndim==1:mask=mask[None].expand_as(logits)
        if mask.shape!=logits.shape or not mask.any(-1).all():raise ValueError('Invalid action mask')
        logits=logits.masked_fill(~mask,-1e9)
    return torch.distributions.Categorical(logits=logits)

class WindowPath(nn.Module):
    def __init__(self,shapes,outputs,c):
        super().__init__();emb=c['embedding_dim'];te=c['terrain_embedding_dim']
        self.config_input_latches=bool(c.get('input_latches',False))
        self.codes=nn.Embedding(256,emb);self.tiles=nn.Embedding(257,te);self.classes=nn.Embedding(257,te)
        self.shapes={k:tuple(v) for k,v in shapes.items()}
        numeric=sum(int(np.prod(shapes[k])) for k in ('scalars','entities','projectiles','triggers'))
        if 'resident_valid' in shapes:numeric+=int(np.prod(shapes['resident_valid']))
        code_keys=('categorical','entity_codes','projectile_codes','trigger_codes')
        codes=sum(int(np.prod(shapes[k])) for k in code_keys)*emb
        if 'resident_codes' in shapes:codes+=int(np.prod(shapes['resident_codes']))*emb
        layers=[];width=numeric+codes+1
        for target in c['mlp_sizes']:layers.extend([nn.Linear(width,target),nn.LayerNorm(target),nn.ReLU()]);width=target
        self.mlp=nn.Sequential(*layers)
        self.spatial_keys=tuple(k for k in ('dynamic_actors','rope_points') if k in shapes)
        spatial_channels=sum(int(shapes[k][-1]) for k in self.spatial_keys)
        if any(tuple(shapes[k][:2])!=(12,17) for k in self.spatial_keys):raise ValueError('Spatial feature grids must align to 12x17 terrain')
        layers=[];channels=2*te+1+spatial_channels
        for target in c['cnn_channels']:layers.extend([nn.Conv2d(channels,target,3,padding=1),nn.ReLU()]);channels=target
        layers.extend([nn.Flatten(),nn.Linear(channels*12*17,c['cnn_projection']),nn.ReLU()]);self.cnn=nn.Sequential(*layers)
        normalize=bool(c.get('branch_feature_normalization',False))
        self.structured_normalize=nn.LayerNorm(width,elementwise_affine=False) if normalize else nn.Identity()
        self.terrain_normalize=nn.LayerNorm(c['cnn_projection'],elementwise_affine=False) if normalize else nn.Identity()
        self.gru=nn.GRU(width+c['cnn_projection'],c['gru_hidden'],c['gru_layers'],batch_first=True)
        self.head=nn.Linear(c['gru_hidden'],outputs);self.last_hidden_device=None
    def forward(self,o):
        supplied=set(o)-{'history_valid'};required=set(self.shapes)
        if required-supplied:raise ValueError(f'Observation schema lacks model inputs: {sorted(required-supplied)}')
        if supplied-required:raise ValueError(f'Checkpoint schema would ignore observation fields: {sorted(supplied-required)}')
        b,t=o['history_valid'].shape
        flat={k:v.reshape(b*t,*v.shape[2:]) for k,v in o.items() if k!='history_valid'}
        scalars=flat['scalars'][...,:self.shapes['scalars'][-1]]
        categorical=self.codes(flat['categorical'][...,:self.shapes['categorical'][-1]].long())
        if self.config_input_latches:
            categorical=torch.cat([categorical[:,:9],categorical[:,9:]*scalars[:,-1,None,None]],1)
        pieces=[scalars,categorical.flatten(1)]
        for numeric,codes,mask_index in (('entities','entity_codes',3),('projectiles','projectile_codes',2),('triggers','trigger_codes',4)):
            nshape=self.shapes[numeric];cshape=self.shapes[codes]
            num=flat[numeric][...,:nshape[-1]];raw_codes=flat[codes][...,:cshape[-1]]
            x=torch.cat([num,self.codes(raw_codes.long()).flatten(2)],-1)
            pieces.append((x*num[...,mask_index,None]).flatten(1))
        if 'resident_codes' in self.shapes:
            resident_mask=flat['resident_valid'][...,:self.shapes['resident_valid'][-1]].float()
            resident_codes=self.codes(flat['resident_codes'][...,:self.shapes['resident_codes'][-1]].long())
            pieces.append(resident_mask.flatten(1))
            pieces.append((resident_codes*resident_mask[...,None]).flatten(1))
        valid=o['history_valid'].reshape(b*t,1)
        structured=self.mlp(torch.cat([*pieces,valid],-1))
        terrain_parts=[self.tiles(flat['tiles'].long()),self.classes(flat['classes'].long()),flat['terrain_valid'][...,None]]
        terrain_parts.extend(flat[k].float() for k in self.spatial_keys)
        terrain=torch.cat(terrain_parts,-1).permute(0,3,1,2)
        fused=self.frame_features(structured,terrain,flat).reshape(b,t,-1)
        # Invalid padded observations carry zero features; their presence remains explicit in the MLP.
        fused=fused*o['history_valid'][...,None]
        hidden=fused.new_zeros(self.gru.num_layers,b,self.gru.hidden_size);self.last_hidden_device=str(hidden.device)
        outputs,_=self.gru(fused,hidden)
        return self.head(outputs[:,-1])

    def frame_features(self,structured,terrain,flat):
        """Per-timestamp fusion boundary; optional visual paths extend this hook."""
        return torch.cat([self.structured_normalize(structured),self.terrain_normalize(self.cnn(terrain))],-1)

class TemporalActorCritic(nn.Module):
    def __init__(self,shapes,actions,config):
        super().__init__();configure_precision(config);offsets_checked(config['history_offsets'])
        self.architecture=BRANCH_NORM_ARCHITECTURE if config.get('branch_feature_normalization',False) else ARCHITECTURE
        self.shapes={k:tuple(v) for k,v in shapes.items()};self.config=dict(config);self.actions=actions
        self.observation_schema=config.get('observation_version','legacy_v3')
        if self.observation_schema in ('rea_expanded_v1','rea_expanded_v2') and not {'dynamic_actors','rope_points'}<=set(self.shapes):
            raise ValueError('Expanded REA observation requires dynamic actor and rope grids in model shapes')
        if self.observation_schema=='rea_expanded_v2' and not {'resident_codes','resident_valid'}<=set(self.shapes):
            raise ValueError('REA v2 observation requires resident record inputs in model shapes')
        self.window_keys=tuple(self.shapes)+('history_valid',)
        temperature=config.get('policy_temperature',1.)
        if not np.isfinite(temperature) or temperature<=0:raise ValueError('Positive finite policy temperature required')
        self.actor=WindowPath(shapes,actions,config);self.critic=WindowPath(shapes,1,config)
        nn.init.orthogonal_(self.actor.head.weight,gain=.01);nn.init.zeros_(self.actor.head.bias)
    def forward(self,o):
        device=next(self.parameters()).device
        if any(v.device!=device for v in o.values()):raise RuntimeError('Input/model device mismatch')
        if o['history_valid'].shape[1]!=len(self.config['history_offsets']):raise ValueError('Temporal length mismatch')
        missing=set(self.shapes)-set(o)
        if missing:raise ValueError(f'Observation schema lacks model inputs: {sorted(missing)}')
        extra=set(o)-set(self.shapes)-{'history_valid'}
        if extra:raise ValueError(f'Checkpoint schema would ignore observation fields: {sorted(extra)}')
        return self.actor(o),self.critic(o).squeeze(-1)
    @torch.no_grad()
    def act(self,obs,deterministic=False,action_mask=None):
        logits,value=self(window_tensors(obs,next(self.parameters()).device))
        dist=masked_distribution(logits/self.config.get('policy_temperature',1.),action_mask);action=dist.logits.argmax(-1) if deterministic else dist.sample()
        return int(action.item()),float(dist.log_prob(action).item()),float(value.item())
    def counts(self):return {k:sum(p.numel() for p in getattr(self,k).parameters()) for k in ('actor','critic')}
    @torch.no_grad()
    def act_batch(self,observations,deterministic=False,action_masks=None):
        """One CUDA call for independent CPU emulator windows; no shared hidden state."""
        x=window_tensors({k:np.stack([o[k] for o in observations]) for k in self.window_keys},next(self.parameters()).device,batch=True)
        logits,values=self(x);dist=masked_distribution(logits/self.config.get('policy_temperature',1.),action_masks)
        actions=dist.logits.argmax(-1) if deterministic else dist.sample()
        return actions.cpu().numpy(),dist.log_prob(actions).cpu().numpy(),values.cpu().numpy()

def save_temporal(path,model,**metadata):
    torch.save(dict(architecture=getattr(model,'architecture',ARCHITECTURE),config=model.config,shapes=model.shapes,actions=model.actions,
        state_dict=model.state_dict(),metadata=metadata),path)

def load_temporal(path,device):
    # Keep serialized state on CPU while constructing/loading the model. Mapping
    # the payload to CUDA first duplicates every parameter there before Module.to
    # allocates the model and initializes cuDNN's flattened GRU weights.
    p=torch.load(path,map_location='cpu',weights_only=False)
    config=p.get('config',p.get('model_config'));configure_precision(config)
    if p.get('architecture') in (ARCHITECTURE,BRANCH_NORM_ARCHITECTURE):
        expected=BRANCH_NORM_ARCHITECTURE if config.get('branch_feature_normalization',False) else ARCHITECTURE
        if p['architecture']!=expected:raise ValueError('Architecture header/normalization config mismatch')
        m=TemporalActorCritic(p['shapes'],p['actions'],config)
    elif p.get('architecture')=='ram_visual_window_gru_v1':
        from .visual import VisualActorCritic
        m=VisualActorCritic(p['shapes'],p['actions'],config)
    elif p.get('architecture')=='baseline_current_cuda_control_v1':
        from .baseline_gpu_control import BaselineCudaControl
        m=BaselineCudaControl(p['shapes'],p['actions'],p['config'])
    else:raise ValueError('Incompatible checkpoint; no archived baseline parameter transfer is supported')
    m.load_state_dict(p['state_dict']);m=m.to(device);return m,p
