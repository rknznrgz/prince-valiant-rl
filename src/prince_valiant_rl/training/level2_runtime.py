"""Task-scoped Level2 raw-state baseline, keeping Level1 network dimensions.

Raw slot codes have no enemy/kill/hitbox labels. This partial observation profile
is distinct from a complete combat-state extractor and from the v2 placeholder.
Caller must establish Level2 provenance by a pinned reset/recording route.
"""
from collections import deque
import numpy as np
from .level2_model_contract import load_contract

SCHEMA='level2_control_energy_raw_v3'
SCALAR_ADDRESSES=(0x56,0x57,0x58,0x59,0x6E,0x6F,0x65,0x67,0x413,0x414,
                  0xFC,0xF7,0xFD,0x5E,0x5F,0x60,0x0D,0x0F,0x400,0x401,0x410,0x411,0x412)
CODE_ADDRESSES=(*range(0xC1,0xC9),0x5F,0x60,0xF7,0x0D,0x0F,0x400,0x401,0x410,0x411,0x412,0x63,0x65,0x414)
WEIGHTS={'completion':100.,'game_over':-25.,'energy_loss_fraction':-5.,'elapsed_seconds':-.01,
         'enemy_kills':0.,'stage_progress':0.,'score':0.,'shots_fired':0.}

def checked_ram(ram):
    a=np.asarray(ram)
    if a.shape[-1:]!=(2048,) or a.dtype!=np.uint8:raise ValueError('Expected uint8 2048-byte RAM')
    return a

def transition(pre,post,*,level2_scope):
    if level2_scope is not True:raise ValueError('Externally verified Level2 reset/route required')
    pre=checked_ram(pre);post=checked_ram(post)
    if pre.ndim!=1 or post.ndim!=1:raise ValueError('Single transition required')
    if int(pre[0xF3]) or int(pre[0x60])>=128:raise ValueError('Cannot advance a terminal Level2 task')
    success=int(post[0xF3])==127
    failure=int(post[0x60])==128 and int(post[0xF7])==255
    if int(post[0xF3]) not in (0,127) or (int(post[0x60])>=128 and not failure):
        raise ValueError('Unrecognized terminal/mode transition')
    if success and failure:raise ValueError('Conflicting terminal evidence')
    loss=max(0,int(pre[0xFC])-int(post[0xFC]))/255.
    components={'completion':100.*success,'game_over':-25.*failure,
                'energy_loss_fraction':-5.*loss,'elapsed_seconds':-.01/60.}
    return dict(reward=sum(components.values()),components=components,terminated=success or failure,
                end_reason='level_complete' if success else 'game_over' if failure else None,
                energy_loss_fraction=loss,scope='Pinned Level2 task only; not a RAM-only global level classifier')

class Encoder:
    def __init__(self):
        contract=load_contract();self.shapes=contract['shapes'];self.offsets=np.array(contract['model_config']['history_offsets'])
        self.actions=tuple(contract['action_masks'])
        self.codes={'categorical','classes','entity_codes','projectile_codes','resident_codes','tiles','trigger_codes'}
    def windows(self,ram,valid):
        ram=checked_ram(ram);valid=np.asarray(valid,dtype=np.float32)
        if ram.shape[:-1]!=valid.shape or ram.ndim!=3 or ram.shape[1]!=7:
            raise ValueError('Expected batch x seven history slots x RAM and validity')
        if not np.isin(valid,[0,1]).all():raise ValueError('Binary history validity required')
        bt=ram.shape[:2]
        result={k:np.zeros((*bt,*shape),dtype=np.int64 if k in self.codes else np.float32) for k,shape in self.shapes.items()}
        result['tiles'].fill(256);result['classes'].fill(256)
        result['scalars'][...,:len(SCALAR_ADDRESSES)]=ram[...,list(SCALAR_ADDRESSES)].astype(np.float32)/255.
        result['categorical'][...]=ram[...,list(CODE_ADDRESSES)]
        result['history_valid']=valid
        return result
    def indexed(self,ram,indices):
        indices=np.asarray(indices,dtype=np.int64)
        history=indices[:,None]+self.offsets[None,:]
        return self.windows(ram[np.maximum(history,0)],history>=0)

class History:
    def __init__(self,ram,encoder=None):
        self.encoder=Encoder() if encoder is None else encoder
        self.rows=deque([checked_ram(ram).copy()],maxlen=61)
        self.total=1
    def append(self,ram):self.rows.append(checked_ram(ram).copy());self.total+=1
    def observation(self):
        a=np.stack(self.rows);ix=len(a)-1+self.encoder.offsets
        valid=ix>=0
        return {k:v[0] for k,v in self.encoder.windows(a[np.maximum(ix,0)][None],valid[None]).items()}

def ten_step_arrays(rewards,terminals,gamma=.99997):
    rewards=np.asarray(rewards,dtype=np.float64);terminals=np.asarray(terminals,dtype=bool)
    if rewards.ndim!=1 or rewards.shape!=terminals.shape or not len(rewards) or not np.isfinite(rewards).all():
        raise ValueError('Invalid scalar replay')
    if not 0<gamma<=1:raise ValueError('Invalid gamma')
    n=len(rewards);out=np.zeros(n,np.float64);following=np.arange(n,dtype=np.int64)
    alive=np.ones(n,bool);bootstrap=np.ones(n,bool);discount=np.ones(n,np.float64)
    for k in range(10):
        ids=np.arange(n)+k;active=alive&(ids<n)
        out[active]+=discount[active]*rewards[ids[active]]
        discount[active]*=gamma;following[active]=ids[active]+1
        ended=np.zeros(n,bool);ended[active]=terminals[ids[active]]
        bootstrap[ended]=False;alive&=~ended
    # Nonterminal data exhaustion is truncation: retain bootstrap, never fake success.
    return dict(reward=out.astype(np.float32),following=following,discount=discount.astype(np.float32),bootstrap=bootstrap)
