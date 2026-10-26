"""Exact native-frame windows; unavailable pre-reset history is zero with a mask."""
from collections import deque
import numpy as np
import gymnasium as gym
from .schema import FEATURE_KEYS

def offsets_checked(offsets):
    offsets=tuple(offsets)
    if not offsets or offsets[-1]!=0 or list(offsets)!=sorted(set(offsets)) or any(type(x)!=int or x>0 for x in offsets):
        raise ValueError('History offsets must be unique ascending integer native frames ending in zero')
    return offsets

def empty_observation(obs):
    return {k:np.full_like(v,256) if k in ('tiles','classes') else np.zeros_like(v)
            for k,v in obs.items() if k!='history_valid'}

class FrameHistory:
    def __init__(self,offsets=(-10,-5,0)):
        self.offsets=offsets_checked(offsets);self.frames=deque(maxlen=1-self.offsets[0]);self.current=None
    def reset(self,obs):
        self.frames.clear();self.current=None;self.keys=tuple(k for k in obs if k!='history_valid');self.push(0,obs)
    def push(self,frame,obs):
        if self.current is not None and frame!=self.current+1:raise ValueError('Missing or reordered native history frame')
        keys=getattr(self,'keys',tuple(k for k in obs if k!='history_valid'))
        if set(k for k in obs if k!='history_valid')!=set(keys):raise ValueError('Observation schema changed inside temporal history')
        self.frames.append((frame,{k:np.array(obs[k],copy=True) for k in keys}));self.current=frame
    def window(self):
        if self.current is None:raise RuntimeError('Reset history first')
        lookup=dict(self.frames);valid=[];rows=[];blank=empty_observation(self.frames[-1][1])
        for offset in self.offsets:
            frame=self.current+offset;valid.append(frame in lookup);rows.append(lookup.get(frame,blank))
            if frame>=0 and frame not in lookup:raise ValueError('Unavailable non-padding native frame')
        return dict(**{k:np.stack([r[k] for r in rows]) for k in self.keys},history_valid=np.asarray(valid,np.float32))

class TemporalEnv(gym.Wrapper):
    """CPU emulator; one shared history consumed at worker and manager selections."""
    def __init__(self,env,offsets=(-10,-5,0)):
        super().__init__(env);self.history=FrameHistory(offsets);self.base=None
        spaces={k:gym.spaces.Box(np.broadcast_to(v.low,(len(offsets),*v.shape)).copy(),
                np.broadcast_to(v.high,(len(offsets),*v.shape)).copy(),dtype=v.dtype)
                for k,v in env.observation_space.spaces.items()}
        spaces['history_valid']=gym.spaces.Box(0,1,(len(offsets),),np.float32)
        self.observation_space=gym.spaces.Dict(spaces)
    def reset(self,*,seed=None,options=None):
        if self.base is not None:self.base.frame_observer=None
        obs,info=self.env.reset(seed=seed,options=options)
        self.base=self.env if hasattr(self.env,'encoder') else self.env.env
        self.history.reset(obs);self.base.frame_observer=self.history.push
        return self.history.window(),info
    def step(self,action):
        _,r,t,u,info=self.env.step(action)
        return self.history.window(),r,t,u,info
    def close(self):
        if self.base is not None:self.base.frame_observer=None
        self.env.close()
