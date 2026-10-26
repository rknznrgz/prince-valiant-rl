"""Optional demonstration-corridor potential; never a policy input or controller.

This measures proximity to positions actually occupied on the verified route,
not terrain safety, hitboxes, or whether an action is correct. Existing progress
and completion events remain the task objective. No timestamp matching is used.
"""
import hashlib
from pathlib import Path
import numpy as np
from .game_state import word
from .navigation import live, map_section
from .paths import ROOT

VERSION='level1_route_corridor_potential_v1'
PHASE_VERSION='level1_route_corridor_stone_phase_v2'
CELL=4


def distance_potential(occupied):
    distance=np.where(occupied,0.,1e6).astype(np.float32)
    # Exact Manhattan distance transform on a four-pixel grid. Vertical error
    # is twice as costly as horizontal error; these are shaping scales only.
    for axis,cost in ((-2,4.),(-1,8.)):
        v=np.moveaxis(distance,axis,0)
        for i in range(1,len(v)):v[i]=np.minimum(v[i],v[i-1]+cost)
        for i in range(len(v)-2,-1,-1):v[i]=np.minimum(v[i],v[i+1]+cost)
    return -np.minimum(distance/64.,1.)


def build_corridor(rams):
    occupied=np.zeros((3,449,64),bool)
    for r in rams:
        if live(r) and word(r,0x58)<256:
            occupied[map_section(r),min(448,word(r,0x56)//CELL),word(r,0x58)//CELL]=True
    if not occupied.reshape(3,-1).any(axis=1).all():
        raise ValueError('Require demonstrated live positions in all three sections')
    return distance_potential(occupied),occupied


def stone_phase_key(r):
    # Scope is the executed first-bridge kind11 at X432. No meaning is applied
    # to reused slots, other objects, rope tips, or unverified later stones.
    if (not live(r) or map_section(r)!=0 or int(r[0xC1])!=11 or
            word(r,0x28C)!=432 or not 360<=word(r,0x56)<=480):return None
    phase=int(r[0x99]);counter=int(r[0x2BA])
    if phase not in (129,130,131,132) or counter>=120:return None
    return phase-129,counter//10


def build_phase_corridor(rams):
    occupied=np.zeros((4,12,449,64),bool);samples=np.zeros((4,12),np.int32)
    for r in rams:
        key=stone_phase_key(r)
        if key is not None and word(r,0x58)<256:
            occupied[*key,min(448,word(r,0x56)//CELL),word(r,0x58)//CELL]=True;samples[key]+=1
    return distance_potential(occupied),samples


class RouteGuidance:
    def __init__(self,config):
        version=config.get('route_guidance_version')
        if version not in (VERSION,PHASE_VERSION):raise ValueError('Explicit corridor version required')
        path=Path(config['route_guidance_data'])
        if not path.is_absolute():path=ROOT/path
        if hashlib.sha256(path.read_bytes()).hexdigest()!=config['route_guidance_sha256']:
            raise ValueError('Corridor digest mismatch')
        self.phase_table=None
        with np.load(path,allow_pickle=False) as z:
            self.table=z['potential']
            if version==PHASE_VERSION:
                self.phase_table=z['phase_potential'];self.phase_samples=z['phase_samples']
                if self.phase_table.shape!=(4,12,449,64) or self.phase_samples.shape!=(4,12):raise ValueError('Invalid phase corridor')
                if not np.isfinite(self.phase_table).all() or self.phase_table.min() < -1 or self.phase_table.max()>0:raise ValueError('Invalid phase potential range')
        if self.table.shape!=(3,449,64) or not np.isfinite(self.table).all() or self.table.min() < -1 or self.table.max()>0:
            raise ValueError('Invalid bounded corridor table')

    def potential(self,r):
        if not live(r):return 0.
        y=word(r,0x58)
        if y>=256:return -1.
        key=stone_phase_key(r) if getattr(self,'phase_table',None) is not None else None
        if key is not None and self.phase_samples[key]>=8:
            return float(self.phase_table[*key,min(448,word(r,0x56)//CELL),y//CELL])
        return float(self.table[map_section(r),min(448,word(r,0x56)//CELL),y//CELL])

    def difference(self,a,b,gamma,*,bootstrap_valid=True):
        return gamma*(self.potential(b) if bootstrap_valid else 0.)-self.potential(a)
