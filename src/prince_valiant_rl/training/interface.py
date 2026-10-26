"""Final training interface: no labels, pixels, or undifferentiated RAM inputs."""
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..paths import ROOT
from ..scene_state import ForestTerrain
from ..game_state import extract_game_state
from ..rl_env import PrinceValiantRamEnv
from .events import EventTracker, DEFAULT_WEIGHTS, reward_components, section, word, bootstrap_allowed, live_position

BASE_MASKS = (0,128,64,32,16,2,1,129,130,65)
BUTTONS = ((128,'RIGHT'),(64,'LEFT'),(32,'DOWN'),(16,'UP'),(8,'START'),(4,'SELECT'),(2,'B'),(1,'A'))
def action_name(mask):
    return '+'.join(name for bit,name in BUTTONS if mask & bit) or 'NOOP'

def load_catalog():
    return tuple(json.loads((ROOT/'config/training.json').read_text())['action_masks'])

class FeatureEncoder:
    """Fixed scaling has no fitted or running statistics to leak/freeze.

    Categorical raw codes remain integer embedding inputs. Per-slot masks
    distinguish vacant arrays. Sprite rectangles are deliberately excluded.
    """
    def __init__(self):
        self.terrain = ForestTerrain(ROOT/'roms/prince_valiant.nes')

    def __call__(self, r, previous, history, state=None):
        state = state if state is not None else extract_game_state(r, previous, terrain_decoder=self.terrain)
        p = state['player']; valid = state['validity']['player_position_valid'] and section(r) is not None
        x,y = p['world_x'],p['y']; x,y = (x,y) if valid else (0,0)
        s = section(r)
        motion_valid=previous is not None and live_position(r) and live_position(previous) and section(previous)==s
        vx=p['velocity_x_derived'] if motion_valid else None
        vy=p['velocity_y_derived'] if motion_valid else None
        scalar = [x/1792,y/192,word(r,0x6E)/1792,valid,
                  (p['health'] or 0)/255,(p['retries_remaining'] or 0)/8,
                  int(r[0x60])==0,int(r[0x60])==1,state['terminal']['game_over'],
                  (vx or 0)/8,(vy or 0)/16,
                  vx is not None,vy is not None,
                  p['vertical_step']/16,p['horizontal_step_magnitude']/8]
        scalar += [float(s==i) for i in range(3)]
        h = history.observable().copy(); h[:3] = np.maximum(h[:3],0)/1792
        scalar += h.tolist()
        entities=[]; codes=[]
        for e in state['entities']:
            occupied = e['occupied']
            entities.append([(e['world_x']-x)/256 if occupied else 0,
                             (e['y']-y)/192 if occupied else 0,
                             e['screen_anchor_x']/256 if occupied else 0,
                             float(occupied),float(e['render_eligible']),float(e['hidden']),
                             e['motion_or_timer_raw']/65535 if occupied else 0])
            codes.append([e['kind_code'],e['flags_raw'],e['phase_raw'],e['direction_raw'],
                          e['animation_raw'],*e['arguments_raw']])
        shots=[]; shot_codes=[]
        for e in state['player_projectiles']:
            shots.append([(e['world_x']-x)/256 if e['active'] else 0,
                          (e['y']-y)/192 if e['active'] else 0,float(e['active']),float(e['flying'])])
            shot_codes.append([e['direction_raw'],e['state_raw'],e['animation_raw']])
        terrain=state['nearby_terrain']; mask=terrain['valid'].astype(np.float32)
        # Index256 denotes unavailable; valid code zero retains its own embedding.
        tile=np.where(mask,terrain['tile_ids'],256).astype(np.int64)
        collision=np.where(mask,terrain['collision_classes'],256).astype(np.int64)
        scalar += [float(state['hud']['valid']),*[(v/9) for v in state['hud']['score_digits']],
                   *[(v/9) for v in state['hud']['timer_digits']],
                   float(terrain['player_offset'][0])/(17*16),float(terrain['player_offset'][1])/(12*16)]
        trigger=state['triggers']
        rect=np.asarray(trigger['rectangles'],np.float32).copy()
        rect[:,0]=(rect[:,0]-x)/256; rect[:,1]=(rect[:,1]-y)/192
        rect[:,2:]/=256; rect*=np.asarray(trigger['valid'])[:,None]
        cat=[int(r[a]) for a in (0x65,0x5F,0x5E,0x33,0x2F4,0x4D,0x529,0xF3,0x284)]
        return dict(scalars=np.asarray(scalar,np.float32),
                    categorical=np.asarray(cat,np.int64),entities=np.asarray(entities,np.float32),
                    entity_codes=np.asarray(codes,np.int64),projectiles=np.asarray(shots,np.float32),
                    projectile_codes=np.asarray(shot_codes,np.int64),tiles=tile,classes=collision,terrain_valid=mask,
                    triggers=np.concatenate([rect,np.asarray(trigger['valid'],np.float32)[:,None]],axis=1),
                    trigger_codes=np.clip(trigger['action_arguments_raw'],0,255).astype(np.int64))

class TrainingEnv(gym.Env):
    """Corrected rewards; legacy inspector/env reward behavior stays compatible.

    Every repeated native frame is observed and charged. Reward history is seeded
    from authentic checkpoint prehistory, never transferred between episodes.
    """
    def __init__(self, checkpoint='gameplay', task='manager', frame_skip=1, horizon=600,
                 weights=None, render_mode=None):
        if not isinstance(frame_skip,int) or frame_skip<1:raise ValueError('frame_skip must be positive')
        checkpoint_path=Path(checkpoint)
        if checkpoint_path.suffix!='.json':checkpoint_path=ROOT/'savestates'/f'{checkpoint}.json'
        elif not checkpoint_path.is_absolute():checkpoint_path=ROOT/checkpoint_path
        self.core=PrinceValiantRamEnv(checkpoint=checkpoint_path,
                                     frame_skip=1,max_episode_frames=horizon,render_mode=render_mode,
                                     action_masks=load_catalog())
        self.action_space=self.core.action_space; self.task=task; self.frame_skip=frame_skip
        self.weights=weights or DEFAULT_WEIGHTS[task]; self.encoder=FeatureEncoder()
        self.frame_observer=None  # Optional exact native-frame structured observation hook.
        sample=self.encoder(np.zeros(2048,np.uint8),None,EventTracker())
        self.observation_space=spaces.Dict({k: spaces.Box(-4096,4096,v.shape,v.dtype) if v.dtype==np.float32
                                           else spaces.Box(0,256,v.shape,v.dtype) for k,v in sample.items()})

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tracker=EventTracker(); self._previous=None; self._last=None
        def seed_history(r):
            self.tracker.seed(r); self._previous=self._last; self._last=r
        _,info=self.core.reset(seed=seed,options={'replay_observer':seed_history})
        self.ram=self.core._ram.copy(); self.decisions=0
        if section(self.ram) is None:
            self.core.close();raise ValueError('Training checkpoint is outside verified forest map scope')
        obs=self.encoder(self.ram,self._previous,self.tracker)
        if self.observation_space is None:
            self.observation_space=spaces.Dict({k: spaces.Box(-4096,4096,v.shape,v.dtype) if v.dtype==np.float32
                                                else spaces.Box(0,256,v.shape,v.dtype) for k,v in obs.items()})
        info.update(reward_history=self.tracker.observable().tolist(),elapsed_seconds=0,policy_decisions=0)
        return obs,info

    def step(self, action):
        totals={}; weighted={}; total=0; elapsed=0;availability={}
        for _ in range(self.frame_skip):
            old=self.ram
            _,_,term,trunc,info=self.core.step(action)
            self.ram=self.core._ram.copy()
            if section(self.ram) is None:
                term=False;trunc=True;self.core._done=True;info['end_reason']='scope_exit'
            events=self.tracker.transition(old,self.ram)
            reward,parts=reward_components(events,self.weights); total+=reward; elapsed+=1
            for key,value in events.items():
                availability[key]=availability.get(key,0)+int(value is not None)
                if value is not None: totals[key]=(totals.get(key) or 0)+value
                elif key not in totals: totals[key]=None
            for key,part in parts.items():
                if part['available']:
                    if weighted.get(key) is None: weighted[key]=dict(part,unweighted=0,weighted=0,available_frames=0,unavailable_frames=0)
                    if weighted[key]['weighted'] is None:weighted[key]['weighted']=0;weighted[key]['unweighted']=0
                    weighted[key]['available']=True
                    weighted[key]['unweighted']+=part['unweighted']; weighted[key]['weighted']+=part['weighted']
                    weighted[key]['available_frames']+=1
                else:
                    if key not in weighted:weighted[key]=dict(part,available_frames=0,unavailable_frames=0)
                    weighted[key]['unavailable_frames']+=1
            # The core already decoded this exact RAM/previous pair. Reuse it
            # and the encoded final frame rather than decoding terrain 3 times.
            obs=self.encoder(self.ram,old,self.tracker,state=info['state'])
            if self.frame_observer is not None:
                self.frame_observer(self.core.elapsed_frames,obs)
            if term or trunc: break
        self.decisions+=1
        info.update(events=totals,reward_components=weighted,transition_frames=elapsed,
                    event_available_frames=availability,
                    elapsed_seconds=info['elapsed_frames']/60,policy_decisions=self.decisions,
                    bootstrap_valid=bootstrap_allowed(term,trunc,info['end_reason']),
                    reward_history=self.tracker.observable().tolist())
        return obs,float(total),term,trunc,info

    def render(self): return self.core.render()
    def close(self): self.core.close()
