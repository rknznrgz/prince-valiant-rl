"""Verified RAM interventions for worker-only controlled forest exercises."""
import copy,json
from collections import Counter
import gymnasium as gym
import numpy as np
from ..paths import ROOT
from .starts import CachedStart
from .interface import load_catalog,TrainingEnv
from .events import EventTracker,section,live_position,word,DEFAULT_WEIGHTS

def scene_catalog():return json.loads((ROOT/'config/controlled_scenes.json').read_text())

class ControlledScene(gym.Env):
    """Full owner-local snapshots first, semantic-guarded RAM edits second.

    Scene IDs, edits and masks are info/config only, never actor observation.
    Type/phase/record identity is rechecked every native frame; slots are not
    treated as permanent entities. Success is task completion, not game-over.
    """
    def __init__(self,scene_id,frame_skip=1,config=None,allow_unvalidated=False):
        self.config=copy.deepcopy(config or scene_catalog()['scenes'][scene_id]);self.scene_id=scene_id
        if not allow_unvalidated and self.config['status']!='validated':raise ValueError('Scene not validated: '+scene_id)
        if set(self.config.get('remove_types',[]))-{2,4,16,17,18,19,20,21}:raise ValueError('Unverified removal type')
        if any(not 1<=v<=255 for v in self.config.get('health',[])):raise ValueError('Invalid player health')
        if any(v not in (0,1) for v in self.config.get('facing',[])):raise ValueError('Unverified facing value')
        if any(not 1<=v<=24 for v in self.config.get('hits_remaining',[])):raise ValueError('Unverified type14 counter range')
        if set(self.config.get('immortal_packed_types',[]))-{19}:raise ValueError('Packed counter protection verified only for type19')
        if frame_skip<1 or not isinstance(frame_skip,int):raise ValueError('Positive frame_skip required')
        if not np.isfinite(self.config.get('constraint_failure_cost',0)) or self.config.get('constraint_failure_cost',0)<0:raise ValueError('Nonnegative finite task constraint cost required')
        self.frame_skip=frame_skip;self.cache={};self.env=None;self.catalog=load_catalog()
        self.action_space=gym.spaces.Discrete(len(self.catalog));self.rom=(ROOT/'roms/prince_valiant.nes').read_bytes()
        prototype=TrainingEnv(self.config['sources'][0],task=self.config['worker'],horizon=self.config['timeout'])
        if self.config.get('input_latches'):
            from .input_features import configure_encoder
            configure_encoder(prototype,True)
        self.observation_space=prototype.observation_space;prototype.close()
        self.allowed=np.asarray([not((self.config.get('mask_attacks') and mask&2) or (self.config.get('mask_start') and mask&8)) for mask in self.catalog],bool)
        self.edits=Counter();self.terminal=False

    def _keep(self,r,slot):
        return int(r[0xC1+slot])==self.config.get('keep_type') and ('keep_slot' not in self.config or slot==self.config['keep_slot'])

    def _rom_byte(self,address):
        if 0x8000<=address<0xC000:return self.rom[16+2*16384+address-0x8000]
        if 0xC000<=address<0x10000:return self.rom[16+7*16384+address-0xC000]
        raise ValueError('Unverified scene object-list pointer')

    def _intervene(self,r):
        if int(r[0x4D])!=2:return
        remove=set(self.config.get('remove_types',[]));keep=self.config.get('keep_type')
        # Do not block a kept combat type's initial ROM record. Dynamic kind4
        # ground spawn controllers are selectively disabled as a verified type.
        pointer=word(r,0x80)
        if pointer>=0x8000:
            for record in range(25):
                kind=self._rom_byte(pointer+10*record)
                if kind==0:break
                if kind in remove and kind!=keep:
                    if r[0x2CE+record]!=128:r[0x2CE+record]=128;self.edits['spawn_records']+=1
        for slot in range(10):
            kind=int(r[0xC1+slot])
            if kind in remove and not self._keep(r,slot):
                r[0xC1+slot]=0;self.edits[f'removed_type_{kind}']+=1
            # Exact executed DEC $B7,X zero branch: only type14 phase1.
            if self.config.get('immortal_type14') and kind==14 and r[0x99+slot]==1:
                if r[0xB7+slot]!=24:self.edits['counter_refills']+=1
                r[0xB7+slot]=24
            if kind in self.config.get('immortal_packed_types',[]):
                # Executed bank2 $87E7-$87FA preserves high3 bits, subtracts
                # low5 and converts the actor to its defeat behavior at zero.
                value=(int(r[0xB7+slot])&224)|31
                if r[0xB7+slot]!=value:self.edits['packed_counter_refills']+=1
                r[0xB7+slot]=value

    def _before(self,r):
        self._intervene(r);self._pre=r.copy()

    def _after(self,r):
        # Capture gameplay outcomes before intervention cleanup. A removal is
        # never a kill. Each counter refill is compared against executed RAM.
        pre=self._pre;defeat14=False;defeat19=False;hits=0
        if self.config['worker']=='evade':
            for slot in range(10):
                if self._keep(pre,slot) and int(r[0xC1+slot])!=int(pre[0xC1+slot]):self.intervention_failure=True
        for slot in range(10):
            old=int(pre[0xC1+slot]);new=int(r[0xC1+slot])
            if old==14 and pre[0x99+slot]==1:
                defeated=(new==13 and pre[0x204]==0 and r[0x204]==1)
                defeat14|=defeated
                if defeated:hits+=1
                elif new==14 and r[0x99+slot]==1 and int(r[0xB7+slot])==int(pre[0xB7+slot])-1:hits+=1
            if old==19 and self.config.get('keep_type')==19 and slot==self.config.get('keep_slot'):
                # Executed type19 hit callback -> type8 defeat animation. No
                # slot disappearance is counted. Identity/nearby anchor guarded.
                same_record=pre[0xE9+slot]==r[0xE9+slot]
                defeated=bool(new in (7,8) and (int(pre[0xB7+slot])&31)==1 and same_record and abs(word(pre,0x28C+2*slot)-word(r,0x28C+2*slot))<=8)
                defeat19|=defeated
                if defeated:hits+=1
                elif new==19 and (int(pre[0xB7+slot])&31)-(int(r[0xB7+slot])&31)==1:hits+=1
        self.hits+=hits;self.defeats+=int(defeat14 or defeat19)
        self.dead|=bool(pre[0x60]==0 and r[0x60]==1)
        if self.config['worker']=='progress':
            dest=self.config['destination']
            self.reached|=bool(live_position(r) and section(r)==dest['section'] and word(r,0x56)>=dest['x'] and word(r,0x58)<192)
        raw_types=r[0xC1:0xCB].tolist();self._intervene(r)
        self.frame_log.append(dict(pre_type14_counter=int(pre[0xB7]),post_type14_counter=int(r[0xB7]),
                                   hits=hits,defeated=bool(defeat14 or defeat19),health=int(r[0xFC]),types=r[0xC1:0xCB].tolist(),raw_types=raw_types))
        if self.dead or self.reached or self.defeats:
            # End at the actual base frame, even inside a held multi-frame action.
            self.env.core.max_episode_frames=self.env.core.elapsed_frames+1

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);c=self.config
        source=(options or {}).get('source') or c['sources'][int(self.np_random.integers(len(c['sources'])))]
        if source not in c['sources']:raise ValueError('Source outside configured verified variants')
        if source not in self.cache:self.cache[source]=CachedStart(source,task=c['worker'],horizon=c['timeout'])
        e,_,info=self.cache[source].reset();self.env=e;e.frame_skip=self.frame_skip;e.core.max_episode_frames=c['timeout']
        from .input_features import configure_encoder
        configure_encoder(e,c.get('input_latches',False))
        self.observation_space=e.observation_space;self.edits=Counter();r=e.core.native.ram_buffer()
        if c.get('keep_type')==14 and not any(r[0xC1+s]==14 and r[0x99+s]==1 for s in range(10)):raise ValueError('Type14 phase1 source required')
        if c.get('keep_type')==19 and r[0xC1+c['keep_slot']]!=19:raise ValueError('Type19 source identity mismatch')
        if 'health' in c:r[0xFC]=int(self.np_random.choice(c['health']))
        if 'facing' in c:r[0x65]=int(self.np_random.choice(c['facing']))
        if c.get('clear_player_projectiles'):
            for a in (0x509,0x50B,0x50D):r[a]=0
        for slot in range(10):
            if r[0xC1+slot]==14 and r[0x99+slot]==1 and 'hits_remaining' in c:r[0xB7+slot]=int(self.np_random.choice(c['hits_remaining']))
        self._intervene(r);e.ram=r.copy();e.core._ram=e.ram.copy()
        # Clear derived motion across interventions; seed existing authentic
        # progress history, without rewarding pre-reset distance or restored HP.
        e._previous=None;e._last=e.ram.copy();e.tracker.seed(e.ram)
        self.reached=self.dead=False;self.intervention_failure=False;self.hits=self.defeats=0;self.terminal=False;self.frame_log=[]
        e.core.before_native_frame=self._before;e.core.after_native_frame=self._after
        obs=e.encoder(e.ram,None,e.tracker)
        info.update(scene_id=self.scene_id,interventions=dict(self.edits),action_mask=self.allowed.copy(),game_modified=True)
        return obs,info

    def step(self,action):
        if self.terminal:raise RuntimeError('Reset finished scene')
        if not self.action_space.contains(action) or not self.allowed[int(action)]:raise ValueError('Action is masked; index mapping is unchanged')
        self.frame_log=[];self.hits=self.defeats=0
        obs,_,game_term,game_trunc,info=self.env.step(action);c=self.config;events=copy.deepcopy(info['events'])
        if c['worker'] in ('fight','evade'):
            events.update(enemy_damage=self.hits if c.get('keep_type') in (14,19) else None,
                          enemy_kills=self.defeats if c.get('keep_type')==19 else None,encounter_cleared=self.defeats)
        weights=copy.deepcopy(DEFAULT_WEIGHTS[c['worker']]);weights.update(c.get('reward_weights',{}))
        # These task episodes end on death onset, before the later retry write.
        # Charge the verified death once rather than omit the delayed life cost.
        weights['death_started']=-150.
        if c['worker']=='evade':weights['elapsed_seconds']=0.;events['survived_seconds']=info['transition_frames']/60 if not self.dead else max(0,info['transition_frames']-1)/60;weights['survived_seconds']=c.get('survival_per_second',1.)
        if c['worker']=='evade':
            # Explicit edited-task constraint, not a game death/kill/escape.
            # Charge once at an alive identity-loss boundary; death already has
            # its own cost. Default zero preserves the comparison baseline.
            events['scene_constraint_failed']=int(self.intervention_failure and not self.dead and not game_term)
            weights['scene_constraint_failed']=-c.get('constraint_failure_cost',0.)
        events['destination_reached']=int(self.reached);weights['destination_reached']=c.get('destination_reward',20.) if c['worker']=='progress' else 0.
        from .events import reward_components
        reward,parts=reward_components(events,weights)
        # Scene-specific detectors replace unavailable global detectors. Keep
        # their coverage explicit rather than inherit the global None status.
        for key,part in parts.items():
            available=info['transition_frames'] if part['available'] else 0
            part.update(available_frames=available,unavailable_frames=info['transition_frames']-available)
            info['event_available_frames'][key]=available
        for key in ('enemy_damage','enemy_kills','encounter_cleared','destination_reached','survived_seconds'):
            if key in events:info['event_available_frames'][key]=info['transition_frames'] if events[key] is not None else 0
        timeout=self.env.core.elapsed_frames>=c['timeout']
        survival_success=timeout and c['worker']=='evade' and not self.dead and section(self.env.ram) is not None
        success=(self.reached or bool(self.defeats) or survival_success) and not self.intervention_failure
        terminated=bool(game_term or self.dead or self.reached or self.defeats) and not self.intervention_failure;truncated=bool(not terminated and (timeout or game_trunc or self.intervention_failure))
        self.terminal=terminated or truncated
        reason='scene_death' if self.dead else 'scene_success' if self.reached or self.defeats else 'survival_horizon' if survival_success else info['end_reason']
        if self.intervention_failure:reason='intervention_failure'
        info.update(events=events,reward_components=parts,scene_id=self.scene_id,scene_success=success,scene_failure=self.dead,
                    game_terminated=game_term,task_terminated=terminated and not game_term,end_reason=reason,
                    bootstrap_valid=not terminated and not self.intervention_failure and info['bootstrap_valid'],action_mask=self.allowed.copy(),
                    interventions=dict(self.edits),native_frame_interventions=copy.deepcopy(self.frame_log),game_modified=True)
        return obs,float(reward),terminated,truncated,info

    def render(self):return self.env.render()
    def close(self):
        for pool in self.cache.values():
            pool.env.core.before_native_frame=None;pool.env.core.after_native_frame=None;pool.close()
