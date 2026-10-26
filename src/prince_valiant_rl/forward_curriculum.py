"""Forward exposure scheduling; all goals are evaluation-only, never policy inputs."""
import hashlib,json
import numpy as np
from .backward_level1 import BackwardCurriculum
from .paths import ROOT
from .navigation import WIDTHS

VERSION='forward_route_exposure_v1'

def load_manifest(config):
    path=ROOT/config['forward_manifest']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=config['forward_manifest_sha256']:
        raise ValueError('Forward checkpoint manifest changed')
    return json.loads(path.read_text())

def milestone_reached(goal,info):
    """Coordinate/gate witness, not a kill detector or collision-box estimate."""
    if info.get('end_reason')=='unsupported_mode':return False
    if info.get('event_totals',{}).get('lives_lost',0) or info.get('event_totals',{}).get('death_started',0):return False
    if goal['kind']=='completion':return bool(info['completion'])
    if goal['kind']=='objective':
        return (info.get('event_totals',{}).get('objective_cleared',0)>0
                and info.get('health',0)>0 and info.get('life_phase_raw')==0
                and any(info.get('objective_flags_raw',{}).get(a)==1 for a in goal['addresses']))
    return (info['section']==goal['section'] and goal['section'] in range(len(WIDTHS))
            and goal['x']<=info['player_x']<WIDTHS[goal['section']]
            and 0<=info['player_y']<192 and info['health']>0
            and info.get('life_phase_raw')==0 and info.get('player_state_raw')==0)

class ForwardExposureCurriculum(BackwardCurriculum):
    def __init__(self,config,valid,start,end):
        super().__init__(config,valid,start,end)
        if config['forward_curriculum_version']!=VERSION:raise ValueError('Unknown forward curriculum')
        self.manifest=load_manifest(config);self.zone=self.manifest['zones'][config['forward_stage_index']]
        self.frontier=self.zone['frame']
        self.support=[]
        if config.get('forward_support_checkpoints'):
            path=ROOT/config['forward_support_checkpoints']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=config['forward_support_sha256']:raise ValueError('Support checkpoint bank changed')
            self.support=[p['frame'] for p in json.loads(path.read_text())['checkpoints'] if p['roundtrip_verified']]

    def sample(self,rng,role=None):
        if role=='coverage' and self.config.get('policy_recovery_ids') and rng.random()<self.config.get('policy_recovery_fraction',.5):
            return dict(recovery_checkpoint=str(rng.choice(self.config['policy_recovery_ids'])),frame=self.frontier,
                horizon=self.config['forward_training_horizon'],stage=self.stage,frontier=self.frontier,
                rehearsal=True,kind='policy_recovery',training_segment=True)
        if role=='full_start':
            return dict(frame=self.start,horizon=self.config['full_horizon'],stage=self.stage,frontier=self.frontier,
                        rehearsal=True,kind='full_start',training_segment=False)
        if role=='coverage':
            last=min(len(self.manifest['zones'])-1,self.config['forward_stage_index']+self.config.get('forward_lookahead_zones',0))
            upper=self.manifest['zones'][last+1]['frame'] if last+1<len(self.manifest['zones']) else self.end
            anchors=sorted({f for z in self.manifest['zones'][self.config['forward_stage_index']:last+1] for f in z['checkpoint_frames']}
                           |{f for f in self.support if self.frontier<=f<upper})
            frame=self.frontier if rng.random()<.5 else int(rng.choice(anchors))
            kind='current_zone'
        elif role=='frontier':
            earlier=self.config.get('forward_earlier_reset_frames',[z['frame'] for z in self.manifest['zones'][:self.config['forward_stage_index']]])
            frame=int(rng.choice(earlier)) if earlier else self.start;kind='earlier_zone'
        else:raise ValueError('Forward collectors require explicit supported roles')
        if frame not in self.valid:raise ValueError('Unverified forward start')
        horizon=max(self.config['forward_training_horizon'],min(7200,3*(self.zone['goal']['witness_frame']-self.zone['frame'])+300))
        return dict(frame=frame,horizon=horizon,stage=self.stage,frontier=self.frontier,
                    rehearsal=True,kind=kind,training_segment=frame!=self.start)

    def observe(self,case,success,frames):
        # External 30-epoch reviews control FORWARD advancement. Never call the
        # inherited backward advancement logic or treat a milestone as terminal.
        return False
