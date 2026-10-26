"""Exact, explicitly authorized demonstration-only continuation of committed2000.

The historical retention failure remains a failure; no online collector is used.
"""
from pathlib import Path
import hashlib
import json
import math
import time

STUDY_ID = 'dqfd_demo_continuation_20261025'
ROOT = Path(__file__).resolve().parents[3]
PARENT_REL = 'training/level1_dqfd_demonstration_extension_20261025'
AUTH_REL = 'config/dqfd_demo_continuation_20261025.json'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_budget(row, protocol_sha256):
    from .dqfd_study_budget import canonical_hash
    fields={'schema','study_id','protocol_sha256','authorization_sha256','predecessor_sha256',
        'original_total_ticks','prior_charged_ticks','remaining_ticks','preparation_charge_ticks','receipt_sha256'}
    if set(row)!=fields or row['study_id']!=STUDY_ID or row['receipt_sha256']!=canonical_hash(row):
        raise ValueError('continuation budget identity/hash invalid')
    if row['protocol_sha256']!=protocol_sha256:raise ValueError('learning protocol differs')
    auth=ROOT/AUTH_REL
    if digest(auth)!=row['authorization_sha256']:raise ValueError('continuation authority changed')
    authorization=json.loads(auth.read_text())
    expected={'start_update':2000,'end_update':5000,'model_io_seconds':1800,'demo_seconds':1800,
              'slice_seconds':3600,'episode_count_max':3,'native_frames_max':108000}
    if authorization.get('allocation')!=expected:raise ValueError('continuation allocation differs')
    predecessor=ROOT/'analysis/level1_dqfd_extended_study_20261025/postrun_budget.json'
    if digest(predecessor)!=row['predecessor_sha256']:raise ValueError('continuation predecessor changed')
    previous=json.loads(predecessor.read_text())
    for key in ['original_total_ticks','prior_charged_ticks','remaining_ticks','preparation_charge_ticks']:
        if type(row[key]) is not int or row[key]<0:raise ValueError('invalid continuation ticks')
    if (row['original_total_ticks']!=126000000000 or row['preparation_charge_ticks']!=6000000000
            or row['prior_charged_ticks']!=previous['used_ticks']+row['preparation_charge_ticks']
            or row['remaining_ticks']!=row['original_total_ticks']-row['prior_charged_ticks']):
        raise ValueError('continuation accounting mismatch')
    remaining=row['remaining_ticks']/10000000
    return {'study_id':STUDY_ID,'original_total_active_seconds':12600,'legacy_original_total_active_seconds':5400,
        'updates':3000,'start_update':2000,'end_update':5000,'batch_size':128,'beta':.6,'seed':20261024,
        'model_io_active_seconds':1800,'demo_encoding_active_seconds':1800,'slice_active_seconds':3600,
        'model_io_active_ticks':18000000000,'demo_encoding_active_ticks':18000000000,'slice_active_ticks':36000000000,
        'prior_charged_ticks':row['prior_charged_ticks'],'remaining_active_ticks':row['remaining_ticks'],
        'remaining_original_active_seconds':f'{remaining:.7f}',
        'remaining_seconds_deadline':math.nextafter(remaining,-math.inf),
        'execution_budget_receipt_sha256':row['receipt_sha256'],'authorization_sha256':row['authorization_sha256'],
        'base_protocol_sha256':row['authorization_sha256'],'whole_pilot_native_frames_remaining':747626}


def restore_payload(learner,payload,*,torch_module):
    """Restore every learning state and verify exact hashes before any update."""
    from .dqfd_pretraining import learner_state_hashes
    counters=payload['counters']
    if any(counters.get(k)!=2000 for k in ['successful_steps','attempted_steps','applied_steps']):
        raise ValueError('continuation requires exact2000 counters')
    if counters.get('target_copies')!=2 or counters.get('failed_after_step') or counters.get('unknown_step_outcomes'):
        raise ValueError('ambiguous source learning state')
    learner.online.load_state_dict(payload['online'],strict=True)
    learner.target.load_state_dict(payload['target'],strict=True);learner.target.eval()
    learner.optimizer.load_state_dict(payload['optimizer'])
    learner.demo_index._priority[:]=payload['demo_priority']
    learner.rng.bit_generator.state=payload['rng']
    for key,value in counters.items():setattr(learner.counters,key,int(value))
    torch_module.set_rng_state(payload['torch_cpu_rng'])
    if payload['torch_cuda_rng']:torch_module.cuda.set_rng_state_all(payload['torch_cuda_rng'])
    if learner_state_hashes(learner)!=payload['state_hashes']:
        raise ValueError('restored learning-state hashes differ from committed2000')


def create_continuation_learner(*,root,device='cuda:0'):
    import copy
    import numpy as np
    import torch
    from . import dqfd_production as production
    from .dqfd import DuelingQ,optimizer_for
    from .dqfd_execution import DQfDBatchLearner
    from .dqfd_replay import DemoReplayIndex
    from .dqfd_transitions import TransitionCache
    from .dqfd_pretraining import load_committed_milestone
    root=Path(root);parent=root/PARENT_REL
    authorization=json.loads((root/AUTH_REL).read_text())
    if digest(parent/'checkpoint_2000.pt')!=authorization['parent_checkpoint_sha256']:
        raise ValueError('authorized parent checkpoint changed')
    accepted=json.loads((parent/'phase_handoff.accepted.json').read_text())
    handoff=json.loads((parent/'phase_handoff.json').read_text())
    if (accepted['status']!='accepted' or accepted['phase_handoff_sha256']!=digest(parent/'phase_handoff.json')
            or handoff['successful_updates']!=2000 or (parent/'phase_failed.json').exists()):
        raise ValueError('parent phase is not a clean committed completed2000 phase')
    pins=production._validate_inputs(root)
    receipt=json.loads((parent/'checkpoint_2000.pt.receipt.json').read_text())
    online,_=load_committed_milestone(parent/'checkpoint_2000.pt',expected_update=2000,
        expected_pins={k:receipt['metadata'][k] for k in ['actor_sha256','manifest_sha256','READY_sha256','identity_sha256','protocol_sha256','code_sha256','budget_sha256']},
        expected_config_hash=pins['config'],model_factory=DuelingQ)
    payload=torch.load(parent/'checkpoint_2000.pt',map_location='cpu',weights_only=False)
    online.to(device).train();target=copy.deepcopy(online).eval()
    for p in target.parameters():p.requires_grad_(False)
    optimizer=optimizer_for(online)
    cache=TransitionCache(root/production.CACHE_REL)
    try:
        index=DemoReplayIndex(cache,gamma=.99997,n_step=10)
        learner=DQfDBatchLearner(online,target,optimizer,index,None,np.random.default_rng(20261024),
            demo_only=True,phase='authorized_demo_continuation',phase_identity=pins['manifest'])
        restore_payload(learner,payload,torch_module=torch)
        learner.model_metadata=dict(payload['metadata'])
        learner.model_metadata['continuation_parent_checkpoint_sha256']=authorization['parent_checkpoint_sha256']
        return learner
    except BaseException:
        cache.close();raise


def run_continuation(learner,*,output_dir,episode_fn,update_receipt_fn,
        checkpoint_receipt_fn=None,boundary_fn=None,clock=time.monotonic,started_at=None,
        execution_budget,checkpoint_fn=None):
    from .dqfd_pretraining import _save_milestone,learner_state_hashes
    if (execution_budget.get('study_id')!=STUDY_ID or learner.counters.successful_steps!=2000
            or learner.counters.attempted_steps!=2000 or not learner.demo_only or learner.store is not None):
        raise ValueError('invalid authorized demonstration continuation state')
    checkpoint_fn=checkpoint_fn or _save_milestone
    root=Path(output_dir)
    if root.exists():raise FileExistsError('continuation output occupied')
    root.mkdir(parents=True)
    start=clock() if started_at is None else started_at
    demo_seconds=0.;milestones=[];charged=0
    def guard(stage):
        elapsed=max(0.,clock()-start)
        if math.ceil(elapsed*10000000)>min(execution_budget['remaining_active_ticks'],execution_budget['slice_active_ticks']):
            raise TimeoutError('continuation cumulative/slice deadline')
        if elapsed-demo_seconds>execution_budget['model_io_active_seconds']:
            raise TimeoutError('continuation model-I/O deadline')
        if demo_seconds>execution_budget['demo_encoding_active_seconds']:
            raise TimeoutError('continuation demo deadline')
        if boundary_fn:boundary_fn(stage)
    for update in range(2001,5001):
        guard('optimizer_update_enter')
        row=learner.update(.6,receipt_fn=update_receipt_fn)
        if row['successful_step']!=update:raise ValueError('continuation update counter drift')
        guard('optimizer_update_leave')
        if update%1000:continue
        path=root/f'checkpoint_{update}.pt'
        checkpoint=checkpoint_fn(learner,path,update)
        if checkpoint_receipt_fn:checkpoint_receipt_fn({'update':update,**checkpoint})
        guard('checkpoint_readback')
        before=learner_state_hashes(learner);demo_start=clock()
        evidence=dict(episode_fn(update,path,'continuation_candidate'))
        demo_seconds+=max(0.,clock()-demo_start)
        if learner_state_hashes(learner)!=before:raise ValueError('continuation demo mutated learning state')
        evidence.update(update=update,label='continuation_candidate',checkpoint_sha256=checkpoint['checkpoint_sha256'])
        frames=(int(evidence.get('native_frames',0))+int(evidence.get('initialization_native_frames',0))
                +int(evidence.get('reset_provenance',{}).get('restoration_native_frames',0)))
        charged+=frames
        if frames>36000 or charged>108000:raise ValueError('continuation native budget exceeded')
        if (evidence.get('status')!='complete' or evidence.get('terminal_valid') is not True
                or evidence.get('runtime_identity_matched') is not True
                or evidence.get('reset_provenance_verified') is not True
                or not math.isfinite(float(evidence.get('cumulative_applied_reward',float('nan'))))
                or not 0<int(evidence.get('video_bytes',0))<=100*1024*1024):
            raise RuntimeError('continuation episode did not commit valid evidence')
        milestones.append(evidence);guard('demo_encoding_leave')
        if sum(int(e['video_bytes']) for e in milestones)>300*1024*1024:
            raise ValueError('continuation video budget exceeded')
    guard('finalization_enter')
    return {'successful_updates':5000,'attempted_updates':learner.counters.attempted_steps,
        'target_copies':learner.counters.target_copies,'milestones':milestones,'extra_best_demo':None,
        'fixed2000_retention_passed':False,'historical_fixed2000_retention_passed':False,
        'demo_gate_encoding_active_seconds':demo_seconds,'model_io_active_seconds':clock()-start-demo_seconds,
        'whole_supervised_active_seconds':clock()-start,'charged_demo_native_frames':charged,
        'episode_count':3,'state_hashes':learner_state_hashes(learner)}
