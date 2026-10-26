"""Single-policy backward PPO, with separately flagged human BC+PPO extensions."""
from _bootstrap import ROOT
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import shutil
import tempfile
import time
import zipfile
from types import SimpleNamespace

import numpy as np
import torch

from prince_valiant_rl.backward_level1 import OUT,HumanRoute,Level1Env,BackwardCurriculum,level1_child
from prince_valiant_rl.navigation import NavigationEncoder
from prince_valiant_rl.control_history import ControlHistory
from prince_valiant_rl.training.process_envs import ProcessEnvs,grouped_advantages,condition_critic,resident_bytes
from prince_valiant_rl.training.temporal_models import TemporalActorCritic,cuda_device,window_tensors,indexed_window_tensors,WINDOW_KEYS,masked_distribution,load_temporal,configure_precision,infer_shapes
from prince_valiant_rl.training.temporal import FrameHistory
from prince_valiant_rl.training.window_storage import compact_window,materialize_observation_batch
from prince_valiant_rl.training.models import parameter_hash
from prince_valiant_rl.training.checkpoint_io import atomic_torch_save
from prince_valiant_rl.training.evaluation_capture import (
    ArtifactBudgetGuard, ArtifactBudgetInterruption, BoundedJsonlWriter,
    DEFAULT_ARTIFACT_BUDGET_BYTES, DEFAULT_RESERVE_BYTES,
    DEFAULT_TEMP_OVERHEAD_BYTES, DiskGuard, ReferenceFingerprintReader,
    SERIALIZATION_VERSION, assess_storage, fingerprint_policy_input,
    fingerprints_equal, partial_capture_status, policy_input_schema,
    StorageGuardInterruption,
)

ACTIVE_POOL=None


def write_json(path,data):
    temporary=Path(path).with_suffix('.pending');temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)


def sha256_file(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def log(path,row,console=None):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    if console:print(console,flush=True)


def training_memory_sample(pool,phase):
    """Boundary-only RAM/commit/GPU sample for the memory-fixed runner."""
    workers=[dict(pid=int(pid),rss_bytes=resident_bytes(int(pid))) for pid in pool.pids]
    known=[row['rss_bytes'] for row in workers if row['rss_bytes'] is not None]
    system=None
    if os.name=='nt':
        import ctypes
        class MemoryStatusEx(ctypes.Structure):
            _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong),
                ('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),
                ('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),
                ('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual',ctypes.c_ulonglong)]
        value=MemoryStatusEx();value.dwLength=ctypes.sizeof(value)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(MemoryStatusEx)]
        kernel.GlobalMemoryStatusEx.restype=ctypes.c_int
        ok=kernel.GlobalMemoryStatusEx(ctypes.byref(value))
        if ok:
            system=dict(memory_load_percent=int(value.dwMemoryLoad),
                physical_total_bytes=int(value.ullTotalPhys),physical_available_bytes=int(value.ullAvailPhys),
                commit_limit_bytes=int(value.ullTotalPageFile),commit_available_bytes=int(value.ullAvailPageFile))
        else:system=dict(error=f'GlobalMemoryStatusEx failed: {ctypes.get_last_error()}')
    return dict(phase=phase,wall_time_unix=time.time(),runner_pid=os.getpid(),
        runner_rss_bytes=resident_bytes(os.getpid()),workers=workers,
        worker_rss_total_bytes=sum(known) if len(known)==len(workers) else None,
        worker_rss_available_count=len(known),system=system,
        gpu=dict(allocated_bytes=int(torch.cuda.memory_allocated()),
            reserved_bytes=int(torch.cuda.memory_reserved()),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved())))


def training_case(curriculum,rng,c,index):
    recovery_checkpoint=c.get('fixed_training_recovery_checkpoint')
    if recovery_checkpoint is not None:
        frame=int(c['fixed_training_anchor_frame'])
        return dict(frame=frame,horizon=int(c['full_horizon']),stage=curriculum.stage,
            frontier=frame,rehearsal=False,kind='fixed_recovery_snapshot',training_segment=True,
            episodic_life=False,recovery_checkpoint=str(recovery_checkpoint))
    if c.get('fixed_training_anchor_frame') is not None:
        frame=int(c['fixed_training_anchor_frame'])
        return dict(frame=frame,horizon=int(c['full_horizon']),stage=curriculum.stage,
            frontier=frame,rehearsal=False,kind='fixed_anchor',training_segment=True,
            episodic_life=bool(c.get('episodic_life_training',False)))
    roles=c.get('collector_roles')
    if roles is not None and len(roles)!=c['environments']:raise ValueError('One reset role per CPU owner required')
    case=curriculum.sample(rng,role=roles[index] if roles is not None else None)
    if c.get('episodic_life_training',False):case['episodic_life']=True
    return case


def apply_fixed_anchor_prefix(pool,indices,c,cases,states,episode_prefix_frames,
                              episode_prefix_rewards,prefix_frames,folder,update):
    """Advance configured scripted anchor actions outside the PPO rollout."""
    prefix=int(c.get('fixed_training_anchor_noop_prefix_frames',0))
    if not prefix:return states,prefix_frames
    if prefix!=1 or c.get('frame_skip')!=1:
        raise ValueError('The anchored PPO pilot requires exactly one native NOOP prefix frame')
    budget=int(c.get('fixed_training_prefix_frame_budget',0))
    if prefix_frames+len(indices)*prefix>budget:
        raise RuntimeError('Fixed-anchor scripted-prefix frame budget would be exceeded')
    if len(cases)!=len(indices) or any(int(case['frame'])!=int(c['fixed_training_anchor_frame'])
            or int(case['horizon'])!=int(c['full_horizon']) for case in cases):
        raise RuntimeError('Fixed-anchor reset case does not match the declared frame/horizon')
    for worker_id in indices:
        reset_info=states[worker_id]['info']
        if (int(reset_info.get('reset_frame',-1))!=int(c['fixed_training_anchor_frame'])
                or int(reset_info.get('reset_replay_frames',-1))!=0
                or reset_info.get('ram_sha256')!=c.get('fixed_training_anchor_ram_sha256')):
            raise RuntimeError('Fixed-anchor worker did not restore the verified zero-replay state')
    noop_index=list(c['action_masks']).index(0)
    results=pool.step_many([noop_index]*len(indices),indices)
    if len(results)!=len(indices):raise RuntimeError('Scripted-prefix results do not match reset workers')
    for worker_id,case,result in zip(indices,cases,results):
        info=result['info'];advanced=int(info.get('transition_frames',0))
        if (advanced!=prefix or result.get('terminated') or result.get('truncated')
                or info.get('end_reason') is not None
                or info.get('elapsed_frames')!=prefix
                or info.get('reset_frame')!=int(c['fixed_training_anchor_frame'])):
            raise RuntimeError('Fixed-anchor prefix crossed a boundary or failed native-frame accounting')
        if int(info.get('reset_replay_frames',-1))!=0:
            raise RuntimeError('Fixed-anchor reset replay advanced native frames outside the prefix budget')
        record=dict(update=int(update),worker_id=int(worker_id),reset_frame=int(case['frame']),
            horizon_native_frames_including_prefix=int(case['horizon']),
            prefix_control_action_index=noop_index,prefix_control_mask=0,
            prefix_native_frames=advanced,episode_elapsed_after_prefix=int(info['elapsed_frames']),
            episode_end_reason=info.get('end_reason'),ram_sha256=info.get('ram_sha256'),
            reset_replay_frames=int(info['reset_replay_frames']),
            observer_validity=info.get('combat_observer_diagnostics',{}).get('status','unavailable'),
            score_validity=info.get('score_gain_coverage',{}).get('status','unavailable'))
        log(folder/'anchor_prefixes.jsonl',record)
        states[worker_id]=result
        episode_prefix_frames[worker_id]=advanced
        episode_prefix_rewards[worker_id]=float(result.get('reward',0.))
        prefix_frames+=advanced
    return states,prefix_frames


def empty_combat_scope_collection(worker_count):
    return [dict(worker_id=i,native_frames=0,supported_native_frames=0,unsupported_native_frames=0,
        observer_available_native_frames=0,observer_unavailable_native_frames=0,
        verified_kind18_defeats=0,unverified_kind18_defeats=0,unverified_kind18_records=[],
        abi2_kind18_defeat_reward=0.) for i in range(worker_count)]


def accumulate_combat_scope_transition(collection,worker_id,info):
    """Accumulate one worker decision, including unfinished episode steps."""
    row=collection[worker_id]
    transition_frames=int(info['transition_frames'])
    coverage=info['combat_reward_coverage']['abi2_kind18_defeat']
    supported=int(coverage['supported_native_frames'])
    unsupported=int(coverage['unavailable_native_frames'])
    if supported<0 or unsupported<0 or supported+unsupported!=transition_frames:
        raise RuntimeError('Combat reward scope frame telemetry does not reconcile with transition duration')
    diagnostics=info['combat_observer_diagnostics']
    status=diagnostics['status']
    if status=='available':
        observed_frames=int(diagnostics['decision_native_frames'])
        if observed_frames!=transition_frames:
            raise RuntimeError('ABI2 observer frame telemetry does not reconcile with transition duration')
        defeats=diagnostics['counts']['by_enemy_kind']['18']['defeats']
        if isinstance(defeats,bool) or not isinstance(defeats,int):
            raise RuntimeError('ABI2 kind18 defeat count is missing or malformed')
        unverified=diagnostics['counts']['unverified']['by_enemy_kind']['18']['defeats']
        if isinstance(unverified,bool) or not isinstance(unverified,int) or unverified<0:
            raise RuntimeError('ABI2 unverified kind18 defeat count is missing or malformed')
        if int(info['events'].get('abi2_kind18_defeat',0))!=defeats:
            raise RuntimeError('ABI2 kind18 defeat count differs from the reward event total')
        row['observer_available_native_frames']+=observed_frames
        row['verified_kind18_defeats']+=defeats
        row['unverified_kind18_defeats']+=unverified
        if unverified:
            row['unverified_kind18_records'].extend(
                dict(record) for record in diagnostics.get('records', [])
                if record.get('event_type')==2 and record.get('enemy_kind')==18)
    elif status=='disabled':
        row['observer_unavailable_native_frames']+=transition_frames
    else:
        raise RuntimeError(f'ABI2 combat observer status is not usable: {status}')
    row['native_frames']+=transition_frames
    row['supported_native_frames']+=supported
    row['unsupported_native_frames']+=unsupported
    row['abi2_kind18_defeat_reward']+=float(info['reward_components'].get('abi2_kind18_defeat',0.))


def summarize_combat_scope_collection(collection):
    workers=[]
    for source in collection:
        row=dict(source)
        row['verified_kind18_defeats']=(row['verified_kind18_defeats']
            if row['observer_unavailable_native_frames']==0
            and row['observer_available_native_frames']==row['native_frames'] else None)
        workers.append(row)
    native=sum(row['native_frames'] for row in workers)
    unavailable_observer=sum(row['observer_unavailable_native_frames'] for row in workers)
    observer_available=sum(row['observer_available_native_frames'] for row in workers)
    defeats_available=unavailable_observer==0 and observer_available==native
    return dict(unit='native_frames',reward_unit='undiscounted_reward_component',native_frames=native,
        supported_native_frames=sum(row['supported_native_frames'] for row in workers),
        unsupported_native_frames=sum(row['unsupported_native_frames'] for row in workers),
        observer_available_native_frames=observer_available,
        observer_unavailable_native_frames=unavailable_observer,
        observed_kind18_defeats=(sum(row['verified_kind18_defeats']+row['unverified_kind18_defeats'] for row in workers)
            if defeats_available else None),
        verified_kind18_defeats=(sum(row['verified_kind18_defeats'] for row in workers)
            if defeats_available else None),
        unverified_kind18_defeats=(sum(row['unverified_kind18_defeats'] for row in workers)
            if defeats_available else None),
        verified_kind18_defeat_count_available=defeats_available,
        abi2_kind18_defeat_reward=sum(row['abi2_kind18_defeat_reward'] for row in workers),
        by_worker=workers)


def empty_score_scope_collection(worker_count):
    return [dict(worker_id=i,native_frames=0,available_native_frames=0,unavailable_native_frames=0,
        quarantined_native_frames=0,quarantined_episodes=0,zero_score_reward_native_frames=0,
        measured_zero_native_frames=0,positive_displayed_score_points=0,score_reward=0.) for i in range(worker_count)]


def accumulate_score_scope_transition(collection,worker_id,info):
    row=collection[worker_id];native=int(info['transition_frames'])
    coverage=info.get('score_gain_coverage',{});status=coverage.get('status','unavailable')
    if status=='available':
        decoded=int(coverage.get('last_decision_valid_native_frames',coverage.get('last_decision_native_frames',0)))
        quarantined=int(coverage.get('last_decision_quarantined_native_frames',0))
        if decoded+quarantined!=native:
            raise RuntimeError('Displayed-score frame telemetry does not reconcile with transition duration')
        points=coverage.get('last_decision_valid_score_points',coverage.get('last_decision_score_points'))
        if isinstance(points,bool) or not isinstance(points,int) or points<0:
            raise RuntimeError('Displayed-score transition points are missing or malformed')
        row['available_native_frames']+=decoded
        row['positive_displayed_score_points']+=points
        row['quarantined_native_frames']+=quarantined
    elif status=='quarantined':
        decoded=int(coverage.get('last_decision_valid_native_frames',0))
        quarantined=int(coverage.get('last_decision_quarantined_native_frames',0))
        if decoded+quarantined!=native or quarantined<=0:
            raise RuntimeError('Quarantined displayed-score frame telemetry does not reconcile with transition duration')
        points=coverage.get('last_decision_valid_score_points')
        if isinstance(points,bool) or not isinstance(points,int) or points<0:
            raise RuntimeError('Valid pre-quarantine score points are missing or malformed')
        row['available_native_frames']+=decoded
        row['unavailable_native_frames']+=quarantined
        row['quarantined_native_frames']+=quarantined
        row['positive_displayed_score_points']+=points
        row['quarantined_episodes']+=int(coverage.get('quarantine_started_this_decision',False))
    elif status in ('disabled','failed','unavailable'):
        row['unavailable_native_frames']+=native
    else:
        raise RuntimeError(f'Displayed-score tracking status is not usable: {status}')
    row['zero_score_reward_native_frames']+=int(coverage.get('last_decision_zero_reward_native_frames',0))
    row['measured_zero_native_frames']+=int(coverage.get('last_decision_measured_zero_native_frames',0))
    row['native_frames']+=native
    row['score_reward']+=float(info.get('reward_components',{}).get('score_gain',0.))


def summarize_score_scope_collection(collection):
    workers=[dict(row) for row in collection]
    native=sum(row['native_frames'] for row in workers)
    available=sum(row['available_native_frames'] for row in workers)
    unavailable=sum(row['unavailable_native_frames'] for row in workers)
    valid=unavailable==0 and available==native
    return dict(unit='native_frames',native_frames=native,available_native_frames=available,
        unavailable_native_frames=unavailable,valid=valid,
        quarantined_native_frames=sum(row['quarantined_native_frames'] for row in workers),
        quarantined_episodes=sum(row['quarantined_episodes'] for row in workers),
        zero_score_reward_native_frames=sum(row['zero_score_reward_native_frames'] for row in workers),
        measured_zero_native_frames=sum(row['measured_zero_native_frames'] for row in workers),
        positive_displayed_score_points=(sum(row['positive_displayed_score_points'] for row in workers) if valid else None),
        score_reward=sum(row['score_reward'] for row in workers),by_worker=workers)


def reset_resumed_worker_state(prior,worker_count,collector_roles,source,route_start):
    """Fork rollout streams while retaining checkpoint learning/curriculum state."""
    saved_states=prior.get('environment_states',[])
    if len(saved_states)!=worker_count:
        raise ValueError('Worker reset requires one saved source snapshot per configured worker')
    if len(collector_roles)!=worker_count or any(role!='full_start' for role in collector_roles):
        raise ValueError('Worker reset requires every collector role to be full_start')
    result=dict(prior,environment_states=[],cases=[],episode_returns={},recent=[],
        matched_worker_reset=dict(source=str(source),saved_worker_snapshots_discarded=len(saved_states),
            collector_roles=list(collector_roles),reset_frame=int(route_start),
            model_optimizer_rng_curriculum_preserved=True,episode_returns_and_recent_window_cleared=True))
    return result


def reset_collector_role_worker_state(prior,worker_count,collector_roles,source):
    """Discard saved live episodes while preserving all learner/curriculum state."""
    saved_states=prior.get('environment_states',[])
    saved_cases=prior.get('cases',[])
    if len(saved_states)!=worker_count or len(saved_cases)!=worker_count:
        raise ValueError('Collector-role adaptation requires one saved snapshot and case per worker')
    if len(collector_roles)!=worker_count or any(role not in ('frontier','coverage','full_start') for role in collector_roles):
        raise ValueError('Collector-role adaptation requires one supported role per worker')
    discarded=Counter(case.get('kind','legacy') for case in saved_cases)
    result=dict(prior,environment_states=[],cases=[],episode_returns={},recent=[])
    result['matched_worker_reset']=dict(source=str(source),
        saved_worker_snapshots_discarded=len(saved_states),saved_live_cases_discarded=len(saved_cases),
        discarded_case_kind_counts=dict(discarded),collector_roles=list(collector_roles),
        reset_anchor=int(prior['curriculum']['frontier']),
        saved_episode_boundaries_preserved=True,old_episodes_relabelled=False,
        new_episodes_assigned_configured_roles=True,
        model_optimizer_rng_counters_and_curriculum_preserved=True,
        curriculum_outcomes_preserved=True,
        score_tracking='reseed each new worker baseline from its reset RAM; no retroactive score reward')
    return result


def is_frontier_mastery_transition(transition):
    """A mastery/refinement event that moves collection to an earlier frontier."""
    return (transition.get('kind') in ('advance_earlier','refine_easier')
        and int(transition.get('new',0))<int(transition.get('old',0)))


def reset_record(worker_id,case,result,frames,update):
    info=result['info'];score=info.get('score_gain_coverage',{})
    observer=info.get('combat_observer_diagnostics',{})
    return dict(frames=int(frames),update=int(update),worker_id=int(worker_id),case=dict(case),
        score_validity=score.get('status','unavailable'),score_baseline_points=score.get('current_score_points'),
        observer_validity=observer.get('status','unavailable'))


def entropy_weight(c,frames):
    schedule=c.get('entropy_schedule')
    if schedule is None:return float(c['entropy_coefficient'])
    fraction=min(1.,max(0.,frames/float(schedule['frames'])))
    return float(schedule['start']+(schedule['end']-schedule['start'])*fraction)


def critic_warmup_rollout(start_frames,initial_until,rewarmup_until):
    # A rollout crossing the boundary must finish critic-only. Checking its
    # ending counter silently rounded requested warmup DOWN by one rollout.
    return start_frames<max(initial_until,rewarmup_until)


def training_budget_targets(start_frames,start_updates,frame_budget,update_budget):
    """Return one exact cumulative stop counter for a training block."""
    if (frame_budget is None)==(update_budget is None):
        raise ValueError('Specify exactly one of additional frames or outer PPO updates')
    if frame_budget is not None:
        if frame_budget<0:raise ValueError('Frame budget cannot be negative')
        return int(start_frames)+int(frame_budget),None
    if update_budget<1:raise ValueError('Outer PPO update budget must be positive')
    return None,int(start_updates)+int(update_budget)


CURRICULUM_ADAPTATION_FIELDS={
    'coverage_reset_frames','advancement_threshold','fine_step_regions','collector_roles',
    'bc_auxiliary_schedule','bc_focus_sampling','actor_learning_rate',
    'initial_failure_fallback_starts','initial_fallback_min_frames'}
SCORE_REWARD_ADAPTATION_PATHS={'weights.score_gain'}
INITIAL_FALLBACK_FIELDS={'initial_failure_fallback_starts','initial_fallback_min_frames'}


def configuration_diff_paths(old,new,prefix=''):
    """Return the exact changed config paths, including nested reward fields."""
    if isinstance(old,dict) and isinstance(new,dict):
        paths=set()
        for key in set(old)|set(new):
            path=f'{prefix}.{key}' if prefix else str(key)
            if key not in old or key not in new:
                paths.add(path)
            else:
                paths.update(configuration_diff_paths(old[key],new[key],path))
        return paths
    if type(old) is not type(new):return {prefix}
    try:
        equal=old==new
        if isinstance(equal,np.ndarray):equal=bool(np.array_equal(old,new))
        else:equal=bool(equal)
    except (TypeError,ValueError):
        equal=False
    return set() if equal else {prefix}


def configuration_changed_keys(old,new):
    return {path.split('.',1)[0] for path in configuration_diff_paths(old,new)}


def validate_curriculum_adaptation(old,new,changes=None,route=None):
    actual=configuration_changed_keys(old,new)
    differences=actual if changes is None else set(changes)
    if not differences or not differences<=CURRICULUM_ADAPTATION_FIELDS or not differences<=actual:
        raise ValueError('Curriculum adaptation may change only reset anchors/roles, advancement settings, or the declared BC-retention schedule')
    # Explicit fork-only settings: reset schedule, BC replay sampling/retention,
    # and actor step size. Reward, observations, actions, model, and PPO dynamics
    # remain fixed so each adaptation can be attributed to its declared change.
    if 'advancement_threshold' in differences and not 0.0 < float(new['advancement_threshold']) <= 1.0:
        raise ValueError('Curriculum advancement threshold must be in (0,1]')
    if 'initial_failure_fallback_starts' in differences:
        starts=new['initial_failure_fallback_starts']
        if not isinstance(starts,list) or any(isinstance(x,bool) or not isinstance(x,int) for x in starts):
            raise ValueError('Initial fallback starts must be an explicit list of integer verified route frames')
        if len(set(starts))!=len(starts) or any(x<0 for x in starts):
            raise ValueError('Initial fallback starts must be unique nonnegative route frames')
    if 'initial_fallback_min_frames' in differences:
        frames=new['initial_fallback_min_frames']
        if isinstance(frames,bool) or not isinstance(frames,int) or frames<0:
            raise ValueError('Initial fallback native-frame threshold must be a nonnegative integer')
    if 'fine_step_regions' in differences:
        for region in new['fine_step_regions']:
            if len(region)!=3 or int(region[0])>=int(region[1]) or int(region[2])<=0:
                raise ValueError('Backward step regions must be [low, high, positive_step]')
    if 'collector_roles' in differences:
        roles=new['collector_roles']
        if len(roles)!=new['environments'] or any(r not in ('frontier','coverage','full_start') for r in roles):
            raise ValueError('Collector roles must match environment count and use supported reset kinds')
    if 'bc_auxiliary_schedule' in differences:
        schedule=new['bc_auxiliary_schedule']
        if (set(schedule)!={'start','end','frames'} or
                not 0.0<=float(schedule['start'])<=1.0 or
                not 0.0<=float(schedule['end'])<=1.0 or int(schedule['frames'])<=0):
            raise ValueError('BC retention schedule must use bounded start/end weights and a positive frame horizon')
    if 'bc_focus_sampling' in differences:
        focus=new['bc_focus_sampling']
        if not 0.0<=float(focus.get('fraction',-1))<=0.75 or not focus.get('ranges'):
            raise ValueError('Invalid targeted BC focus sampling configuration')
        if any(len(r)!=2 or int(r[0])<0 or int(r[1])<=int(r[0]) for r in focus['ranges']):
            raise ValueError('BC focus ranges must be nonempty half-open frame intervals')
    if 'actor_learning_rate' in differences and float(new['actor_learning_rate'])<=0:
        raise ValueError('Actor learning rate must be positive')
    if route is not None and differences&INITIAL_FALLBACK_FIELDS:
        candidates=route.valid[route.valid<=route.end-new['initial_remaining_frames']]
        if not len(candidates):raise ValueError('No valid initial curriculum frontier')
        frontier=int(candidates[-1])
        upper=min(route.end-1,int(new.get('latest_action_dependent_reset',route.end-1)))
        for frame in new.get('initial_failure_fallback_starts',[]):
            if frame<=frontier or frame>upper or frame not in route.valid:
                raise ValueError('Initial fallback start is outside the eligible verified route suffix')
    return differences


def validate_timing_adaptation(old,new):
    differences={k for k in set(new)|set(old) if new.get(k)!=old.get(k)}
    if differences!={'frame_skip'}:
        raise ValueError('Timing adaptation may change only the held native-frame interval')
    if not isinstance(new['frame_skip'],int) or not 1<=new['frame_skip']<=8:
        raise ValueError('Timing comparison requires an integer interval from1to8 native frames')


def validate_normalization_adaptation(old,new):
    from prince_valiant_rl.training.temporal_models import BRANCH_NORM_ARCHITECTURE
    differences={k for k in set(new)|set(old) if new.get(k)!=old.get(k)}
    if old.get('branch_feature_normalization',False) or not new.get('branch_feature_normalization',False):
        raise ValueError('Only explicit baseline -> branch normalization supported')
    if differences-{'evaluation_suffix_trials'}!={'architecture','branch_feature_normalization'} or new['architecture']!=BRANCH_NORM_ARCHITECTURE:
        raise ValueError('Only architecture/branch normalization flags may change in this fork')


def rng_state(rng):
    return dict(numpy=rng.bit_generator.state,python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def load_rng(rng,state):
    rng.bit_generator.state=state['numpy'];random.setstate(state['python']);torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def validate_reward_adaptation(old,new):
    """Allow only the explicitly labelled, additive, tested potential variant."""
    from prince_valiant_rl.level1_potential import VERSION
    if old.get('final_encounter_potential',False) or not new.get('final_encounter_potential',False):
        raise ValueError('Reward adaptation requires baseline -> potential variant')
    if new.get('reward_version')!=VERSION or old.get('reward_version','level1_task_v1')==VERSION:
        raise ValueError('Explicit new reward version required')
    if new['gamma']!=old['gamma'] or new['frame_skip']!=old['frame_skip']:
        raise ValueError('Hold timing and discount fixed during reward adaptation')
    if set(new['weights'])!=set(old['weights'])|{'combat_potential'}:
        raise ValueError('Only combat_potential may be added')
    if any(new['weights'][k]!=v for k,v in old['weights'].items()):
        raise ValueError('Existing reward coefficients must remain unchanged')
    if not 0<new['weights']['combat_potential']<=8:
        raise ValueError('Potential coefficient outside documented bounded range')


def validate_score_reward_adaptation(old,new,changes=None):
    """Permit only the tested zero -> configured displayed-score activation."""
    actual=configuration_diff_paths(old,new)
    differences=actual if changes is None else set(changes)
    if changes is None:
        if 'weights.score_gain' not in actual or any(not path.startswith('weights.') for path in actual):
            raise ValueError('Score reward adaptation may change only weights.score_gain')
    elif differences!={'weights.score_gain'} or not differences<=actual:
        raise ValueError('Score reward adaptation may change only weights.score_gain')
    old_weights=old.get('weights',{});new_weights=new.get('weights',{})
    if set(old_weights)!=set(new_weights) or any(
            old_weights[k]!=new_weights[k] for k in old_weights if k!='score_gain'):
        raise ValueError('Score reward adaptation must preserve every other reward weight')
    if old_weights.get('score_gain',0.)!=0. or new_weights.get('score_gain')!=0.00005:
        raise ValueError('Score reward adaptation permits only score_gain 0 -> 0.00005')
    return differences


EVALUATION_ONLY_TRAINING_METADATA_PATHS=frozenset({
    'bc_auxiliary_data','bc_auxiliary_expected_rows','bc_auxiliary_format',
    'bc_auxiliary_schedule.start','bc_auxiliary_sha256','bc_enabled','bc_extra_ratio',
    'bc_sampling_method','curriculum_mode','disable_curriculum_advancement','experiment',
    'fixed_training_anchor_frame','fixed_training_anchor_noop_prefix_frames',
    'fixed_training_anchor_ram_sha256','fixed_training_prefix_frame_budget',
    'frontier_jitter_frames','full_horizon','seed',
})


def validate_adaptation_composition(old,new,*,curriculum=False,score_reward=False,route=None,
        allow_evaluation_seed=False,allow_evaluation_fallback_metadata=False,
        allow_evaluation_collector_roles=False,allow_evaluation_no_stall_limit=False,
        allow_evaluation_training_metadata=False):
    """Validate the exact source -> target diff against the explicitly enabled union."""
    paths=configuration_diff_paths(old,new)
    changed_keys={path.split('.',1)[0] for path in paths}
    ignored_evaluation_paths=(paths&INITIAL_FALLBACK_FIELDS
        if allow_evaluation_fallback_metadata else set())
    if allow_evaluation_collector_roles:
        ignored_evaluation_paths|=paths&{'collector_roles'}
    if allow_evaluation_no_stall_limit:
        if int(new.get('development_stall_frames',-1))!=int(new.get('full_horizon',0))+1:
            raise ValueError('Evaluation no-stall override must exceed the configured full horizon')
        ignored_evaluation_paths|=paths&{'development_stall_frames'}
    if allow_evaluation_training_metadata:
        ignored_evaluation_paths|=paths&EVALUATION_ONLY_TRAINING_METADATA_PATHS
    ignored_evaluation_keys={path.split('.',1)[0] for path in ignored_evaluation_paths}
    curriculum_changes=(changed_keys&CURRICULUM_ADAPTATION_FIELDS)-ignored_evaluation_keys
    score_changes=paths&SCORE_REWARD_ADAPTATION_PATHS
    allowed_paths=set()
    if curriculum:
        allowed_paths.update(path for path in paths
            if path.split('.',1)[0] in CURRICULUM_ADAPTATION_FIELDS)
    if score_reward:allowed_paths.update(SCORE_REWARD_ADAPTATION_PATHS)
    if allow_evaluation_seed:allowed_paths.add('evaluation_seed')
    unexpected=paths-allowed_paths-ignored_evaluation_paths
    if unexpected:
        raise ValueError('Adaptation flags do not authorize config changes: '+repr(sorted(unexpected)))
    if curriculum:
        validate_curriculum_adaptation(old,new,curriculum_changes,route=route)
    if score_reward:
        validate_score_reward_adaptation(old,new,score_changes)
    if not curriculum and curriculum_changes:
        raise ValueError('Curriculum config changes require --curriculum-adaptation')
    if not score_reward and any(path.startswith('weights.score_gain') for path in paths):
        raise ValueError('Score reward config changes require --score-reward-adaptation')
    return dict(changes=paths-ignored_evaluation_paths,curriculum_changes=curriculum_changes,
        score_changes=score_changes,ignored_evaluation_metadata=ignored_evaluation_paths)


def validate_score_curriculum_adaptation(old,new,route=None):
    """Backward-compatible entry point for the explicit combined adaptation."""
    plan=validate_adaptation_composition(old,new,curriculum=True,score_reward=True,route=route)
    return plan['changes']


def validate_scoped_kind18_reward_resume(old,new,allow_evaluation_seed=False,
        allow_collector_roles=False):
    """Permit observer/reward changes and only explicit supported role overrides."""
    differences={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    allowed={'combat_observer_diagnostics','weights'}|({'evaluation_seed'} if allow_evaluation_seed else set())
    if allow_collector_roles:allowed.add('collector_roles')
    if differences-allowed:
        raise ValueError('Scoped kind18 reward resume may change only observer activation and its weight: '+repr(sorted(differences)))
    if new.get('combat_observer_diagnostics') is not True:
        raise ValueError('Scoped kind18 reward requires the ABI2 observer in every arm')
    if allow_collector_roles:
        roles=new.get('collector_roles',[])
        if len(roles)!=new.get('environments') or any(role!='full_start' for role in roles):
            raise ValueError('Explicit worker reset requires every collector role to be full_start')
    old_weights=old.get('weights',{});new_weights=new.get('weights',{})
    if (set(old_weights)-{'abi2_kind18_defeat'})!=(set(new_weights)-{'abi2_kind18_defeat'}):
        raise ValueError('Scoped kind18 reward resume may add only weights.abi2_kind18_defeat')
    if any(old_weights[k]!=new_weights[k] for k in set(old_weights)-{'abi2_kind18_defeat'}):
        raise ValueError('Existing reward coefficients must remain unchanged')
    old_weight=float(old_weights.get('abi2_kind18_defeat',0.))
    new_weight=float(new_weights.get('abi2_kind18_defeat',0.))
    if old_weight not in (0.,.25) or new_weight not in (0.,.25):
        raise ValueError('Matched pilot permits only scoped kind18 weights 0 or 0.25')


def validate_navigation_reward_adaptation(old,new):
    """Only the approved three coefficient doublings, with explicit versions."""
    from prince_valiant_rl.backward_level1 import NAVIGATION_REWARD_VERSION
    from prince_valiant_rl.level1_potential import VERSION
    if old.get('reward_version')!=VERSION or new.get('reward_version')!=NAVIGATION_REWARD_VERSION:
        raise ValueError('Explicit original -> navigation reward-v2 fork required')
    if new.get('potential_version')!=VERSION or not old.get('final_encounter_potential'):
        raise ValueError('Preserve the verified potential implementation')
    differences={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    if differences!={'weights','reward_version','potential_version'}:
        raise ValueError('Reward-v2 fork may not change timing, observations, curriculum or optimization')
    expected=dict(old['weights'])
    for key in ('new_progress','checkpoint_reached','level_completed'):expected[key]*=2
    if new['weights']!=expected:
        raise ValueError('Only the three approved reward doublings are permitted')


def validate_time_cost_adaptation(old,new):
    """Explicit reward-v2 -> requested -.03/sec; all other settings fixed."""
    if old.get('reward_version')!='level1_progress_milestones_v2' or new.get('reward_version')!='level1_time_cost_003_v3':
        raise ValueError('Explicit reward-v2 -> time-cost003 fork required')
    differences={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    if differences!={'weights','reward_version'}:
        raise ValueError('Time-cost fork must preserve all other task/training settings')
    if old['weights'].get('elapsed_seconds')!=-.01:
        raise ValueError('Expected preserved -.01/sec source')
    expected=dict(old['weights'],elapsed_seconds=-.03)
    if new['weights']!=expected:
        raise ValueError('Only elapsed_seconds=-.03 may change')


def validate_forward_progress_adaptation(old,new):
    """Double first-time forward progress only, keeping -.03/sec and timing."""
    if old.get('reward_version')!='level1_time_cost_003_v3' or new.get('reward_version')!='level1_forward020_v4':
        raise ValueError('Explicit time-cost003 -> forward020 fork required')
    differences={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
    if differences!={'weights','reward_version'}:
        raise ValueError('Forward-progress fork must preserve all other settings')
    if old['weights'].get('new_progress')!=.1 or old['weights'].get('elapsed_seconds')!=-.03:
        raise ValueError('Expected preserved .10 progress and -.03/sec source')
    if new['weights']!=dict(old['weights'],new_progress=.2):
        raise ValueError('Only new_progress=.20 may change')


def optional_training_counter(metadata,key):
    """Read a persisted training counter without treating absence as recorded zero."""
    if key not in metadata:return None
    value=metadata[key]
    if isinstance(value,bool) or not isinstance(value,int) or value<0:
        raise ValueError(f'Checkpoint training counter {key!r} must be a nonnegative integer')
    return value


def validate_evaluation_identifier(value):
    if value is None:return None
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',value):
        raise ValueError('Evaluation identifier must be 1-80 ASCII letters, digits, dots, underscores, or hyphens and start alphanumeric')
    return value


def snapshots(pool):
    for i in range(len(pool.parents)):pool.send(i,'snapshot',None)
    return [pool.receive(i) for i in range(len(pool.parents))]


def adaptation_validation_result(model,prior,config,adaptation_plan):
    """Describe the planned state transition without constructing an optimizer or workers."""
    from prince_valiant_rl.game_state import decode_display_score
    optimizer_state=prior.get('optimizer',{})
    groups=optimizer_state.get('param_groups',[])
    saved_state=optimizer_state.get('state',{})
    actor_ids=set(groups[0].get('params',[])) if groups else set()
    critic_ids=set(groups[1].get('params',[])) if len(groups)>1 else set()
    actor_state_count=sum(parameter in saved_state for parameter in actor_ids)
    critic_state_count=sum(parameter in saved_state for parameter in critic_ids)
    frames=int(prior['frames'])
    score_adaptation=bool(adaptation_plan['score_changes'])
    rewarmup_until=(max(int(prior.get('critic_rewarmup_until',0)),
        frames+int(config.get('critic_rewarmup_frames',16384))) if score_adaptation
        else int(prior.get('critic_rewarmup_until',0)))
    score_scores=[decode_display_score(snapshot['fields']['ram'])
        for snapshot in prior.get('environment_states',[])]
    curriculum_plan=prior.get('curriculum_adaptation',{})
    score_plan=prior.get('score_reward_adaptation',{})
    curriculum_state=prior.get('curriculum',{})
    adaptation=prior.get('curriculum_adaptation',{})
    reset=prior.get('matched_worker_reset',{})
    expected_critic_states=0 if score_adaptation else critic_state_count
    return dict(stage='adaptation_validation',status='passed',
        workers_started=False,optimizer_created=False,optimizer_state_application='not_applied',
        optimizer_steps=0,emulator_frames=0,
        changes=sorted(adaptation_plan['changes']),
        curriculum_changes=sorted(adaptation_plan['curriculum_changes']),
        score_changes=sorted(adaptation_plan['score_changes']),
        frames=frames,updates=int(prior['updates']),decisions=int(prior['decisions']),
        actor_optimizer_steps=int(prior['actor_optimizer_steps']),
        actor_hash=parameter_hash(model.actor),
        actor_parameters_preserved=True,
        actor_optimizer_state_count=actor_state_count,
        critic_optimizer_state_count_before_reset=critic_state_count,
        expected_critic_optimizer_state_count_after_reset=expected_critic_states,
        actor_optimizer_state_preserved=(bool(score_plan.get('actor_optimizer_preserved'))
            if score_adaptation else True),
        critic_optimizer_reset_planned=score_adaptation,
        optimizer_state_preserved=not score_adaptation,
        critic_rewarmup_until=rewarmup_until,critic_rewarmup_frames=max(0,rewarmup_until-frames),
        worker_snapshot_count=len(prior.get('environment_states',[])),
        worker_snapshots_discarded=int(reset.get('saved_worker_snapshots_discarded',0)),
        worker_snapshots_preserved=(bool(score_plan.get('worker_snapshots_preserved'))
            if score_adaptation else not bool(reset)),
        score_baselines_seed_from_restored_ram=(bool(score_scores) and all(score is not None for score in score_scores)
            and not bool(reset)),
        score_baselines_reseeded_from_new_reset_ram=bool(reset),
        decoded_saved_snapshot_scores=score_scores,score_points_before_adaptation=0,
        retroactive_score_reward=False,score_tracking_preserved=True,
        rng_and_counters_preserved=(bool(score_plan.get('rng_and_counters_preserved'))
            if score_adaptation else True),
        curriculum_failure_history_preserved=(bool(curriculum_plan.get('failure_history_preserved'))
            or bool(reset.get('curriculum_outcomes_preserved'))),
        curriculum_failure_frames=int(curriculum_state.get('initial_failure_frames',0)),
        curriculum_outcomes_preserved=len(curriculum_state.get('outcomes',[])),
        curriculum_outcomes_preserved_by_adaptation=bool(reset.get('curriculum_outcomes_preserved')),
        discarded_live_case_kinds=reset.get('discarded_case_kind_counts',{}),
        curriculum=curriculum_state)


def validate_checkpoint_experiment_provenance(metadata,checkpoint_config,evaluation_config,route,*,stage,
        allow_evaluation_provenance_difference=False):
    """Validate run labels here; the evaluation-only model contract checks runtime compatibility."""
    training_experiment=metadata.get('experiment',checkpoint_config.get('experiment','backward_level1_v1'))
    model_config_experiment=checkpoint_config.get('experiment','backward_level1_v1')
    evaluation_config_experiment=evaluation_config.get('experiment','backward_level1_v1')
    if metadata.get('route_sha256')!=route.meta.get('route_sha256'):
        raise ValueError('Checkpoint demonstration route identity differs from the evaluation route')
    if stage!='evaluate':
        if allow_evaluation_provenance_difference:
            raise ValueError('--allow-evaluation-provenance-difference is evaluation-only')
        if training_experiment!=evaluation_config_experiment:
            raise ValueError('Training resume experiment identity differs from its configured experiment')
    elif training_experiment!=evaluation_config_experiment:
        if not allow_evaluation_provenance_difference:
            raise ValueError('Checkpoint training experiment differs from evaluation config; explicit evaluation-only provenance allowance required')
    # With the explicit evaluation-only allowance, experiment strings remain
    # provenance labels. The caller immediately validates the complete model
    # and route observation contract before preparing or starting evaluation.
    model_config_identity_matches=(model_config_experiment==evaluation_config_experiment)
    ignored_training_metadata=(configuration_diff_paths(checkpoint_config,evaluation_config)
        & EVALUATION_ONLY_TRAINING_METADATA_PATHS if allow_evaluation_provenance_difference else set())
    return dict(training_experiment_identity=training_experiment,
        checkpoint_model_config_experiment=model_config_experiment,
        evaluation_config_experiment=evaluation_config_experiment,
        identity_only_difference=bool(training_experiment!=evaluation_config_experiment
            and model_config_identity_matches),
        checkpoint_model_config_identity_matches_evaluation_config=model_config_identity_matches,
        evaluation_only_training_metadata_paths_ignored=sorted(ignored_training_metadata),
        full_runtime_contract_validation_required=bool(allow_evaluation_provenance_difference),
        evaluation_only_allowance_used=bool(allow_evaluation_provenance_difference))


def checkpoint_route_observation_window(config,route,frame):
    """Encode the real verified route history without creating an emulator or worker."""
    offsets=tuple(config['history_offsets'])
    earliest=max(route.start,int(frame)+offsets[0])
    controls=(ControlHistory.from_recording(route.buttons,earliest,route.start,
        config.get('control_age_cap_frames',120)) if config.get('control_edge_features',False) else None)
    encoder=NavigationEncoder(config.get('feature_groups'),config.get('observation_version','legacy_v3'))
    context=SimpleNamespace(config=config,encoder=encoder,controls=controls)
    history=FrameHistory(offsets)
    for index,route_frame in enumerate(range(earliest,int(frame)+1)):
        if index and controls is not None:
            controls.push(int(route.buttons[route_frame-1]))
        has_previous=route_frame>route.start
        previous=route.ram[route_frame-1] if has_previous else None
        last_action=int(route.buttons[route_frame-1]) if has_previous else 0
        observation=Level1Env.encode(context,route.ram[route_frame],previous,route.tracker(route_frame),
            last_action,has_previous,controls=controls)
        if index==0:history.reset(observation)
        else:history.push(index,observation)
    return history.window()


def validate_evaluation_checkpoint_contract(model,checkpoint_config,evaluation_config,route,frame):
    """Validate the loaded actor against the live evaluator's full input contract."""
    from prince_valiant_rl.training.temporal_models import ARCHITECTURE,BRANCH_NORM_ARCHITECTURE
    expected_architecture=(BRANCH_NORM_ARCHITECTURE if evaluation_config.get('branch_feature_normalization',False)
        else ARCHITECTURE)
    if model.architecture!=expected_architecture or checkpoint_config.get('architecture')!=evaluation_config.get('architecture'):
        raise ValueError('Checkpoint/evaluation architecture identity differs')
    if checkpoint_config.get('observation_version')!=evaluation_config.get('observation_version'):
        raise ValueError('Checkpoint/evaluation observation version differs')
    if checkpoint_config.get('feature_groups')!=evaluation_config.get('feature_groups'):
        raise ValueError('Checkpoint/evaluation feature groups differ')
    if checkpoint_config.get('feature_schema_version')!=evaluation_config.get('feature_schema_version'):
        raise ValueError('Checkpoint/evaluation feature schema differs')
    for key in ('history_offsets','action_masks','branch_feature_normalization','network_precision',
            'mlp_sizes','cnn_channels','cnn_projection','gru_hidden','gru_layers','embedding_dim',
            'terrain_embedding_dim','frame_skip'):
        if checkpoint_config.get(key)!=evaluation_config.get(key):
            raise ValueError('Checkpoint/evaluation interface differs: '+key)
    if evaluation_config.get('frame_skip')!=1:
        raise ValueError('This suffix evaluator requires one decision per native frame')
    action_masks=list(evaluation_config['action_masks'])
    if model.actions!=len(action_masks):
        raise ValueError('Checkpoint policy output width differs from the evaluation action catalog')
    normalization_modules={
        'actor_structured':model.actor.structured_normalize,
        'actor_terrain':model.actor.terrain_normalize,
        'critic_structured':model.critic.structured_normalize,
        'critic_terrain':model.critic.terrain_normalize,
    }
    normalization_valid=all(isinstance(module,torch.nn.LayerNorm) and not module.elementwise_affine
        for module in normalization_modules.values())
    if not evaluation_config.get('branch_feature_normalization',False) or not normalization_valid:
        raise ValueError('Checkpoint does not use the declared fixed branch feature normalization')
    state_shapes={key:list(value.shape) for key,value in sorted(model.state_dict().items())}
    shape_signature=hashlib.sha256(json.dumps(state_shapes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if state_shapes.get('actor.head.weight',[None])[0]!=len(action_masks):
        raise ValueError('Loaded actor head shape does not match the action catalog')
    if int(frame) not in set(map(int,route.valid)):
        raise ValueError('Observation-contract frame is not a verified route reset')
    observation=checkpoint_route_observation_window(evaluation_config,route,int(frame))
    observed_shapes=infer_shapes(observation)
    expected_shapes={key:tuple(value) for key,value in model.shapes.items()}
    if observed_shapes!=expected_shapes:
        raise ValueError(f'Checkpoint observation shapes differ from actual route encoder: {observed_shapes!r} != {expected_shapes!r}')
    if tuple(observation['history_valid'].shape)!=(len(evaluation_config['history_offsets']),):
        raise ValueError('Actual route history length differs from the checkpoint temporal window')
    model.eval()
    with torch.no_grad():
        logits,_=model(window_tensors(observation,next(model.parameters()).device))
    if tuple(logits.shape)!=(1,len(action_masks)) or not torch.isfinite(logits).all():
        raise ValueError('Actual route observation did not produce finite logits of the declared action width')
    return dict(status='passed',architecture=model.architecture,
        parameter_tensor_count=len(state_shapes),parameter_shape_sha256=shape_signature,
        policy_output_width=model.actions,action_masks=action_masks,
        observation_version=evaluation_config['observation_version'],
        feature_schema_version=evaluation_config['feature_schema_version'],
        feature_groups=list(evaluation_config['feature_groups']),
        observation_keys=list(observed_shapes),observation_shapes={key:list(value) for key,value in observed_shapes.items()},
        history_offsets=list(evaluation_config['history_offsets']),
        history_valid=observation['history_valid'].astype(float).tolist(),
        feature_normalization=dict(configured=bool(evaluation_config['branch_feature_normalization']),
            modules={key:type(value).__name__ for key,value in normalization_modules.items()},
            affine_parameters=False),
        frame_skip=int(evaluation_config['frame_skip']),route_sha256=route.meta['route_sha256'],
        observation_contract_frame=int(frame),actual_route_window_inference='finite logits; no emulator frames')


def validate_action_omission_plan(args,config):
    """Validate one precommitted one-frame action omission without starting an emulator."""
    plan_path=args.evaluation_action_omission_plan
    condition=args.evaluation_action_omission_condition
    if (plan_path is None)!=(condition is None):
        raise ValueError('Action-omission plan and condition must be supplied together')
    if plan_path is None:return None
    if args.stage!='evaluate':
        raise ValueError('Action-omission overrides are available only during evaluation')
    manifest=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if manifest.get('status')!='precommitted_before_gameplay':
        raise ValueError('Action-omission plan was not precommitted before gameplay')
    if (int(args.reset_frame or -1)!=389 or args.no_reset_jitter is not True
            or int(args.evaluation_horizon_frames or 0)!=36000
            or args.evaluation_mode!='greedy' or int(args.trials)!=1
            or int(args.evaluation_seed or -1)!=20261022
            or int(args.delayed_takeover_noop_prefix_frames or 0)!=0):
        raise ValueError('Action-omission suite settings differ from the precommitted reset-389 evaluation')
    allow_precommitted_actor_difference=bool(getattr(args,'allow_precommitted_actor_difference',False))
    if sha256_file(args.resume)!=manifest.get('evaluation_actor_sha256') and not allow_precommitted_actor_difference:
        raise ValueError('Action-omission actor hash differs from precommit')
    if sha256_file(args.config)!=manifest.get('evaluation_config_sha256') and not allow_precommitted_actor_difference:
        raise ValueError('Action-omission config hash differs from precommit')
    if allow_precommitted_actor_difference:
        source_config_path=ROOT/manifest['evaluation_config_path']
        if sha256_file(source_config_path)!=manifest.get('evaluation_config_sha256'):
            raise ValueError('Precommitted evaluation config source hash mismatch')
        source_config=json.loads(source_config_path.read_text(encoding='utf-8-sig'))
        evaluation_contract=('architecture','action_masks','observation_version','feature_schema_version',
            'history_offsets','feature_groups','frame_skip','weights','reward_version','combat_observer_diagnostics',
            'verified_enemy_kills','score_gain_weight','deployment_rule','network_precision','mlp_sizes','gru_hidden',
            'gru_layers','cnn_channels','cnn_projection','embedding_dim','terrain_embedding_dim')
        differences=[key for key in evaluation_contract if config.get(key)!=source_config.get(key)]
        if differences:
            raise ValueError('Changed model endpoint does not preserve the precommitted evaluation contract: '+repr(differences))
    trace_path=ROOT/manifest['source_trace_path']
    if sha256_file(trace_path)!=manifest.get('source_trace_sha256'):
        raise ValueError('Action-omission source trace hash differs from precommit')
    reference_episode=json.loads((ROOT/manifest['source_episode_result_path']).read_text(encoding='utf-8'))
    schema=json.loads((ROOT/manifest['source_policy_input_schema_path']).read_text(encoding='utf-8'))
    input_path=ROOT/manifest['source_policy_input_path']
    if (not input_path.is_file()
            or int(schema['record_bytes'])!=int(manifest['source_policy_input_record_bytes'])):
        raise ValueError('Action-omission reference input stream is missing or has an invalid length')
    reference_fingerprint_manifest=None
    reference_trace=None
    if args.compact_evaluation_capture:
        if args.reference_policy_input_fingerprints is None:
            raise ValueError('Compact capture requires --reference-policy-input-fingerprints')
        reference_fingerprint_manifest=json.loads(
            Path(args.reference_policy_input_fingerprints).read_text(encoding='utf-8'))
        if (reference_fingerprint_manifest.get('status')!='verified'
                or reference_fingerprint_manifest.get('serialization_version')!=SERIALIZATION_VERSION
                or reference_fingerprint_manifest.get('source_trace_sha256')!=manifest.get('source_trace_sha256')
                or reference_fingerprint_manifest.get('source_policy_inputs_sha256')!=manifest.get('source_policy_inputs_sha256')
                or int(reference_fingerprint_manifest.get('record_bytes',-1))!=int(schema['record_bytes'])):
            raise ValueError('Compact reference fingerprint bundle does not match the precommitted source')
        if (int(input_path.stat().st_size)!=int(reference_fingerprint_manifest.get('record_count',-1))*int(schema['record_bytes'])
                or int(reference_fingerprint_manifest.get('record_count',-1))<1):
            raise ValueError('Compact reference fingerprint count differs from saved policy-input bytes')
        fingerprint_data=Path(reference_fingerprint_manifest['data_path'])
        if (not fingerprint_data.is_file()
                or sha256_file(fingerprint_data)!=reference_fingerprint_manifest.get('data_sha256')):
            raise ValueError('Compact reference fingerprint data is missing or changed')
    else:
        reference_trace=json.loads(trace_path.read_text(encoding='utf-8'))
        if input_path.stat().st_size!=len(reference_trace)*int(schema['record_bytes']):
            raise ValueError('Action-omission reference input stream is missing or has an invalid length')
    selected=[row for row in manifest['conditions'] if row.get('condition')==condition]
    if len(selected)!=1 or selected[0].get('status')!='precommitted':
        raise ValueError(f'No eligible precommitted action omission exists for condition {condition!r}')
    selected=selected[0]
    index=int(selected['policy_decision_index_0based'])
    if args.compact_evaluation_capture:
        if index>=int(reference_fingerprint_manifest['record_count']):
            raise ValueError('Precommitted intervention is outside the saved source trace')
        reference=None
        reader=ReferenceFingerprintReader(reference_fingerprint_manifest['data_path'],
            expected_record_count=int(reference_fingerprint_manifest['record_count']))
        prior_ram=reference_episode['initial_state']['ram_sha256']
        try:
            for decision in range(index+1):
                reference=reader.read(decision)
                if reference is None:
                    raise ValueError('Compact reference fingerprint stream ended before intervention')
                if decision<index:
                    prior_ram=reference.get('ram_sha256')
        finally:
            reader.close()
        expected_pre_ram=reference_episode['initial_state']['ram_sha256'] if index==0 else prior_ram
    else:
        if not 0<=index<len(reference_trace):
            raise ValueError('Precommitted intervention is outside the saved source trace')
        reference=reference_trace[index]
        expected_pre_ram=(reference_episode['initial_state']['ram_sha256'] if index==0
            else reference_trace[index-1].get('ram_sha256'))
    if (int(reference.get('action',-1))!=int(selected['reference_action_index'])
            or int(reference.get('mask',-1))!=int(selected['reference_action_mask'])
            or int(reference['mask'])==0
            or int(selected['decision_1based'])!=index+1
            or int(selected['pre_action_route_frame'])!=389+index
            or selected.get('reference_pre_action_ram_sha256')!=expected_pre_ram):
        raise ValueError('Precommitted action, mask, decision index, or route frame differs from source trace')
    noop_index=list(config['action_masks']).index(0)
    if (int(selected.get('executed_action_index',-1))!=noop_index
            or int(selected.get('executed_action_mask',-1))!=0):
        raise ValueError('Precommitted intervention does not execute the catalog NOOP action')
    if reference_episode.get('initial_state',{}).get('ram_sha256')!=manifest.get('source_initial_ram_sha256'):
        raise ValueError('Precommitted initial reset RAM hash differs from source episode report')
    if not reference_episode.get('validity',{}).get('status')=='valid':
        raise ValueError('Source episode trace is not marked valid')
    source_validity=reference_episode.get('validity',{})
    if any(source_validity.get(key) is not True for key in (
            'frame_accounting','reset_matches_verified_route','policy_input_capture_matches_decisions')):
        raise ValueError('Source episode lacks verified reset, frame, or policy-input capture evidence')
    selected=dict(selected)
    selected.update(reference_trace=reference_trace,reference_episode=reference_episode,
        reference_input_path=input_path,reference_record_bytes=int(schema['record_bytes']),
        reference_input_schema=schema,noop_action_index=noop_index,source_trace_path=trace_path,
        same_checkpoint_probabilities_required=bool(
            sha256_file(args.resume)==manifest.get('evaluation_actor_sha256')
            and sha256_file(args.config)==manifest.get('evaluation_config_sha256')))
    if args.compact_evaluation_capture:
        selected.update(reference_fingerprint_manifest=reference_fingerprint_manifest,
            reference_fingerprint_path=Path(reference_fingerprint_manifest['data_path']),
            reference_record_count=int(reference_fingerprint_manifest['record_count']),
            reference_initial_ram_sha256=reference_episode['initial_state']['ram_sha256'])
    return selected


def action_with_one_shot_omission(proposed_action,decision_index,plan,already_applied):
    """Return the executed action; only the precommitted decision may change once."""
    proposed_action=int(proposed_action)
    if decision_index==int(plan['policy_decision_index_0based']):
        if already_applied:raise RuntimeError('Precommitted action omission was encountered more than once')
        if int(plan['reference_action_mask'])==0:
            raise RuntimeError('A NOOP cannot be selected as the omitted non-NOOP action')
        return int(plan['executed_action_index']),True
    return proposed_action,False


def compare_reference_greedy_action(reference,proposed_action,actual_probabilities,action_masks,*,
        require_same_checkpoint_probabilities=False):
    """Compare a greedy proposal with its reference action and retain probability diagnostics.

    Probability equality is a gate only when the evaluator is reproducing the same
    checkpoint inference contract. A changed actor may have different confidence
    while selecting the same action from the same state and complete model input.
    """
    source_action=int(reference.get('action',-1));source_mask=int(reference.get('mask',-1))
    proposed_action=int(proposed_action);masks=list(action_masks)
    source_catalog_match=(0<=source_action<len(masks) and masks[source_action]==source_mask)
    proposed_catalog_match=(0<=proposed_action<len(masks))
    action_match=(source_catalog_match and proposed_catalog_match
        and proposed_action==source_action and masks[proposed_action]==source_mask)
    source_probabilities=reference.get('action_probabilities')
    source_probability_vector=None
    max_abs_difference=None;probabilities_match=None
    if source_probabilities is not None:
        try:
            expected=np.asarray(source_probabilities,dtype=np.float64)
            actual=np.asarray(actual_probabilities,dtype=np.float64)
            if expected.ndim==1 and np.isfinite(expected).all():
                source_probability_vector=expected.tolist()
            if (expected.shape==actual.shape and expected.ndim==1
                    and np.isfinite(expected).all() and np.isfinite(actual).all()):
                max_abs_difference=float(np.max(np.abs(expected-actual)))
                probabilities_match=bool(np.allclose(actual,expected,rtol=0.,atol=1e-7))
            else:
                probabilities_match=False
        except (TypeError,ValueError):
            probabilities_match=False
    inference_match=(probabilities_match is True) if require_same_checkpoint_probabilities else True
    return dict(source_action=source_action,source_mask=source_mask,
        proposed_action=proposed_action,
        proposed_mask=masks[proposed_action] if proposed_catalog_match else None,
        action_match=bool(action_match),source_catalog_match=bool(source_catalog_match),
        source_probabilities_match=probabilities_match,
        source_probability_max_abs_difference=max_abs_difference,
        source_probabilities=source_probability_vector,
        same_checkpoint_probability_match_required=bool(require_same_checkpoint_probabilities),
        inference_probability_match=bool(inference_match),
        accepted=bool(action_match and inference_match))


def policy_input_schema_matches_reference(actual,reference):
    """Check model input field names, order, dtypes, shapes, and byte layout."""
    return (list(actual.get('model_window_keys',[]))==list(reference.get('model_window_keys',[]))
        and actual.get('arrays')==reference.get('arrays')
        and int(actual.get('record_bytes',-1))==int(reference.get('record_bytes',-2)))


def fingerprint_schema_matches_reference(actual,reference):
    """Compare fingerprint schema metadata separately from the field contents."""
    if (actual.get('serialization_version')!=reference.get('serialization_version')
            or actual.get('comparison_scope')!=reference.get('comparison_scope')):
        return False
    actual_fields=actual.get('fields',{});reference_fields=reference.get('fields',{})
    return (list(actual_fields)==list(reference_fields)
        and all(actual_fields[key].get('dtype')==reference_fields[key].get('dtype')
            and actual_fields[key].get('shape')==reference_fields[key].get('shape')
            for key in actual_fields))


def evaluation_preparation_result(args,config,route,checkpoint,provenance,checkpoint_metadata):
    """Return the validated suffix-evaluation request without creating an emulator."""
    if args.stage!='evaluate' or args.resume is None:
        raise ValueError('--evaluation-preparation-only requires evaluation with --resume')
    if args.evaluation_mode not in ('greedy','sampled'):
        raise ValueError('Offline suffix preparation requires one explicit evaluation mode')
    if args.trials!=1 or args.reset_frame is None or not args.no_reset_jitter:
        raise ValueError('Offline suffix preparation requires one fixed, non-jittered start')
    if args.reset_frame not in set(int(frame) for frame in route.valid):
        raise ValueError('Evaluation reset frame is not a valid recorded route state')
    if args.evaluation_horizon_frames is None or args.evaluation_horizon_frames<1:
        raise ValueError('Offline suffix preparation requires a positive fixed native-frame horizon')
    if args.evaluation_seed is None:
        raise ValueError('Offline matched preparation requires an explicit common evaluation seed')
    training_frames=optional_training_counter(checkpoint_metadata,'frames')
    training_updates=optional_training_counter(checkpoint_metadata,'updates')
    evaluation_id=validate_evaluation_identifier(args.evaluation_id)
    if training_frames is None and evaluation_id is None:
        raise ValueError('Checkpoint has no recorded training frame counter; --evaluation-id is required for output naming')
    mode_seed=int(args.evaluation_seed)+int(args.evaluation_mode=='greedy')
    omission=validate_action_omission_plan(args,config)
    return dict(stage='evaluation_preparation',status='passed',checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),config_sha256=sha256_file(args.config),
        mode=args.evaluation_mode,mode_seed=mode_seed,evaluation_seed_base=int(args.evaluation_seed),
        trials=args.trials,start_frame=int(args.reset_frame),reset_jitter=False,
        horizon_native_frames=int(args.evaluation_horizon_frames),
        training_frames=training_frames,training_updates=training_updates,
        evaluation_id=evaluation_id,
        noop_prefix_frames=int(args.delayed_takeover_noop_prefix_frames or 0),
        prefix_frames_included_in_horizon=True,
        policy_input_capture_requested=bool(args.save_evaluation_policy_inputs),
        compact_capture_requested=bool(args.compact_evaluation_capture),
        reference_policy_input_fingerprints=(str(args.reference_policy_input_fingerprints)
            if args.reference_policy_input_fingerprints is not None else None),
        score_gain_weight=float(config.get('weights',{}).get('score_gain',0.)),
        enemy_kills_weight=float(config.get('weights',{}).get('enemy_kills',0.)),
        action_omission=(None if omission is None else {key:omission[key] for key in
            ('condition','decision_1based','policy_decision_index_0based','pre_action_route_frame',
             'reference_action_index','reference_action_mask','executed_action_index','executed_action_mask',
             'source_trace_sha256','same_checkpoint_probabilities_required')}),
        checkpoint_provenance=provenance,
        environment_pool_started=False,workers_started=False,optimizer_created=False,
        training_state_restored=False,emulator_frames=0)


def save(path,model,optimizer,pool,curriculum,rng,meta,*,skip_if_identical=False):
    payload=dict(architecture=model.architecture,config=model.config,shapes=model.shapes,actions=model.actions,
        state_dict=model.state_dict(),metadata=dict(**meta,experiment=model.config.get('experiment','backward_level1_v1'),optimizer=optimizer.state_dict(),
            rng=rng_state(rng),curriculum=curriculum.state(),environment_states=snapshots(pool),
            normalization=('fixed schema scales; non-affine per-timestamp branch LayerNorm; no fitted/running statistics'
                           if model.config.get('branch_feature_normalization',False) else 'fixed schema scales; no fitted/running statistics'),bc_enabled=bool(model.config.get('bc_enabled',False))))
    return atomic_torch_save(payload,path,skip_if_identical=skip_if_identical)


def behavior_distribution(logits,epsilon=0.,mask=None,temperature=1.):
    """Actual PPO behavior policy; same mixture used for collection and ratios."""
    if not 0<=epsilon<1:raise ValueError('Exploration epsilon outside[0,1)')
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('Positive finite policy temperature required')
    raw=masked_distribution(logits/temperature,mask)
    if epsilon==0:return raw
    allowed=torch.ones_like(raw.probs) if mask is None else torch.as_tensor(mask,device=logits.device,dtype=logits.dtype).expand_as(raw.probs)
    uniform=allowed/allowed.sum(-1,keepdim=True)
    return torch.distributions.Categorical(probs=(1-epsilon)*raw.probs+epsilon*uniform)


def exploration_epsilon(c,frames):
    schedule=c.get('exploration_epsilon_schedule')
    if schedule is None:return 0.
    fraction=min(1.,max(0.,(frames-c.get('exploration_start_frames',0))/float(schedule['frames'])))
    return float(schedule['start']+(schedule['end']-schedule['start'])*fraction)


def task_score(report,c):
    rows=report['runs']
    if c.get('task_checkpoint_rule')=='deployment':rows=[r for r in rows if r['mode']==c['deployment_rule']]
    return (sum(r['completion'] for r in rows),max(max(i for i,x in enumerate(r['frontiers']) if x>=0) for r in rows),float(np.mean([sum(max(0,x) for x in r['frontiers']) for r in rows])))


def validate_bc_mode(c, explicit, stage):
    labelled=bool(c.get('bc_initialization') or c.get('bc_enabled'))
    if labelled and not explicit:raise ValueError('Primary PPO-only mode rejects BC-initialized policies; use explicit --bc-initialized-extension')
    if explicit and not labelled:raise ValueError('BC extension flag requires declared BC provenance')
    if labelled and stage=='train' and c.get('training_method') not in ('human_BC_plus_PPO_extension_v1','human_BC_plus_PPO_temperature_extension_v1'):
        raise ValueError('BC continuation requires its separately labelled PPO extension configuration')
    if c.get('bc_enabled') and c.get('training_method') not in ('human_BC_plus_PPO_extension_v1','human_BC_plus_PPO_temperature_extension_v1'):
        raise ValueError('Auxiliary cloning is allowed only in the declared extension')


def auxiliary_weight(c,frames):
    if not c.get('bc_enabled'):return 0.
    a=c['bc_auxiliary_schedule'];fraction=min(1.,max(0.,(frames-c.get('bc_auxiliary_start_frames',0))/float(a['frames'])))
    return float(a['start']+(a['end']-a['start'])*fraction)


def load_human_auxiliary(c,device,cache_root=None):
    if not c.get('bc_enabled'):return None
    path=ROOT/c['bc_auxiliary_data']
    if sha256_file(path)!=c['bc_auxiliary_sha256']:
        raise ValueError('Auxiliary dataset digest mismatch')
    joint_format=c.get('bc_auxiliary_format') in ('verified_three_trajectory_joint_v1','verified_full_route_recovery_joint_v1')
    full_route_format=c.get('bc_auxiliary_format')=='verified_full_route_recovery_joint_v1'
    if c.get('bc_auxiliary_format') not in (None,'verified_three_trajectory_joint_v1','verified_full_route_recovery_joint_v1'):
        raise ValueError('Unknown auxiliary dataset format')
    if joint_format:
        joint_window_keys=list(WINDOW_KEYS)
        observation_version=c.get('observation_version','legacy_v3')
        if observation_version in ('rea_expanded_v1','rea_expanded_v2'):
            joint_window_keys.extend(('dynamic_actors','rope_points'))
        if observation_version=='rea_expanded_v2':
            joint_window_keys.extend(('resident_codes','resident_valid'))
        required=(*joint_window_keys,'action','action_mask','confidence','changed_action','frame','trajectory_id')
        cache_root=Path(cache_root or c.get('bc_auxiliary_cache_root') or (Path(tempfile.gettempdir())/'level1_auxiliary_mmap'))
        cache=cache_root/str(c['bc_auxiliary_sha256'])
        cache_read_only=bool(c.get('bc_auxiliary_cache_read_only',False))
        with zipfile.ZipFile(path) as z:
            members={Path(name).stem:name for name in z.namelist() if name.endswith('.npy')}
            missing=set(required)-set(members)
            if missing:raise ValueError('Joint auxiliary archive lacks required data: '+repr(sorted(missing)))
            needed_bytes=sum(z.getinfo(members[k]).file_size for k in required)
            cache_root.mkdir(parents=True,exist_ok=True)
            missing_bytes=sum(z.getinfo(members[k]).file_size for k in required
                if not (cache/f'{k}.npy').is_file()
                or (cache/f'{k}.npy').stat().st_size!=z.getinfo(members[k]).file_size)
            if cache_read_only and missing_bytes:
                raise FileNotFoundError(f'Read-only auxiliary cache is incomplete or mismatched: {cache}')
            if missing_bytes and shutil.disk_usage(cache_root).free<missing_bytes+1024**3:
                raise OSError(f'Insufficient disk headroom for bounded auxiliary cache: need {missing_bytes+1024**3} bytes')
            if not cache_read_only:
                cache.mkdir(parents=True,exist_ok=True)
            data={}
            for key in required:
                target=cache/f'{key}.npy'
                if not target.is_file() or target.stat().st_size!=z.getinfo(members[key]).file_size:
                    if cache_read_only:
                        raise FileNotFoundError(f'Read-only auxiliary cache member is unavailable: {key}')
                    pending=target.with_suffix('.npy.pending')
                    with z.open(members[key]) as src,pending.open('wb') as dst:shutil.copyfileobj(src,dst,length=1024*1024)
                    if pending.stat().st_size!=z.getinfo(members[key]).file_size:
                        pending.unlink(missing_ok=True);raise IOError(f'Incomplete auxiliary cache member: {key}')
                    pending.replace(target)
                data[key]=np.load(target,mmap_mode='r',allow_pickle=False)
        if full_route_format:
            expected_counts={0:18273,1:836,2:837,3:838,4:815}
            expected_rows=21599
        else:
            expected_counts={0:722,1:836,2:837}
            expected_rows=2395
        observed_counts=dict(zip(*np.unique(data['trajectory_id'],return_counts=True)))
        if (len(data['action'])!=expected_rows or observed_counts!=expected_counts
                or c.get('bc_auxiliary_expected_rows')!=expected_rows):
            raise ValueError('Joint auxiliary archive trajectory identity/row counts differ')
        labels=np.asarray(data['action'],dtype=np.int64)
        masks=np.asarray(data['action_mask'],dtype=np.int64)
        catalog=np.asarray(c['action_masks'],dtype=np.int64)
        if (np.any(labels<0) or np.any(labels>=len(catalog))
                or not np.array_equal(catalog[labels],masks)):
            raise ValueError('Joint auxiliary actions do not map to the configured action catalog')
        if c.get('feature_schema_version')!='level1_backward_v5_rea_gaps' or list(c['history_offsets'])!=[-60,-30,-15,-9,-6,-3,0]:
            raise ValueError('Joint auxiliary archive requires the verified v5 seven-history schema')
        data['action']=labels
        data['original_ids']=np.empty(0,dtype=np.int64)
        data['extra_ids']=np.arange(len(labels),dtype=np.int64)
        data['focus_ids']=np.empty(0,dtype=np.int64)
        data['general_ids']=np.empty(0,dtype=np.int64)
    else:
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['dataset_sha256']!=c['bc_auxiliary_sha256']:
            raise ValueError('Auxiliary dataset sidecar digest mismatch')
        for key in ('feature_schema_version','history_offsets','action_masks'):
            if meta[key]!=c[key]:raise ValueError('Auxiliary conditioning/action mismatch: '+key)
        window_keys=tuple(meta.get('policy_input_keys',WINDOW_KEYS))
        required=(*window_keys,'action','confidence','changed_action','source_kind','frame')
        with np.load(path,allow_pickle=False) as z:
            missing=set(required)-set(z.files)
            if missing:raise ValueError('Auxiliary dataset lacks declared policy inputs: '+repr(sorted(missing)))
            data={k:z[k] for k in required}
            # A prepared contiguous demonstration split may also serve as PPO auxiliary
            # data. Only its training segment is eligible; heldout and gap rows must
            # never leak into the auxiliary objective.
            if 'split' in z.files:
                train=z['split']==0
                data={k:v[train] for k,v in data.items()}
        data['original_ids']=np.flatnonzero(data['source_kind']==0)
        data['extra_ids']=np.flatnonzero(data['source_kind']==1)
    counts=np.bincount(data['action'],minlength=len(c['action_masks']))
    weights=np.clip(np.sqrt(len(data['action'])/(np.count_nonzero(counts)*np.maximum(counts,1))),.5,3.)
    data['class_weights']=torch.as_tensor(weights,dtype=torch.float32,device=device)
    focus=c.get('bc_focus_sampling') if not joint_format else None
    if focus:
        selected=np.zeros(len(data['frame']),dtype=bool)
        for low,high in focus['ranges']:selected|=(data['frame']>=int(low))&(data['frame']<int(high))
        data['focus_ids']=np.flatnonzero(selected&(data['source_kind']==0))
        if not len(data['focus_ids']):raise ValueError('Configured BC focus interval has no original human examples')
        data['general_ids']=np.setdiff1d(data['original_ids'],data['focus_ids'],assume_unique=True)
        if not len(data['general_ids']):raise ValueError('Targeted BC focus leaves no general examples for retention')
    elif not joint_format:
        data['focus_ids']=np.empty(0,dtype=np.int64);data['general_ids']=data['original_ids']
    return data


def human_auxiliary_loss(model,data,c,device,rng,n):
    count=int(round(n*c.get('bc_extra_ratio',.4)))
    original_count=n-count
    focus_fraction=float(c.get('bc_focus_sampling',{}).get('fraction',0.)) if len(data['focus_ids']) else 0.
    focus_count=int(round(original_count*focus_fraction))
    ids=rng.permutation(np.r_[rng.choice(data['focus_ids'],focus_count,replace=True) if focus_count else np.empty(0,dtype=np.int64),
        rng.choice(data['general_ids'],original_count-focus_count,replace=True),rng.choice(data['extra_ids'],count,replace=True)])
    data['last_focus_samples']=focus_count
    inputs=indexed_window_tensors(data,ids,model.window_keys,device)
    y=torch.as_tensor(data['action'][ids],device=device,dtype=torch.long)
    weights=torch.as_tensor(data['confidence'][ids]*np.where(data['changed_action'][ids],c.get('bc_edge_weight',4.),1.),device=device,dtype=torch.float32)
    loss=torch.nn.functional.cross_entropy(model.actor(inputs),y,weight=data['class_weights'],reduction='none')
    return (loss*weights).sum()/(data['class_weights'][y]*weights).sum().clamp_min(1e-8)


def optimize(model,optimizer,rollout,c,device,rng,warmup,update,auxiliary=None):
    advantages,returns=grouped_advantages(rollout,c['gamma'],c['gae_lambda'])
    norm=(advantages-advantages.mean())/(advantages.std()+1e-8)
    old_log=np.asarray([r['log'] for r in rollout],np.float32)
    actions=np.asarray([r['action'] for r in rollout],np.int64)
    conditioning_error=None
    if not warmup:
        errors=[]
        with torch.no_grad():
            for first in range(0,len(rollout),c['batch_size']):
                ix=np.arange(first,min(first+c['batch_size'],len(rollout)))
                x=window_tensors(materialize_observation_batch(rollout,ix,model.window_keys),device,batch=True)
                expected=behavior_distribution(model.actor(x),c.get('exploration_epsilon',0.),temperature=c.get('policy_temperature',1.)).log_prob(torch.as_tensor(actions[ix],device=device)).cpu().numpy()
                errors.append(float(np.max(np.abs(expected-old_log[ix]))))
                del x,expected
        conditioning_error=max(errors)
        if conditioning_error>5e-4:raise RuntimeError(f'PPO behavior log probabilities/input conditioning differ before update: {conditioning_error}')
    diagnostics=[];actor_before=parameter_hash(model.actor);critic_before=parameter_hash(model.critic);grad_steps=0;actor_steps=0;epochs=0;gradient_components={};auxiliary_samples=0;auxiliary_focus_samples=0
    if warmup:
        for p in model.actor.parameters():p.requires_grad_(False)
    for epoch in range(c['ppo_epochs']):
        values=[]
        for start in range(0,len(rollout),c['batch_size']):
            if start==0:indices=rng.permutation(len(rollout))
            ix=indices[start:start+c['batch_size']]
            x=window_tensors(materialize_observation_batch(rollout,ix,model.window_keys),device,batch=True)
            target=torch.as_tensor(returns[ix],device=device)
            value=model.critic(x).squeeze(-1);vl=.5*(value-target).square().mean()
            pg=vl.new_zeros(());ent=pg;raw_ent=pg;kl=pg;clip=pg
            if not warmup:
                logits=model.actor(x);raw_ent=masked_distribution(logits).entropy().mean();dist=behavior_distribution(logits,c.get('exploration_epsilon',0.),temperature=c.get('policy_temperature',1.))
                action=torch.as_tensor(actions[ix],device=device)
                log_ratio=dist.log_prob(action)-torch.as_tensor(old_log[ix],device=device);ratio=log_ratio.exp()
                adv=torch.as_tensor(norm[ix],device=device)
                pg=-torch.minimum(ratio*adv,ratio.clamp(1-c['clip_range'],1+c['clip_range'])*adv).mean()
                ent=dist.entropy().mean();kl=((ratio-1)-log_ratio).mean()
                clip=((ratio-1).abs()>c['clip_range']).float().mean()
            bc_loss=pg.new_zeros(());bc_weight=float(c.get('bc_auxiliary_weight',0.))
            if auxiliary is not None and not warmup:
                bc_loss=human_auxiliary_loss(model,auxiliary,c,device,rng,len(ix));auxiliary_samples+=len(ix)
                auxiliary_focus_samples+=int(auxiliary.get('last_focus_samples',0))
            loss=pg+vl-c['entropy_coefficient']*ent+bc_weight*bc_loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
            if not warmup and epoch==0 and start==0 and update%c.get('gradient_diagnostic_interval',10)==0:
                params=tuple(model.actor.parameters())
                for name,term in (('ppo',pg),('weighted_entropy',-c['entropy_coefficient']*ent),('weighted_bc',bc_weight*bc_loss)):
                    if not term.requires_grad:continue
                    grads=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
                    gradient_components[name+'_gradient_norm_first_batch']=float(torch.sqrt(sum(g.detach().square().sum() for g in grads if g is not None)))
            optimizer.zero_grad(set_to_none=True);loss.backward()
            actor_grad=torch.nn.utils.clip_grad_norm_(model.actor.parameters(),.5)
            critic_grad=torch.nn.utils.clip_grad_norm_(model.critic.parameters(),.5)
            optimizer.step();grad_steps+=1;actor_steps+=int(not warmup)
            row=dict(actor_loss=float(pg.detach()),critic_loss=float(vl.detach()),entropy=float(ent.detach()),actor_raw_entropy=float(raw_ent.detach()),bc_loss=float(bc_loss.detach()),weighted_bc_loss=float((bc_weight*bc_loss).detach()),bc_auxiliary_weight=bc_weight,
                approximate_kl=float(kl.detach()),clip_fraction=float(clip.detach()),actor_grad=float(actor_grad),critic_grad=float(critic_grad))
            diagnostics.append(row);values.append(row)
            if not warmup:del logits,dist,action,log_ratio,ratio,adv
            del x,target,value,vl,pg,ent,raw_ent,kl,clip,bc_loss,loss,actor_grad,critic_grad
        epochs+=1
        print(f'PPO update {update} | optimization epoch {epoch+1}/{c["ppo_epochs"]} | '
              f'KL {np.mean([r["approximate_kl"] for r in values]):.5f} critic loss {np.mean([r["critic_loss"] for r in values]):.4f}',flush=True)
        if not warmup and np.mean([r['approximate_kl'] for r in values])>c['target_kl']:break
    if warmup:
        if parameter_hash(model.actor)!=actor_before:raise RuntimeError('Critic warmup changed actor')
        for p in model.actor.parameters():p.requires_grad_(True)
    predicted=[]
    with torch.no_grad():
        for start in range(0,len(rollout),c['batch_size']):
            ix=np.arange(start,min(start+c['batch_size'],len(rollout)))
            x=window_tensors(materialize_observation_batch(rollout,ix,model.window_keys),device,batch=True)
            predicted.extend(model.critic(x).squeeze(-1).cpu().numpy())
            del x
    ev=1-np.var(returns-np.asarray(predicted))/max(float(np.var(returns)),1e-8)
    # Diagnostic only: actual terminal returns wholly observed in THIS rollout.
    # Never replace on-policy GAE targets or invent values at truncations.
    terminal_errors=[]
    for stream in sorted({r.get('env_id',0) for r in rollout}):
        tail=0.;available=False
        for r in reversed([r for r in rollout if r.get('env_id',0)==stream]):
            if not r['continue']:
                available=r.get('end_reason') in ('game_over','level_completed','training_life_loss');tail=0.
            if available:
                tail=r['reward']+c['gamma']**r['duration']*tail
                terminal_errors.append(r['value']-tail)
    terminal_diagnostics=dict(terminal_mc_samples=len(terminal_errors),
        terminal_mc_mse=float(np.mean(np.square(terminal_errors))) if terminal_errors else None,
        terminal_mc_bias=float(np.mean(terminal_errors)) if terminal_errors else None,
        bootstrap_fraction=float(np.mean([r['bootstrap'] for r in rollout])),
        ended_by_reason=dict(Counter(r.get('end_reason') for r in rollout if not r['continue'])))
    return dict(conditioning_logprob_max_error=conditioning_error,bc_auxiliary_samples=auxiliary_samples,bc_focus_samples=auxiliary_focus_samples,**terminal_diagnostics,**{k:float(np.mean([r[k] for r in diagnostics])) for k in diagnostics[0]},
        return_mean=float(returns.mean()),return_std=float(returns.std()),advantage_mean=float(advantages.mean()),
        advantage_std=float(advantages.std()),explained_variance=float(ev),critic_mse=float(np.mean((returns-predicted)**2)),
        warmup=warmup,optimizer_steps=grad_steps,actor_optimizer_steps=actor_steps,optimization_epochs=epochs,
        actor_changed=actor_before!=parameter_hash(model.actor),critic_changed=critic_before!=parameter_hash(model.critic),
        **gradient_components)


def flush_evaluation_values(model,pending,device,batch=512):
    """Same frozen critic conditioning, batched; never used to select actions."""
    with torch.no_grad():
        for start in range(0,len(pending),batch):
            chunk=pending[start:start+batch]
            x=window_tensors({k:np.stack([o[k] for row,o in chunk]) for k in model.window_keys},device,batch=True)
            values=model.critic(x).squeeze(-1).cpu().numpy()
            if not np.isfinite(values).all():raise RuntimeError('Nonfinite evaluation critic values')
            for (row,_),value in zip(chunk,values):row['value']=float(value)
    pending.clear()


def evaluate(model,pool,c,route,folder,frames,*,trials=None,final=False,record=False,reset_frame=None,all_suffix_starts=False,
             no_reset_jitter=False,evaluation_horizon_frames=None,evaluation_mode=None,
             preserve_pool_state=True,save_policy_inputs=False,noop_prefix_frames=0,
             checkpoint_provenance=None,evaluation_id=None,action_omission=None,
             compact_capture=False,storage_guard=None,artifact_guard=None):
    """Development waits are known conditions; held-out waits used only final=True."""
    if noop_prefix_frames not in (0,1,2,4,8,16):
        raise ValueError('The isolated delayed-takeover prefix must be 0, 1, 2, 4, 8, or 16 native frames')
    if noop_prefix_frames and (reset_frame is None or not no_reset_jitter or c['frame_skip']!=1):
        raise ValueError('Delayed-takeover prefixes require an exact suffix reset and frame_skip=1')
    if (evaluation_horizon_frames is not None
            and noop_prefix_frames>=int(evaluation_horizon_frames)):
        raise ValueError('The NOOP prefix must be shorter than the evaluation horizon')
    noop_action_index=c['action_masks'].index(0) if noop_prefix_frames else None
    rng_saved=torch.get_rng_state();cuda_saved=torch.cuda.get_rng_state_all()
    if preserve_pool_state:pool.stash_many()
    begun=time.perf_counter();last_heartbeat=begun;rows=[];total_frames=0;policy_input_artifacts=[]
    omission_applied=False;first_perturbation_divergence=None
    trials=trials or c['evaluation_trials_per_rule'];count=len(pool.parents)
    evaluation_id=validate_evaluation_identifier(evaluation_id)
    if frames is None and evaluation_id is None:
        raise ValueError('Evaluation without a recorded training frame counter requires an explicit evaluation identifier')
    if frames is None and c.get('evaluation_seed') is None:
        raise ValueError('Evaluation without a recorded training frame counter requires an explicit evaluation seed')
    label=(f'evaluation_{evaluation_id}' if evaluation_id is not None else f'frames_{frames:09d}')
    label+=('_confirmation' if final else '')
    if reset_frame is not None:label+=f'_suffix_{reset_frame}'
    target=folder/'evaluation'/label
    ordinal=1
    while target.exists():
        target=folder/'evaluation'/f'{label}_batch{ordinal}';ordinal+=1
    target.mkdir(parents=True)
    try:
        modes=(evaluation_mode,) if evaluation_mode else ('sampled','greedy')
        for mode in modes:
            variations=([60,90,120,150,180] if final else [0,6])
            if reset_frame is not None:
                candidates=(np.asarray([reset_frame],dtype=int) if no_reset_jitter else
                    route.valid[(route.valid>=reset_frame)&(route.valid<=reset_frame+c['frontier_jitter_frames'])])
                if not len(candidates):raise ValueError('No valid suffix evaluation starts')
                variations=(candidates.tolist() if all_suffix_starts else np.unique(candidates[np.linspace(0,len(candidates)-1,min(5,len(candidates))).astype(int)]).tolist())
            explicit_seed=c.get('evaluation_seed')
            mode_seed=(int(explicit_seed)+(int(mode=='greedy')) if explicit_seed is not None else
                c['seed']+int(frames)+int(mode=='greedy')+100000+1009*(ordinal-1))
            torch.manual_seed(mode_seed)
            # Duplicate deterministic conditions do not add evidence.
            n=trials if mode=='sampled' else (len(variations) if all_suffix_starts else min(trials,len(variations)))
            for first in range(0,n,count):
                identifiers=list(range(min(count,n-first)))
                cases=[dict(frame=route.start,horizon=c['full_horizon'],variation=variations[(first+i)%len(variations)]) for i in identifiers]
                if reset_frame is not None:
                    cases=[dict(frame=variations[(first+i)%len(variations)]) for i in identifiers]
                if evaluation_horizon_frames is not None:
                    cases=[dict(case,horizon=int(evaluation_horizon_frames)) for case in cases]
                try:
                    if storage_guard is not None:
                        storage_guard.check()
                    if artifact_guard is not None:
                        artifact_guard.check(force=True)
                except (StorageGuardInterruption,ArtifactBudgetInterruption) as exc:
                    status=partial_capture_status(frames=0,completion=None,
                        terminal_reason='storage_interruption_before_reset',interruption=repr(exc),
                        measurements=getattr(exc,'measurements',None))
                    write_json(target/f'{mode}_{first}.capture_status.json',status)
                    raise
                try:
                    states=pool.reset_many(cases,identifiers)
                except BaseException as exc:
                    if compact_capture:
                        write_json(target/f'{mode}_{first}.capture_status.json',partial_capture_status(
                            frames=0,completion=None,terminal_reason='evaluator_startup_error',
                            interruption=repr(exc)))
                    raise
                states=dict(zip(identifiers,states));active=identifiers.copy()
                initial_states={i:{key:states[i]['info'].get(key) for key in
                    ('health','lives','player_x','player_y','frontiers','objective_flags_raw',
                     'score_gain_coverage','ram_sha256','reset_frame','reset_replay_frames')}
                    for i in identifiers}
                returns=Counter();last_change=Counter();last_signature={}
                traces={i:None if compact_capture else [] for i in identifiers}
                frame_counts=Counter();writers={};pending=[];last_infos={i:states[i]['info'] for i in identifiers}
                journals=({} if compact_capture else {
                    i:(target/f'{mode}_{first+i}.partial.jsonl').open('w',encoding='utf-8',buffering=1)
                    for i in identifiers})
                trace_writers={}
                completed_ids=set()
                if compact_capture:
                    for i in identifiers:
                        trace_writers[i]=BoundedJsonlWriter(target/f'{mode}_{first+i}.partial.jsonl.gz')
                input_writers={};input_raw_writers={};input_index_writers={};input_offsets=Counter();input_schemas={}
                policy_decision_counts=Counter()
                if save_policy_inputs and compact_capture:
                    input_dir=target/'policy_inputs';input_dir.mkdir(parents=True,exist_ok=True)
                elif save_policy_inputs:
                    input_dir=target/'policy_inputs';input_dir.mkdir(parents=True,exist_ok=True)
                    for i in identifiers:
                        stem=f'{mode}_{first+i}'
                        input_path=input_dir/f'{stem}.bin.gz'
                        index_path=input_dir/f'{stem}.index.jsonl'
                        input_raw_writers[i]=input_path.open('wb',buffering=0)
                        input_writers[i]=gzip.GzipFile(filename='',mode='wb',compresslevel=1,
                            fileobj=input_raw_writers[i],mtime=0)
                        input_index_writers[i]=index_path.open('w',encoding='utf-8',buffering=1)
                        policy_input_artifacts.append(dict(mode=mode,trial=first+i,
                            data_path=str(input_path.relative_to(ROOT)),
                            data_encoding='gzip; decompression reproduces concatenated native NumPy record bytes',
                            index_path=str(index_path.relative_to(ROOT)),
                            schema_path=str((input_dir/f'{stem}.schema.json').relative_to(ROOT))))
                deferred=bool(c.get('defer_critic',False));device=next(model.parameters()).device
                if record:
                    import imageio_ffmpeg
                    for i in identifiers[:1]:
                        writers[i]=imageio_ffmpeg.write_frames(str(target/f'{mode}_{first+i}.mp4'),(256,240),fps=30,
                            codec='libx264',pix_fmt_out='yuv420p',quality=7,ffmpeg_log_level='error')
                        writers[i].send(None)
                reference_input_stream=None
                reference_fingerprint_reader=None
                reference_checks_by_id={}
                reference_rows_by_id={}
                reference_previous_ram={i:action_omission.get('reference_initial_ram_sha256')
                    if action_omission is not None else None for i in identifiers}
                policy_fingerprints_by_id={}
                intervention_by_id={}
                proposed_by_id={}
                try:
                    if action_omission is not None:
                        if compact_capture:
                            reference_fingerprint_reader=ReferenceFingerprintReader(
                                action_omission['reference_fingerprint_path'],
                                expected_record_count=action_omission['reference_record_count'])
                        else:
                            reference_input_stream=action_omission['reference_input_path'].open('rb')
                    while active:
                        if storage_guard is not None:
                            storage_guard.check()
                        if artifact_guard is not None:
                            artifact_guard.check()
                        scripted_prefix={i for i in active
                            if frame_counts[i]<noop_prefix_frames}
                        policy_active=[i for i in active if i not in scripted_prefix]
                        action_by_id={i:noop_action_index for i in scripted_prefix}
                        probability_by_id={i:None for i in scripted_prefix}
                        value_by_id={i:None for i in scripted_prefix}
                        observation_by_id={}
                        if policy_active:
                            policy_observations=[states[i]['obs'] for i in policy_active]
                            observation_by_id=dict(zip(policy_active,policy_observations))
                            batched_observations={k:np.stack([o[k] for o in policy_observations])
                                for k in model.window_keys}
                            with torch.no_grad():
                                x=window_tensors(batched_observations,next(model.parameters()).device,batch=True)
                                if save_policy_inputs:
                                    for batch_index,i in enumerate(policy_active):
                                        decision=policy_decision_counts[i]
                                        if compact_capture:
                                            observation={key:batched_observations[key][batch_index]
                                                for key in model.window_keys}
                                            if i not in input_schemas:
                                                input_schemas[i]=dict(
                                                    format='per-field SHA-256 of canonical contiguous NumPy bytes',
                                                    **policy_input_schema(observation,model.window_keys),
                                                    reset_frame=reset_frame,mode=mode,trial=first+i,
                                                    noop_prefix_frames=noop_prefix_frames,
                                                    takeover_route_frame=(int(reset_frame)+noop_prefix_frames
                                                        if reset_frame is not None else None))
                                                write_json(target/'policy_inputs'/f'{mode}_{first+i}.schema.json',
                                                    input_schemas[i])
                                                policy_input_artifacts.append(dict(mode=mode,trial=first+i,
                                                    data_path=str((target/f'{mode}_{first+i}.jsonl.gz').relative_to(ROOT)),
                                                    data_encoding='gzip JSONL with per-field canonical input fingerprints',
                                                    schema_path=str((target/'policy_inputs'/f'{mode}_{first+i}.schema.json').relative_to(ROOT)),
                                                    serialization_version=SERIALIZATION_VERSION,
                                                    comparison_scope='exact model inputs through reference-source exhaustion'))
                                            actual_fingerprint=fingerprint_policy_input(observation,model.window_keys)
                                            policy_fingerprints_by_id[i]=actual_fingerprint
                                            payload=None
                                        else:
                                            if i not in input_schemas:
                                                specs=[];offset=0
                                                for key in model.window_keys:
                                                    sample=np.ascontiguousarray(batched_observations[key][batch_index])
                                                    specs.append(dict(key=key,dtype=sample.dtype.str,shape=list(sample.shape),
                                                        byte_offset=offset,byte_length=int(sample.nbytes)))
                                                    offset+=int(sample.nbytes)
                                                input_schemas[i]=dict(format='concatenated native NumPy array bytes',
                                                    model_window_keys=list(model.window_keys),arrays=specs,
                                                    record_bytes=offset,reset_frame=reset_frame,mode=mode,trial=first+i,
                                                    noop_prefix_frames=noop_prefix_frames,
                                                    takeover_route_frame=(int(reset_frame)+noop_prefix_frames
                                                        if reset_frame is not None else None))
                                                write_json(target/'policy_inputs'/f'{mode}_{first+i}.schema.json',input_schemas[i])
                                            schema=input_schemas[i]
                                            payload=b''.join(np.ascontiguousarray(batched_observations[item['key']][batch_index]).tobytes()
                                                for item in schema['arrays'])
                                            if len(payload)!=schema['record_bytes']:
                                                raise RuntimeError('Captured policy-input byte count changed inside an episode')
                                            actual_fingerprint=None
                                        if action_omission is not None:
                                            actual_pre_ram=states[i]['info'].get('ram_sha256')
                                            if compact_capture:
                                                reference=reference_fingerprint_reader.read(decision)
                                                reference_rows_by_id[i]=reference
                                                if reference is None:
                                                    check=dict(policy_decision_index=decision,decision_1based=decision+1,
                                                        pre_action_route_frame=389+decision,
                                                        expected_pre_action_ram_sha256=None,
                                                        actual_pre_action_ram_sha256=actual_pre_ram,
                                                        pre_action_ram_match=None,
                                                        policy_input_match_exactly=None,
                                                        reference_status='source_trace_exhausted_no_later_labels')
                                                else:
                                                    expected_pre_ram=reference_previous_ram[i]
                                                    schema_match=fingerprint_schema_matches_reference(
                                                        actual_fingerprint,reference['input'])
                                                    input_match=fingerprints_equal(actual_fingerprint,reference['input'])
                                                    ram_match=actual_pre_ram==expected_pre_ram
                                                    check=dict(policy_decision_index=decision,decision_1based=decision+1,
                                                        pre_action_route_frame=389+decision,
                                                        expected_pre_action_ram_sha256=expected_pre_ram,
                                                        actual_pre_action_ram_sha256=actual_pre_ram,
                                                        pre_action_ram_match=ram_match,
                                                        observation_schema_match=schema_match,
                                                        policy_input_match_exactly=input_match,
                                                        reference_status='compared')
                                                    reference_previous_ram[i]=reference.get('ram_sha256')
                                            else:
                                                reference_trace=action_omission['reference_trace']
                                                if decision>=len(reference_trace):
                                                    check=dict(policy_decision_index=decision,decision_1based=decision+1,
                                                        pre_action_route_frame=389+decision,
                                                        expected_pre_action_ram_sha256=None,
                                                        actual_pre_action_ram_sha256=actual_pre_ram,
                                                        pre_action_ram_match=None,
                                                        policy_input_match_exactly=None,
                                                        reference_status='source_trace_exhausted_no_later_labels')
                                                else:
                                                    reference_input_stream.seek(decision*action_omission['reference_record_bytes'])
                                                    reference_payload=reference_input_stream.read(action_omission['reference_record_bytes'])
                                                    if len(reference_payload)!=action_omission['reference_record_bytes']:
                                                        raise RuntimeError('Reference policy-input stream is shorter than its validated source trace')
                                                    actual_schema=policy_input_schema(
                                                        {key:batched_observations[key][batch_index]
                                                            for key in model.window_keys},model.window_keys)
                                                    schema_match=policy_input_schema_matches_reference(
                                                        actual_schema,action_omission['reference_input_schema'])
                                                    input_match=(payload==reference_payload)
                                                    expected_pre_ram=(action_omission['reference_episode']['initial_state']['ram_sha256']
                                                        if decision==0 else reference_trace[decision-1].get('ram_sha256'))
                                                    ram_match=(actual_pre_ram==expected_pre_ram)
                                                    check=dict(policy_decision_index=decision,decision_1based=decision+1,
                                                        pre_action_route_frame=389+decision,
                                                        expected_pre_action_ram_sha256=expected_pre_ram,
                                                        actual_pre_action_ram_sha256=actual_pre_ram,
                                                        pre_action_ram_match=ram_match,
                                                        observation_schema_match=schema_match,
                                                        policy_input_match_exactly=bool(input_match and schema_match),
                                                        policy_input_values_match_exactly=input_match,
                                                        reference_status='compared')
                                                    reference_rows_by_id[i]=reference_trace[decision]
                                            reference_checks_by_id[i]=check
                                            if (decision<=int(action_omission['policy_decision_index_0based'])
                                                    and check.get('reference_status')=='compared'
                                                    and not (check['pre_action_ram_match']
                                                        and check.get('observation_schema_match',True)
                                                        and check['policy_input_match_exactly'])):
                                                expected_input_hash=(reference_rows_by_id[i]['input']['record_sha256']
                                                    if compact_capture else hashlib.sha256(reference_payload).hexdigest())
                                                actual_input_hash=(actual_fingerprint['record_sha256'] if compact_capture
                                                    else hashlib.sha256(payload).hexdigest())
                                                failure=dict(status='stopped_before_intervention_unmatched_source',
                                                    failure_reason='pre_action_state_or_full_input_mismatch',
                                                    condition=action_omission['condition'],**check,
                                                    expected_reference_input_sha256=expected_input_hash,
                                                    actual_policy_input_sha256=actual_input_hash,
                                                    planned_intervention_decision=action_omission['decision_1based'],
                                                    policy_frames_already_executed=int(frame_counts[i]),
                                                    reset_reconstruction_frames=initial_states[i].get('reset_replay_frames'),
                                                    initial_state=initial_states[i],
                                                    intervention_injected=False)
                                                write_json(target/'reference_gate_failure.json',failure)
                                                raise RuntimeError('Evaluation diverged from the reset-389 source before the precommitted intervention')
                                            if (decision>int(action_omission['policy_decision_index_0based'])
                                                    and check.get('reference_status')=='compared'
                                                    and first_perturbation_divergence is None
                                                    and not (check['pre_action_ram_match'] and check['policy_input_match_exactly'])):
                                                first_perturbation_divergence=dict(kind='pre_action_state_or_full_policy_input',**check)
                                            elif (first_perturbation_divergence is not None
                                                    and first_perturbation_divergence.get('kind')=='intervention_post_action_ram'
                                                    and decision==int(action_omission['policy_decision_index_0based'])+1
                                                    and check.get('reference_status')=='compared'):
                                                first_perturbation_divergence['next_pre_action_ram_match']=check['pre_action_ram_match']
                                                first_perturbation_divergence['next_policy_input_match_exactly']=check['policy_input_match_exactly']
                                        if not compact_capture:
                                            input_writers[i].write(payload)
                                            input_writers[i].flush()
                                            input_index_writers[i].write(json.dumps(dict(decision_index=input_offsets[i],
                                                episode_decision_index=decision,
                                                episode_native_frame=decision+1,
                                                route_frame=(int(reset_frame)+decision if reset_frame is not None else None),
                                                byte_offset=input_offsets[i]*schema['record_bytes'],
                                                byte_length=len(payload)))+'\n')
                                            input_index_writers[i].flush()
                                            input_offsets[i]+=1
                                if deferred:logits=model.actor(x);values=None
                                else:logits,values=model(x)
                                dist=behavior_distribution(logits,temperature=c.get('policy_temperature',1.))
                                selected=(logits.argmax(-1) if mode=='greedy' else dist.sample()).cpu().numpy()
                                probabilities=dist.probs.cpu().numpy()
                                values=[None]*len(policy_active) if deferred else values.cpu().numpy()
                            for j,i in enumerate(policy_active):
                                proposed_action=int(selected[j])
                                action_by_id[i]=proposed_action
                                probability_by_id[i]=probabilities[j].tolist()
                                value_by_id[i]=None if deferred else float(values[j])
                                proposed_by_id[i]=proposed_action
                                intervention_by_id[i]=None
                                if action_omission is not None:
                                    decision=policy_decision_counts[i]
                                    if decision<=int(action_omission['policy_decision_index_0based']):
                                        reference=(reference_rows_by_id.get(i) if compact_capture
                                            else action_omission['reference_trace'][decision])
                                        if reference is None:
                                            raise RuntimeError('Reference action/probability row is unavailable before intervention')
                                        comparison=compare_reference_greedy_action(reference,proposed_action,
                                            probability_by_id[i],c['action_masks'],
                                            require_same_checkpoint_probabilities=bool(
                                                action_omission.get('same_checkpoint_probabilities_required',False)))
                                        target_action_match=(decision!=int(action_omission['policy_decision_index_0based'])
                                            or (comparison['source_action']==int(action_omission['reference_action_index'])
                                                and comparison['source_mask']==int(action_omission['reference_action_mask'])
                                                and proposed_action==int(action_omission['reference_action_index'])))
                                        comparison['planned_target_action_match']=bool(target_action_match)
                                        comparison['accepted']=bool(comparison['accepted'] and target_action_match)
                                        reference_checks_by_id[i].update(
                                            reference_action_match=comparison['action_match'],
                                            reference_source_action=comparison['source_action'],
                                            reference_source_mask=comparison['source_mask'],
                                            reference_source_probabilities_match_diagnostic=(
                                                comparison['source_probabilities_match']),
                                            reference_source_probability_max_abs_difference=(
                                                comparison['source_probability_max_abs_difference']),
                                            reference_source_action_probabilities=comparison['source_probabilities'],
                                            same_checkpoint_probability_match_required=(
                                                comparison['same_checkpoint_probability_match_required']),
                                            planned_target_action_match=comparison['planned_target_action_match'])
                                        if not comparison['accepted']:
                                            reason=('target_action_mismatch' if not target_action_match else
                                                'greedy_action_mismatch' if not comparison['action_match'] else
                                                'same_checkpoint_probability_mismatch')
                                            failure=dict(status='stopped_before_intervention_unmatched_policy',
                                                condition=action_omission['condition'],
                                                failure_reason=reason,**comparison,
                                                policy_decision_index=decision,decision_1based=decision+1,
                                                planned_intervention_decision=action_omission['decision_1based'],
                                                policy_frames_already_executed=int(frame_counts[i]),
                                                reset_reconstruction_frames=initial_states[i].get('reset_replay_frames'),
                                                initial_state=initial_states[i],
                                                intervention_injected=False)
                                            write_json(target/'reference_gate_failure.json',failure)
                                            raise RuntimeError('Greedy action did not match the precommitted reference intervention')
                                    executed,applied=action_with_one_shot_omission(proposed_action,decision,
                                        action_omission,omission_applied)
                                    if applied:
                                        if decision!=int(action_omission['policy_decision_index_0based']):
                                            raise RuntimeError('Action omission helper changed an unplanned decision')
                                        if proposed_action!=int(action_omission['reference_action_index']):
                                            raise RuntimeError('Proposed action at the intervention differs from precommit')
                                        action_by_id[i]=executed;omission_applied=True
                                        intervention_by_id[i]=dict(condition=action_omission['condition'],
                                            decision_1based=decision+1,policy_decision_index_0based=decision,
                                            pre_action_route_frame=389+decision,
                                            proposed_action=proposed_action,
                                            proposed_mask=int(c['action_masks'][proposed_action]),
                                            executed_action=executed,executed_mask=int(c['action_masks'][executed]),
                                            intervention='one native-frame NOOP')
                        actions=[action_by_id[i] for i in active]
                        results=pool.step_many(actions,active);next_active=[]
                        for i,action,result in zip(active,actions,results):
                            info=result['info'];total_frames+=info['transition_frames'];returns[i]+=result['reward']
                            frame_counts[i]+=int(info['transition_frames']);last_infos[i]=info
                            signature=(tuple(info['frontiers']),info['health'],info['lives'],info['event_totals'].get('objective_cleared',0))
                            if signature!=last_signature.get(i):last_change[i]=info['elapsed_frames'];last_signature[i]=signature
                            stall=(not final and info['elapsed_frames']-last_change[i]>=c['development_stall_frames'])
                            if stall:info=dict(info,end_reason='development_stall_limit')
                            info=dict(info,terminated=bool(result['terminated']),truncated=bool(result['truncated'] or stall))
                            scripted=i in scripted_prefix
                            proposed_action=proposed_by_id.get(i,int(action))
                            intervention=intervention_by_id.get(i)
                            if intervention is not None:
                                reference_row=(reference_rows_by_id.get(i) if compact_capture
                                    else action_omission['reference_trace'][int(action_omission['policy_decision_index_0based'])])
                                reference_post=reference_row.get('ram_sha256') if reference_row else None
                                post_ram_match=(info.get('ram_sha256')==reference_post)
                                intervention['reference_post_action_ram_sha256']=reference_post
                                intervention['actual_post_action_ram_sha256']=info.get('ram_sha256')
                                intervention['post_action_ram_match']=post_ram_match
                                if not post_ram_match and first_perturbation_divergence is None:
                                    first_perturbation_divergence=dict(kind='intervention_post_action_ram',
                                        policy_decision_index=int(action_omission['policy_decision_index_0based']),
                                        decision_1based=int(action_omission['decision_1based']),
                                        reference_ram_sha256=reference_post,actual_ram_sha256=info.get('ram_sha256'),
                                        full_policy_input_comparison='next decision pending')
                            trace_row=dict(action=int(action),mask=c['action_masks'][int(action)],
                                proposed_action=int(proposed_action),proposed_mask=int(c['action_masks'][int(proposed_action)]),
                                executed_action=int(action),executed_mask=c['action_masks'][int(action)],
                                intervention=intervention,
                                reference_pre_action_ram_match=(reference_checks_by_id.get(i) or {}).get('pre_action_ram_match'),
                                reference_policy_input_match_exactly=(reference_checks_by_id.get(i) or {}).get('policy_input_match_exactly'),
                                observation_schema_match=(reference_checks_by_id.get(i) or {}).get('observation_schema_match'),
                                reference_comparison_status=(reference_checks_by_id.get(i) or {}).get('reference_status'),
                                reference_action_match=(reference_checks_by_id.get(i) or {}).get('reference_action_match'),
                                reference_source_action=(reference_checks_by_id.get(i) or {}).get('reference_source_action'),
                                reference_source_mask=(reference_checks_by_id.get(i) or {}).get('reference_source_mask'),
                                reference_source_action_probabilities=(reference_checks_by_id.get(i) or {}).get(
                                    'reference_source_action_probabilities'),
                                reference_source_probabilities_match_diagnostic=(reference_checks_by_id.get(i) or {}).get(
                                    'reference_source_probabilities_match_diagnostic'),
                                reference_source_probability_max_abs_difference=(reference_checks_by_id.get(i) or {}).get(
                                    'reference_source_probability_max_abs_difference'),
                                same_checkpoint_probability_match_required=(reference_checks_by_id.get(i) or {}).get(
                                    'same_checkpoint_probability_match_required'),
                                planned_target_action_match=(reference_checks_by_id.get(i) or {}).get(
                                    'planned_target_action_match'),
                                actual_pre_action_ram_sha256=(reference_checks_by_id.get(i) or {}).get('actual_pre_action_ram_sha256'),
                                reference_expected_pre_action_ram_sha256=(reference_checks_by_id.get(i) or {}).get('expected_pre_action_ram_sha256'),
                                action_source='scripted_noop_prefix' if scripted else 'greedy_policy',
                                policy_decision_index=None if scripted else policy_decision_counts[i],
                                reward=result['reward'],action_probabilities=probability_by_id[i],
                                value=value_by_id[i],**info)
                            if compact_capture:
                                compact_row={key:trace_row.get(key) for key in (
                                    'action','mask','proposed_action','proposed_mask','executed_action','executed_mask',
                                    'intervention','reference_pre_action_ram_match','reference_policy_input_match_exactly',
                                    'reference_comparison_status','reference_action_match','reference_source_action',
                                    'reference_source_mask','reference_source_action_probabilities',
                                    'reference_source_probabilities_match_diagnostic',
                                    'reference_source_probability_max_abs_difference',
                                    'same_checkpoint_probability_match_required','planned_target_action_match',
                                    'observation_schema_match','actual_pre_action_ram_sha256',
                                    'reference_expected_pre_action_ram_sha256','action_source','policy_decision_index',
                                    'reward','action_probabilities','transition_frames','elapsed_frames','terminated',
                                    'truncated','completion','end_reason','health','lives','frontiers',
                                    'objective_flags_raw','score_gain_coverage','reward_components','reward_totals',
                                    'discounted_reward_totals','availability','event_totals','events','bootstrap_valid',
                                    'discounted_return_valid','discounted_base_return_value','discounted_return_value',
                                    'ram_sha256','section','player_x','player_y','reset_frame','reset_replay_frames',
                                    'game_terminal','training_task_terminal')}
                                compact_row['episode_decision_index']=frame_counts[i]-1
                                compact_row['episode_native_frame']=frame_counts[i]
                                compact_row['route_frame']=(int(reset_frame)+frame_counts[i]-1
                                    if reset_frame is not None else None)
                                compact_row['policy_input_fingerprint']=policy_fingerprints_by_id.get(i)
                                compact_row['input_fingerprint_serialization_version']=SERIALIZATION_VERSION
                                compact_row['input_fingerprint_comparison_scope']=(
                                    'exact model inputs through reference-source exhaustion')
                                trace_writers[i].write(compact_row)
                            else:
                                traces[i].append(trace_row)
                                journals[i].write(json.dumps(traces[i][-1])+'\n')
                                journals[i].flush()
                                if deferred and not scripted:
                                    pending.append((traces[i][-1],observation_by_id[i]))
                            if not scripted:policy_decision_counts[i]+=1
                            if i in writers:
                                # Encode elapsed game time correctly for held multi-frame actions.
                                # Screen sampling is at decision endpoints, not invented native screens.
                                copies=info['elapsed_frames']//2-(info['elapsed_frames']-info['transition_frames'])//2
                                if copies:
                                    pool.send(i,'screen',None);screen=pool.receive(i)
                                    for _ in range(copies):writers[i].send(screen)
                            if result['terminated'] or result['truncated'] or stall:
                                if deferred:flush_evaluation_values(model,pending,device,batch=c['batch_size'])
                                initial_state=initial_states[i]
                                initial_state['route_frame']=int(reset_frame) if reset_frame is not None else None
                                row=dict(mode=mode,trial=first+i,return_value=returns[i],
                                    initial_state=initial_state,noop_prefix_frames=noop_prefix_frames,
                                    executed_prefix_frames=(min(noop_prefix_frames,frame_counts[i]) if compact_capture else
                                        sum(r.get('action_source')=='scripted_noop_prefix' for r in traces[i])),takeover_route_frame=(
                                            int(reset_frame)+noop_prefix_frames if reset_frame is not None
                                        and (frame_counts[i]>noop_prefix_frames if compact_capture else
                                             len(traces[i])>noop_prefix_frames) else None),**info)
                                rows.append(row)
                                if compact_capture:
                                    trace_writers[i].flush();trace_writers[i].close()
                                    partial_trace=target/f'{mode}_{first+i}.partial.jsonl.gz'
                                    final_trace=target/f'{mode}_{first+i}.jsonl.gz'
                                    partial_trace.replace(final_trace)
                                    input_schema_path=target/'policy_inputs'/f'{mode}_{first+i}.schema.json'
                                    capture_status=dict(assessment_status='completed_assessment',
                                        completion=bool(info.get('completion')),
                                        terminal_reason=info.get('end_reason','unavailable'),
                                        natural_terminal=bool(info.get('terminated') or info.get('truncated')),
                                        native_frames=int(frame_counts[i]),
                                        policy_decisions=int(policy_decision_counts[i]),
                                        compact_trace_path=str(final_trace.relative_to(ROOT)),
                                        trace_rows=int(trace_writers[i].rows_written),
                                        policy_input_schema_path=str(input_schema_path.relative_to(ROOT)),
                                        serialization_version=SERIALIZATION_VERSION,
                                        comparison_scope='exact model inputs through reference-source exhaustion',
                                        health=info.get('health'),lives=info.get('lives'),
                                        objective_flags_raw=info.get('objective_flags_raw'),
                                        frontiers=info.get('frontiers'),score_gain_coverage=info.get('score_gain_coverage'),
                                        reward_totals=info.get('reward_totals'),availability=info.get('availability'),
                                        validity='valid' if trace_writers[i].rows_written==frame_counts[i] else 'invalid')
                                    write_json(target/f'{mode}_{first+i}.capture_status.json',capture_status)
                                    completed_ids.add(i)
                                    for artifact in policy_input_artifacts:
                                        if artifact.get('mode')==mode and artifact.get('trial')==first+i:
                                            artifact['data_path']=str(final_trace.relative_to(ROOT))
                                else:
                                    write_json(target/f'{mode}_{first+i}.json',traces[i])
                                    journals[i].close();(target/f'{mode}_{first+i}.partial.jsonl').unlink(missing_ok=True)
                                print(f'EVAL {mode} trial{first+i} complete={info["completion"]} frontiers={info["frontiers"]} '
                                      f'return={returns[i]:.2f} frames={info["elapsed_frames"]} reason={info["end_reason"]}',flush=True)
                            else:next_active.append(i);states[i]=result
                        active=next_active
                        if len(pending)>=c['batch_size']:flush_evaluation_values(model,pending,device,batch=c['batch_size'])
                        if time.perf_counter()-last_heartbeat>=15:
                            print(f'EVAL {mode} active={len(active)} measured frames={total_frames:,} '
                                  f'elapsed={(time.perf_counter()-begun)/60:.1f}m',flush=True)
                            last_heartbeat=time.perf_counter()
                except BaseException as exc:
                    if compact_capture:
                        storage_interruption=(isinstance(exc,(StorageGuardInterruption,ArtifactBudgetInterruption))
                            or isinstance(exc,OSError) and getattr(exc,'errno',None)==28)
                        for i in identifiers:
                            if i in completed_ids:
                                continue
                            capture=trace_writers.get(i)
                            if capture is not None:
                                try:
                                    capture.flush();capture.close()
                                except OSError:
                                    pass
                            last_info=last_infos.get(i,{})
                            reason='storage_interruption' if storage_interruption else 'runtime_error'
                            status=partial_capture_status(frames=frame_counts[i],
                                completion=last_info.get('completion'),
                                terminal_reason=reason,interruption=repr(exc),
                                measurements=getattr(exc,'measurements',None))
                            status.update(last_observed_state={key:last_info.get(key) for key in (
                                'health','lives','objective_flags_raw','frontiers','score_gain_coverage',
                                'reward_totals','availability','ram_sha256')},
                                policy_decisions=int(policy_decision_counts[i]),
                                trace_path=str((target/f'{mode}_{first+i}.partial.jsonl.gz').relative_to(ROOT)),
                                trace_rows=(capture.rows_written if capture is not None else 0),
                                serialization_version=SERIALIZATION_VERSION,
                                comparison_scope='exact model inputs through reference-source exhaustion')
                            write_json(target/f'{mode}_{first+i}.capture_status.json',status)
                    else:
                        for i,trace in traces.items():
                            write_json(target/f'{mode}_{first+i}.partial.json',trace)
                    raise
                finally:
                    if reference_input_stream is not None:reference_input_stream.close()
                    if reference_fingerprint_reader is not None:reference_fingerprint_reader.close()
                    for writer in writers.values():writer.close()
                    for writer in trace_writers.values():
                        if not writer.closed:
                            try:writer.close()
                            except OSError:pass
                    for journal in journals.values():
                        if not journal.closed:journal.close()
                    for writer in input_writers.values():
                        if not writer.closed:writer.close()
                    for writer in input_raw_writers.values():
                        if not writer.closed:writer.close()
                    for writer in input_index_writers.values():
                        if not writer.closed:writer.close()
        default_evaluation_seed=(c['seed']+int(frames) if frames is not None else None)
        report=dict(training_frames=frames,runs=rows,evaluation_frames=total_frames,wall_seconds=time.perf_counter()-begun,
            evaluation_seed_base=(c['evaluation_seed'] if c.get('evaluation_seed') is not None
                else default_evaluation_seed),
            final=final,reset_frame=reset_frame,no_reset_jitter=no_reset_jitter,
            evaluation_horizon_frames=evaluation_horizon_frames,all_suffix_starts=all_suffix_starts,
            noop_prefix_frames=noop_prefix_frames,
            evaluation_id=evaluation_id,
            action_prefix=(dict(kind='normal_game_noop',mask=0,native_frames=noop_prefix_frames)
                if noop_prefix_frames else None),
            independent_human_trajectories=1,development=not final,
            full_start_horizon_frames=c['full_horizon'] if reset_frame is None else None,
            development_stall_limit_frames=c['development_stall_frames'] if not final and c['development_stall_frames']<=c['full_horizon'] else None,
            sampled_completions=sum(r['completion'] for r in rows if r['mode']=='sampled'),
            sampled_trials=sum(r['mode']=='sampled' for r in rows),
            greedy_completions=sum(r['completion'] for r in rows if r['mode']=='greedy'),
            greedy_distinct_starts=sum(r['mode']=='greedy' for r in rows),
            action_omission=(None if action_omission is None else dict(
                condition=action_omission['condition'],decision_1based=action_omission['decision_1based'],
                policy_decision_index_0based=action_omission['policy_decision_index_0based'],
                pre_action_route_frame=action_omission['pre_action_route_frame'],
                reference_action_index=action_omission['reference_action_index'],
                reference_action_mask=action_omission['reference_action_mask'],
                executed_action_index=action_omission['executed_action_index'],
                executed_action_mask=action_omission['executed_action_mask'],
                applied=omission_applied,first_perturbation_divergence=first_perturbation_divergence,
                source_trace_sha256=action_omission['source_trace_sha256'])),
            policy_input_artifacts=policy_input_artifacts,
            checkpoint_provenance=checkpoint_provenance)
        write_json(target/'summary.json',report);log(folder/'evaluations.jsonl',report)
        return report
    finally:
        if preserve_pool_state:pool.restore_many()
        torch.set_rng_state(rng_saved);torch.cuda.set_rng_state_all(cuda_saved)


def curves(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=folder/'updates.jsonl'
    if not path.exists():return
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    fig,axes=plt.subplots(4,2,figsize=(12,13));x=[r['frames'] for r in rows]
    for ax,key in zip(list(axes.flat)[:6],('remaining_demo_seconds','frontier_success_rate','recent_training_return','entropy','approximate_kl','explained_variance')):
        y=[r.get(key,np.nan) if (key!='frontier_success_rate' or r['frontier_episodes']>=20)
           and (key!='entropy' or not r.get('warmup',False)) else np.nan for r in rows]
        title={'frontier_success_rate':'Frontier success (at least20 episodes)',
               'recent_training_return':'Mixed-start training return (not evaluation)'}.get(key,key)
        ax.plot(x,y);ax.set_title(title);ax.set_xlabel('lineage training emulator frames');ax.grid(alpha=.3)
    path=folder/'evaluations.jsonl'
    evaluations=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    suffix_evaluations=[e for e in evaluations if e.get('reset_frame') is not None]
    evaluations=[e for e in evaluations if e.get('reset_frame') is None]
    for mode in ('sampled','greedy'):
        selected=[[r for r in e['runs'] if r['mode']==mode] for e in evaluations]
        xe=[e['training_frames'] for e in evaluations]
        axes[3,0].plot(xe,[np.mean([r['completion'] for r in rs]) for rs in selected],marker='o',label=mode)
        axes[3,1].plot(xe,[np.mean([r['return_value'] for r in rs]) for rs in selected],marker='o',label=mode)
    for ax,title in zip(axes[3],('Full-start development completion fraction','Full-start development return')):
        ax.set_title(title);ax.set_xlabel('lineage training emulator frames');ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(folder/'learning_curves.png');plt.close(fig)
    if suffix_evaluations:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for reset in sorted({e['reset_frame'] for e in suffix_evaluations}):
            selected=[e for e in suffix_evaluations if e['reset_frame']==reset]
            for column,mode in enumerate(('sampled','greedy')):
                runs=[[r for r in e['runs'] if r['mode']==mode] for e in selected]
                xx=[e['training_frames'] for e in selected]
                axes[0,column].plot(xx,[np.mean([r['completion'] for r in rs]) for rs in runs],marker='o',label=f'reset {reset}')
                axes[1,column].plot(xx,[np.mean([r['return_value'] for r in rs]) for rs in runs],marker='o',label=f'reset {reset}')
                axes[0,column].set_title(f'{mode}: fixed suffix completion')
                axes[1,column].set_title(f'{mode}: fixed suffix return')
        for ax in axes.flat:ax.set_xlabel('lineage training emulator frames');ax.grid(alpha=.3);ax.legend()
        fig.tight_layout();fig.savefig(folder/'suffix_learning_curves.png');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('train','evaluate','benchmark'),default='train')
    p.add_argument('--config',type=Path,default=ROOT/'config/backward_level1_v1.json')
    p.add_argument('--output',type=Path,default=OUT/'run01')
    p.add_argument('--observer-diagnostics-output',type=Path,
        help='Runtime-only directory for worker observer failure artifacts')
    budget=p.add_mutually_exclusive_group()
    budget.add_argument('--frames',type=int,help='Additional fresh emulator frames in this bounded block (default 500000 when no budget is supplied)')
    budget.add_argument('--updates',type=int,help='Additional outer PPO collection/update cycles; each cycle collects one rollout and performs its configured optimization passes')
    p.add_argument('--resume',type=Path);p.add_argument('--trials',type=int,default=20)
    p.add_argument('--final',action='store_true');p.add_argument('--record',action='store_true');p.add_argument('--skip-initial-evaluation',action='store_true')
    p.add_argument('--reset-frame',type=int,help='Evaluation-only suffix start; never counted as full-start performance')
    p.add_argument('--all-suffix-starts',action='store_true',help='Evaluate greedy once at EVERY valid native-frame start in suffix jitter range; sampled count still --trials')
    p.add_argument('--curriculum-adaptation',action='store_true',help='Explicitly fork into a NEW output with changed coverage anchors only')
    p.add_argument('--normalization-adaptation',action='store_true',help='Explicit NEW-output compatible-weight branch normalization fork; reset optimizer and rewarm critic')
    p.add_argument('--timing-adaptation',action='store_true',help='Explicit NEW-output held-native-frame timing comparison; preserve weights and optimizer')
    p.add_argument('--reward-adaptation',action='store_true',help='Explicit new-output baseline -> tested potential-reward experiment')
    p.add_argument('--scoped-kind18-reward-resume',action='store_true',help='Permit only ABI2 observer activation and the 0/.25 scoped kind18 defeat coefficient; learning state is preserved')
    p.add_argument('--score-reward-adaptation',action='store_true',help='Activate the tested displayed-score reward while preserving resumed worker, curriculum, RNG, and actor state')
    p.add_argument('--adaptation-validation-only',action='store_true',help='Restore and validate an explicit adaptation plan, then exit before creating workers or training')
    p.add_argument('--evaluation-preparation-only',action='store_true',help='Validate a fixed suffix evaluation through checkpoint/config preparation, then exit before emulator or worker creation')
    p.add_argument('--evaluation-action-omission-plan',type=Path,
        help='Evaluation only: one precommitted normal-game one-frame action omission plan')
    p.add_argument('--evaluation-action-omission-condition',choices=('A','B','C'),
        help='Condition identifier within --evaluation-action-omission-plan')
    p.add_argument('--allow-evaluation-provenance-difference',action='store_true',
        help='Evaluation only: permit a different checkpoint training-run label when its model-config identity and full runtime contract match')
    p.add_argument('--allow-precommitted-actor-difference',action='store_true',
        help='Evaluation only: apply the saved intervention/reference mask to a trained endpoint while preserving the precommit and runtime contract')
    p.add_argument('--reset-resumed-workers',action='store_true',help='Discard saved live worker episodes and reset using configured roles; preserve model, optimizer, RNG, and curriculum')
    p.add_argument('--bc-initialized-extension',action='store_true',help='Explicitly labelled BC initialization/auxiliary extension; never primary PPO-only')
    p.add_argument('--no-stall-limit',action='store_true',help='Evaluation only: keep the full episode horizon without the development stall cutoff')
    p.add_argument('--evaluation-seed',type=int,help='Evaluation-only common random seed base, independent of checkpoint training counter')
    p.add_argument('--evaluation-id',help='Evaluation-only safe identifier for output naming when no training frame counter is recorded')
    p.add_argument('--no-reset-jitter',action='store_true',help='Evaluation only: use exactly --reset-frame with no suffix jitter')
    p.add_argument('--evaluation-horizon-frames',type=int,help='Evaluation only: fixed native-frame episode horizon')
    p.add_argument('--evaluation-mode',choices=('sampled','greedy'),help='Evaluation only: run exactly one policy mode')
    p.add_argument('--serial-evaluation',action='store_true',
        help='Evaluation only: use one CPU environment worker')
    p.add_argument('--save-evaluation-policy-inputs',action='store_true',
        help='Save the encoded per-decision observation windows supplied to the evaluator policy')
    p.add_argument('--compact-evaluation-capture',action='store_true',
        help='Stream compact action/RAM/reward traces and per-field policy-input fingerprints')
    p.add_argument('--reference-policy-input-fingerprints',type=Path,
        help='Verified offline fingerprint manifest for the saved reference policy inputs')
    p.add_argument('--evaluation-artifact-root',type=Path,
        help='Shared new-artifact root used by compact capture budget checks')
    p.add_argument('--evaluation-artifact-budget-mib',type=int,default=DEFAULT_ARTIFACT_BUDGET_BYTES//(1024*1024),
        help='Maximum new artifact budget for compact evaluation (MiB)')
    p.add_argument('--evaluation-artifact-remaining-mib',type=int,
        help='Remaining portion of the shared artifact cap available to this serial episode')
    p.add_argument('--evaluation-minimum-free-mib',type=int,default=DEFAULT_RESERVE_BYTES//(1024*1024),
        help='Minimum disk free-space reserve enforced during compact evaluation (MiB)')
    p.add_argument('--evaluation-temp-overhead-mib',type=int,default=DEFAULT_TEMP_OVERHEAD_BYTES//(1024*1024),
        help='Measured temporary-file allowance included in compact evaluation preflight (MiB)')
    p.add_argument('--delayed-takeover-noop-prefix-frames',type=int,choices=(1,2,4,8,16),
        help='Isolated exact-reset evaluation: advance 1, 2, 4, 8, or 16 normal-game NOOP frames before greedy takeover')
    p.add_argument('--stop-on-frontier-mastery',action='store_true',
        help='Finish and checkpoint the current PPO update, then stop after the first earlier-frontier mastery transition')
    args=p.parse_args()
    if (args.evaluation_action_omission_plan is None)!=(args.evaluation_action_omission_condition is None):
        p.error('--evaluation-action-omission-plan and --evaluation-action-omission-condition must be supplied together')
    if args.evaluation_action_omission_plan is not None and args.stage!='evaluate':
        p.error('Action-omission overrides require --stage evaluate')
    if args.allow_evaluation_provenance_difference and args.stage!='evaluate':
        p.error('--allow-evaluation-provenance-difference requires --stage evaluate')
    if args.evaluation_id is not None and args.stage!='evaluate':
        p.error('--evaluation-id requires --stage evaluate')
    if args.adaptation_validation_only and (args.resume is None or
            not (args.curriculum_adaptation or args.score_reward_adaptation) or args.stage!='train'):
        p.error('--adaptation-validation-only requires a training resume with an explicit curriculum or score adaptation flag')
    if args.evaluation_preparation_only and (args.resume is None or args.stage!='evaluate'):
        p.error('--evaluation-preparation-only requires --stage evaluate and --resume')
    if args.stage=='evaluate' and args.resume is None:p.error('--stage evaluate requires --resume')
    if args.frames is None and args.updates is None:args.frames=500000
    if args.updates is not None and args.stage!='train':p.error('--updates is available only for --stage train')
    c=json.loads(args.config.read_text());folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    if args.scoped_kind18_reward_resume and (args.reward_adaptation or args.score_reward_adaptation):
        raise ValueError('Scoped kind18 reward resume is separate from other reward adaptations')
    if args.reward_adaptation and args.score_reward_adaptation:
        raise ValueError('Potential and score reward adaptations are separate')
    if args.reset_resumed_workers:
        roles=c.get('collector_roles',[])
        if (args.stage!='train' or args.resume is None or not args.scoped_kind18_reward_resume
                or len(roles)!=c.get('environments') or any(role!='full_start' for role in roles)):
            raise ValueError('--reset-resumed-workers requires a training resume with scoped kind18 validation and all collector_roles=full_start')
    if args.evaluation_seed is not None:
        if args.stage!='evaluate':raise ValueError('Evaluation seed override may not change training')
        c['evaluation_seed']=args.evaluation_seed
    if args.no_stall_limit:
        if args.stage!='evaluate':raise ValueError('--no-stall-limit requires --stage evaluate')
        c['development_stall_frames']=c['full_horizon']+1
    if args.no_reset_jitter and (args.stage!='evaluate' or args.reset_frame is None):
        raise ValueError('--no-reset-jitter requires suffix evaluation')
    if args.evaluation_horizon_frames is not None and (args.stage!='evaluate' or args.evaluation_horizon_frames<1):
        raise ValueError('--evaluation-horizon-frames requires evaluation and a positive horizon')
    if args.evaluation_mode is not None and args.stage!='evaluate':
        raise ValueError('--evaluation-mode requires --stage evaluate')
    if args.serial_evaluation and args.stage!='evaluate':
        p.error('--serial-evaluation requires --stage evaluate')
    if args.compact_evaluation_capture and (args.stage!='evaluate' or not args.save_evaluation_policy_inputs):
        p.error('--compact-evaluation-capture requires evaluation and --save-evaluation-policy-inputs')
    if args.reference_policy_input_fingerprints is not None and not args.compact_evaluation_capture:
        p.error('--reference-policy-input-fingerprints requires --compact-evaluation-capture')
    if args.compact_evaluation_capture and args.evaluation_artifact_budget_mib<1:
        p.error('--evaluation-artifact-budget-mib must be positive')
    if args.compact_evaluation_capture and args.evaluation_artifact_remaining_mib is not None and (
            args.evaluation_artifact_remaining_mib<0
            or args.evaluation_artifact_remaining_mib>args.evaluation_artifact_budget_mib):
        p.error('--evaluation-artifact-remaining-mib must be within the shared artifact cap')
    if args.compact_evaluation_capture and (args.evaluation_minimum_free_mib<1
            or args.evaluation_temp_overhead_mib<0):
        p.error('Compact evaluation storage reserve must be positive and temporary overhead nonnegative')
    args.evaluation_id=validate_evaluation_identifier(args.evaluation_id)
    if args.delayed_takeover_noop_prefix_frames is not None and (
            args.stage!='evaluate' or args.reset_frame is None or not args.no_reset_jitter
            or args.trials!=1 or args.evaluation_mode!='greedy' or args.record
            or c.get('frame_skip')!=1):
        raise ValueError('--delayed-takeover-noop-prefix-frames requires one greedy frame_skip=1 episode at an exact suffix reset, without recording')
    validate_bc_mode(c,args.bc_initialized_extension,args.stage)
    if c.get('exploration_epsilon_schedule') and not c.get('defer_critic'):raise ValueError('Mixture-policy collection requires the verified deferred critic path')
    torch.set_num_threads(1)
    checkpoint_preparation=args.adaptation_validation_only or args.evaluation_preparation_only or args.stage=='evaluate'
    device=torch.device('cpu') if args.adaptation_validation_only or args.evaluation_preparation_only else cuda_device()
    rng=np.random.default_rng(c['seed']);random.seed(c['seed']);torch.manual_seed(c['seed'])
    route=HumanRoute(config=c)
    if checkpoint_preparation:
        model,payload=load_temporal(args.resume,device);prior=payload['metadata']
        shapes=model.shapes
    else:
        prototype=Level1Env(c,route);obs,_=prototype.reset();prototype.close()
        shapes=infer_shapes(obs)
    evaluation_provenance=None
    if args.resume:
        if not checkpoint_preparation:
            model,payload=load_temporal(args.resume,device);prior=payload['metadata']
        # load_temporal returns the checkpoint payload as well as the model;
        # the model already owns copied parameters, so don't retain another
        # checkpoint-sized state_dict for the duration of training.
        del payload
        for key in ('route_guidance_version','route_guidance_data','route_guidance_sha256','route_waypoint_version','route_waypoint_data','route_waypoint_sha256','forward_curriculum_version','forward_manifest_sha256','forward_stage_index','forward_lookahead_zones','forward_support_sha256'):
            if model.config.get(key)!=c.get(key):raise ValueError('Incompatible route-guidance resume: '+key)
        if model.config.get('network_precision','legacy_default')!=c.get('network_precision','legacy_default'):raise ValueError('Explicit precision-fork checkpoint required')
        if prior.get('bc_initialization') and not args.bc_initialized_extension:raise ValueError('BC-initialized source requires explicit extension mode')
        if args.timing_adaptation:
            if args.stage not in ('train','evaluate') or args.reward_adaptation or args.curriculum_adaptation or args.normalization_adaptation or folder.resolve()==args.resume.resolve().parent.parent or any(folder.iterdir()):
                raise ValueError('Timing adaptation requires a NEW empty output and no other adaptation')
            timing_config=dict(c)
            if args.no_stall_limit:timing_config['development_stall_frames']=model.config['development_stall_frames']
            validate_timing_adaptation(model.config,timing_config)
            prior=dict(prior)
            prior.update(environment_states=[],cases=[],episode_returns={},recent=[],
                         best_full=(-1,-1,-1.),best_return=-1e9,best_suffix={},
                         timing_adaptation=dict(source=str(args.resume),old_frame_skip=model.config['frame_skip'],
                             new_frame_skip=c['frame_skip'],weights_optimizer_rng_preserved=True,
                             native_discount_unchanged=True,live_segments_reset=True))
            prior['curriculum']=copy.deepcopy(prior['curriculum']);prior['curriculum']['outcomes']=[]
            prior['curriculum']['stage']+=1;prior['curriculum']['stage_start_frames']=prior['frames']
        elif model.config['frame_skip']!=c['frame_skip']:
            raise ValueError('Held-action interval differs; use explicit NEW-output --timing-adaptation')
        if model.config['gamma']!=c['gamma'] or model.config['gae_lambda']!=c['gae_lambda']:
            raise ValueError('Native discount or GAE units differ; an explicit versioned experiment is required')
        if args.normalization_adaptation:
            if args.stage!='train' or args.reward_adaptation or args.curriculum_adaptation or folder.resolve()==args.resume.resolve().parent.parent or any(folder.iterdir()):
                raise ValueError('Normalization adaptation requires training in a NEW empty output')
            validate_normalization_adaptation(model.config,c)
            converted=TemporalActorCritic(shapes,len(c['action_masks']),c).to(device)
            converted.load_state_dict(model.state_dict(),strict=True)
            model=converted
            prior=dict(prior)
            prior.update(environment_states=[],cases=[],episode_returns={},recent=[],
                         best_full=(-1,-1,-1.),best_return=-1e9,best_suffix={},best_frontier=route.end,
                         critic_rewarmup_until=prior['frames']+c.get('normalization_rewarmup_frames',16384),
                         normalization_adaptation=dict(source=str(args.resume),compatible_parameters_only=True,
                             policy_function_changed=True,optimizer_moments_reset=True,live_segments_reset=True,
                             inherited_frontier=prior['curriculum']['frontier']))
            prior['curriculum']=copy.deepcopy(prior['curriculum']);prior['curriculum']['outcomes']=[]
            prior['curriculum']['stage']+=1;prior['curriculum']['stage_start_frames']=prior['frames']
        if args.curriculum_adaptation and (args.stage!='train' or args.reward_adaptation
                or folder.resolve()==args.resume.resolve().parent.parent or any(folder.iterdir())):
            raise ValueError('Curriculum adaptation requires training in a NEW empty output')
        if args.score_reward_adaptation and (args.stage not in ('train','evaluate') or (args.stage=='train'
                and (folder.resolve()==args.resume.resolve().parent.parent or any(folder.iterdir())))):
            raise ValueError('Score reward adaptation requires training in a NEW empty output directory or explicit evaluation')
        adaptation_plan=None
        config_paths=configuration_diff_paths(model.config,c)
        has_score_or_curriculum_changes=any(path.split('.',1)[0] in CURRICULUM_ADAPTATION_FIELDS
            or path=='weights.score_gain' for path in config_paths)
        if (args.curriculum_adaptation or args.score_reward_adaptation or has_score_or_curriculum_changes
                or (args.stage=='evaluate' and 'evaluation_seed' in config_paths)):
            adaptation_plan=validate_adaptation_composition(model.config,c,
                curriculum=args.curriculum_adaptation,score_reward=args.score_reward_adaptation,route=route,
                allow_evaluation_seed=args.stage=='evaluate' and args.evaluation_seed is not None,
                allow_evaluation_fallback_metadata=args.stage=='evaluate',
                allow_evaluation_collector_roles=args.stage=='evaluate',
                allow_evaluation_no_stall_limit=args.stage=='evaluate' and args.no_stall_limit,
                allow_evaluation_training_metadata=(args.stage=='evaluate'
                    and args.allow_evaluation_provenance_difference))
        if args.curriculum_adaptation:
            differences=adaptation_plan['curriculum_changes']
            prior=dict(prior)
            fallback_only=differences<= {'initial_failure_fallback_starts','initial_fallback_min_frames'}
            adaptation=dict(source=str(args.resume),changes=sorted(differences),
                config_changes=sorted(adaptation_plan['changes']),
                previous_coverage=model.config.get('coverage_reset_frames'),new_coverage=c.get('coverage_reset_frames'),
                optimizer_and_rng_preserved=True,live_segments_reset=not fallback_only,
                failure_history_preserved=fallback_only)
            if fallback_only:
                prior['curriculum']=copy.deepcopy(prior['curriculum'])
                old_candidates=prior['curriculum'].get('initial_fallback_candidates')
                if old_candidates is None:
                    # Legacy scheduler snapshots did not serialize fallback state.
                    # Explicit adaptation opts this run into the configured list;
                    # the missing historical failure-frame counter starts at zero.
                    prior['curriculum']['initial_failure_frames']=0
                    prior['curriculum']['initial_fallback_cursor']=0
                    prior['curriculum']['initial_fallback_exhausted']=False
                prior['curriculum']['initial_fallback_candidates']=list(c.get('initial_failure_fallback_starts',[]))
                adaptation.update(live_worker_snapshots_preserved=bool(prior.get('environment_states')),
                    cases_preserved=bool(prior.get('cases')),outcomes_preserved=True,
                    prior_fallback_candidates=old_candidates,
                    new_fallback_candidates=prior['curriculum']['initial_fallback_candidates'],
                    prior_initial_failure_frames=prior['curriculum'].get('initial_failure_frames',0),
                    fallback_cursor=prior['curriculum']['initial_fallback_cursor'],
                    fallback_exhausted=prior['curriculum']['initial_fallback_exhausted'])
            elif differences=={'collector_roles'}:
                prior=reset_collector_role_worker_state(prior,c['environments'],c['collector_roles'],args.resume)
                adaptation.update(failure_history_preserved=True,curriculum_outcomes_preserved=True,
                    curriculum_stage_preserved=True,
                    discarded_live_case_kinds=prior['matched_worker_reset']['discarded_case_kind_counts'],
                    saved_worker_snapshots_discarded=prior['matched_worker_reset']['saved_worker_snapshots_discarded'],
                    new_worker_resets='each worker samples a fresh case using its configured collector role')
            else:
                prior.update(environment_states=[],cases=[],episode_returns={},recent=[])
                prior['curriculum']=copy.deepcopy(prior['curriculum']);prior['curriculum']['outcomes']=[]
                prior['curriculum']['stage']+=1;prior['curriculum']['stage_start_frames']=prior['frames']
            prior['curriculum_adaptation']=adaptation
        evaluation_provenance=validate_checkpoint_experiment_provenance(prior,model.config,c,route,
            stage=args.stage,
            allow_evaluation_provenance_difference=args.allow_evaluation_provenance_difference)
        if args.scoped_kind18_reward_resume:
            validate_scoped_kind18_reward_resume(model.config,c,allow_evaluation_seed=args.stage=='evaluate',
                allow_collector_roles=args.reset_resumed_workers or args.stage=='evaluate')
            prior=dict(prior,scoped_kind18_reward_adaptation=dict(source=str(args.resume),
                old_weight=float(model.config.get('weights',{}).get('abi2_kind18_defeat',0.)),
                new_weight=float(c['weights'].get('abi2_kind18_defeat',0.)),
                optimizer_curriculum_rng_preserved=True,
                environment_states_preserved=not args.reset_resumed_workers))
        for key in ('history_offsets','feature_schema_version','action_masks','mlp_sizes','cnn_channels','cnn_projection','gru_hidden','weights'):
            if key=='weights' and (args.reward_adaptation or args.score_reward_adaptation or args.scoped_kind18_reward_resume):continue
            if model.config[key]!=c[key]:raise ValueError('Incompatible resumed interface: '+key)
        for key in ('control_edge_features','control_age_cap_frames','explicit_map_identity','verified_counter_feature',
                    'vanilla_defense_codes','collector_roles','coverage_reset_frames','truncate_suffix_on_life_loss','branch_feature_normalization','episodic_life_training','policy_temperature','section0_timeout_terminal','terminal_version',
                    'policy_recovery_sha256','policy_recovery_ids','policy_recovery_fraction','forward_earlier_reset_frames',
                    'initial_failure_fallback_starts','initial_fallback_min_frames'):
            if key=='branch_feature_normalization' and args.normalization_adaptation:continue
            if key=='coverage_reset_frames' and args.curriculum_adaptation:continue
            if key=='collector_roles' and args.curriculum_adaptation:continue
            if key in INITIAL_FALLBACK_FIELDS and args.curriculum_adaptation:continue
            if key=='collector_roles' and args.reset_resumed_workers:continue
            if args.stage=='evaluate' and key in ('collector_roles','coverage_reset_frames','truncate_suffix_on_life_loss',
                    'episodic_life_training',*INITIAL_FALLBACK_FIELDS):continue
            if key in INITIAL_FALLBACK_FIELDS and ((key in model.config)!=(key in c)):
                raise ValueError('Incompatible resumed curriculum: '+key)
            if model.config.get(key)!=c.get(key):raise ValueError('Incompatible resumed task/history: '+key)
        if not args.reward_adaptation and (model.config.get('reward_version','level1_task_v1')!=c.get('reward_version','level1_task_v1')
                or model.config.get('final_encounter_potential',False)!=c.get('final_encounter_potential',False)):
            raise ValueError('Reward behavior differs; use explicit new-output --reward-adaptation')
        if args.score_reward_adaptation:
            if args.stage=='train':
                prior=dict(prior)
                prior['score_reward_adaptation']=dict(source=str(args.resume),old_weight=0.0,
                    config_changes=sorted(adaptation_plan['score_changes']),
                    new_weight=0.00005,actor_hash=parameter_hash(model.actor),
                    actor_optimizer_preserved=True,rng_and_counters_preserved=True,
                    worker_snapshots_preserved=True,
                    score_baseline='decoded from each restored current RAM; no retroactive reward',
                    critic_optimizer_reset=True,critic_rewarmup_frames=c.get('critic_rewarmup_frames',16384))
        if args.reward_adaptation:
            if args.stage!='train' or folder.resolve()==args.resume.resolve().parent.parent or any(folder.iterdir()):
                raise ValueError('Reward adaptation requires training in a NEW output directory')
            validate_reward_adaptation(model.config,c)
            prior=dict(prior)
            adaptation=dict(source=str(args.resume),old_reward=model.config.get('reward_version','level1_task_v1'),
                new_reward=c['reward_version'],actor_hash=parameter_hash(model.actor),critic_optimizer_reset=True,
                episodes_restarted=True,no_parameter_transfer_from_archived_experiments=True)
            prior.update(reward_adaptation=adaptation,environment_states=[],cases=[],episode_returns={},recent=[],
                best_full=(-1,-1,-1.),best_return=-1e9,best_suffix={},evaluation_frames=0,
                last_evaluation_frames=prior['frames'],critic_rewarmup_until=prior['frames']+c.get('critic_rewarmup_frames',16384))
            prior['curriculum']=copy.deepcopy(prior['curriculum']);prior['curriculum']['outcomes']=[]
            prior['curriculum']['stage']+=1;prior['curriculum']['stage_start_frames']=prior['frames']
        if args.reset_resumed_workers:
            prior=reset_resumed_worker_state(prior,c['environments'],c['collector_roles'],args.resume,route.start)
        if args.stage=='evaluate' and args.allow_evaluation_provenance_difference:
            evaluation_provenance['validated_model_contract']=validate_evaluation_checkpoint_contract(
                model,model.config,c,route,args.reset_frame if args.reset_frame is not None else route.start)
        model.config=c.copy()
    else:
        if args.timing_adaptation:raise ValueError('Timing adaptation requires a preserved source checkpoint')
        if args.normalization_adaptation:raise ValueError('Normalization adaptation requires a preserved source checkpoint')
        if args.curriculum_adaptation:raise ValueError('Curriculum adaptation requires a preserved source checkpoint')
        if args.score_reward_adaptation:raise ValueError('Score reward adaptation requires a preserved source checkpoint')
        if args.reward_adaptation:raise ValueError('Reward adaptation needs a baseline source checkpoint')
        if args.stage=='train' and (folder/'checkpoints/latest.pt').exists():raise ValueError('Use --resume or a NEW output directory')
        model=TemporalActorCritic(shapes,len(c['action_masks']),c).to(device);prior={}
    if args.stage=='evaluate':
        # Evaluation consumes only provenance labels and the displayed training
        # counters. Do not retain optimizer, RNG, curriculum, or worker state.
        prior={key:prior[key] for key in ('experiment','route_sha256','frames','updates','bc_initialization')
               if key in prior}
        evaluation_training_frames=optional_training_counter(prior,'frames')
        evaluation_training_updates=optional_training_counter(prior,'updates')
        if evaluation_training_frames is None and args.evaluation_id is None:
            raise ValueError('Checkpoint has no recorded training frame counter; --evaluation-id is required for output naming')
        if evaluation_training_frames is None and c.get('evaluation_seed') is None:
            raise ValueError('Checkpoint has no recorded training frame counter; provide an explicit --evaluation-seed')
    if args.adaptation_validation_only:
        result=adaptation_validation_result(model,prior,c,adaptation_plan)
        print(json.dumps(result),flush=True)
        return
    if args.evaluation_preparation_only:
        if evaluation_provenance is not None:
            write_json(folder/'checkpoint_provenance.json',evaluation_provenance)
        result=evaluation_preparation_result(args,c,route,args.resume,evaluation_provenance,prior)
        print(json.dumps(result),flush=True)
        return
    action_omission=validate_action_omission_plan(args,c)
    if args.compact_evaluation_capture and not args.evaluation_preparation_only:
        artifact_root=args.evaluation_artifact_root or folder
        remaining_budget=(args.evaluation_artifact_remaining_mib
            if args.evaluation_artifact_remaining_mib is not None
            else args.evaluation_artifact_budget_mib)
        storage_preflight=assess_storage(
            [artifact_root,tempfile.gettempdir()],
            artifact_budget_bytes=int(args.evaluation_artifact_budget_mib)*1024*1024,
            reserve_bytes=int(args.evaluation_minimum_free_mib)*1024*1024,
            temp_overhead_bytes=int(args.evaluation_temp_overhead_mib)*1024*1024)
        write_json(folder/'evaluation_storage_preflight.json',storage_preflight)
        if storage_preflight['status']!='passed':
            raise RuntimeError('Compact evaluation storage preflight failed; no environment workers or gameplay started')
    if args.stage=='evaluate' and evaluation_provenance is not None:
        write_json(folder/'checkpoint_provenance.json',evaluation_provenance)
    print(json.dumps(dict(stage='initialize',fresh=not bool(args.resume),counts=model.counts(),device=str(device),
        architecture=model.architecture,numerical_precision=configure_precision(c),normalization_adaptation=args.normalization_adaptation,
        timing_adaptation=args.timing_adaptation,frame_skip=c['frame_skip'],policy_temperature=c.get('policy_temperature',1.),episodic_life_training=c.get('episodic_life_training',False),
        route_sha256=route.meta['route_sha256'],actions=c['action_masks'],bc_enabled=bool(c.get('bc_enabled',False)),bc_initialization=bool(c.get('bc_initialization',False)),training_method=c.get('training_method','primary_PPO_only'),
        reward_version=c.get('reward_version','level1_task_v1'),reward_adaptation=args.reward_adaptation)),flush=True)
    if args.stage!='evaluate':
        with torch.no_grad():model(window_tensors(obs,device))
    if args.stage!='evaluate':
        assert model.actor.last_hidden_device==model.critic.last_hidden_device==str(device)
    assert not ({id(x) for x in model.actor.parameters()} & {id(x) for x in model.critic.parameters()})
    if args.stage=='benchmark':
        reports=[]
        for n in (2,4,6):
            pool=ProcessEnvs(n,c,c['history_offsets'],child_target=level1_child)
            try:
                states=pool.reset_many([dict(frame=route.end-360,horizon=128*c['frame_skip']+1) for _ in range(n)]);start=time.perf_counter();measured_frames=0
                for _ in range(128):
                    observations=[s['obs'] for s in states]
                    if c.get('defer_critic',False):
                        with torch.no_grad():
                            x=window_tensors({k:np.stack([o[k] for o in observations]) for k in model.window_keys},device,batch=True)
                            acts=behavior_distribution(model.actor(x),temperature=c.get('policy_temperature',1.)).sample().cpu().numpy()
                    else:
                        with torch.no_grad():
                            x=window_tensors({k:np.stack([o[k] for o in observations]) for k in model.window_keys},device,batch=True)
                            logits,_=model(x);acts=behavior_distribution(logits,temperature=c.get('policy_temperature',1.)).sample().cpu().numpy()
                    states=pool.step_many(acts)
                    measured_frames+=sum(s['info']['transition_frames'] for s in states)
                    ended=[i for i,s in enumerate(states) if s['terminated'] or s['truncated']]
                    if ended:
                        replacements=pool.reset_many([dict(frame=route.end-360,horizon=128*c['frame_skip']+1) for _ in ended],ended)
                        for i,s in zip(ended,replacements):states[i]=s
                report=dict(environments=n,frames=measured_frames,decisions=n*128,seconds=time.perf_counter()-start,torch_in_cpu_owners=[s['torch_imported'] for s in pool.owner_metadata])
                report['fps']=report['frames']/report['seconds'];reports.append(report);print(json.dumps(report),flush=True)
            finally:pool.close()
        write_json(folder/'throughput.json',reports);return
    optimizer=None;curriculum=None
    if args.stage!='evaluate':
        optimizer=torch.optim.Adam([dict(params=model.actor.parameters(),lr=c['actor_learning_rate']),
                                    dict(params=model.critic.parameters(),lr=c['critic_learning_rate'])],eps=1e-6)
        if c.get('forward_curriculum_version'):
            from prince_valiant_rl.forward_curriculum import ForwardExposureCurriculum
            curriculum=ForwardExposureCurriculum(c,route.valid,route.start,route.end)
        else:curriculum=BackwardCurriculum(c,route.valid,route.start,route.end)
        if prior:
            optimizer.load_state_dict(prior['optimizer']);curriculum.load(prior['curriculum'])
            if args.normalization_adaptation:optimizer.state.clear()
            if args.reward_adaptation or (args.score_reward_adaptation and args.stage=='train'):
                for parameter in model.critic.parameters():optimizer.state.pop(parameter,None)
            # An explicit config change is a logged corrective experiment, not hidden resume drift.
            optimizer.param_groups[0]['lr']=c['actor_learning_rate'];optimizer.param_groups[1]['lr']=c['critic_learning_rate']
            load_rng(rng,prior['rng'])
    if c.get('combat_observer_diagnostics',False):
        c['_observer_failure_dir']=str(args.observer_diagnostics_output or folder/'observer_failures')
    if args.serial_evaluation:c['environments']=1
    pool=ProcessEnvs(c['environments'],c,c['history_offsets'],child_target=level1_child)
    global ACTIVE_POOL
    ACTIVE_POOL=pool
    c.pop('_observer_failure_dir',None)  # Runtime-only; keep the saved experiment config exact.
    if any(r['torch_imported'] for r in pool.owner_metadata):raise RuntimeError('CUDA framework imported in CPU owners')
    if args.stage=='evaluate':
        frames=evaluation_training_frames;updates=evaluation_training_updates;error=None
        try:
            write_json(folder/'activity.json',dict(active=True,pid=os.getpid(),owners=pool.pids,
                frames=frames,updates=updates,
                stage='evaluation',trials=args.trials,reset_frame=args.reset_frame,no_stall_limit=args.no_stall_limit,
                noop_prefix_frames=args.delayed_takeover_noop_prefix_frames or 0))
            if args.all_suffix_starts and args.reset_frame is None:raise ValueError('--all-suffix-starts requires a suffix reset frame')
            evaluate(model,pool,c,route,folder,frames,trials=args.trials,final=args.final,record=args.record,
                reset_frame=args.reset_frame,all_suffix_starts=args.all_suffix_starts,
                no_reset_jitter=args.no_reset_jitter,evaluation_horizon_frames=args.evaluation_horizon_frames,
                evaluation_mode=args.evaluation_mode,preserve_pool_state=False,
                save_policy_inputs=args.save_evaluation_policy_inputs,
                noop_prefix_frames=args.delayed_takeover_noop_prefix_frames or 0,
                checkpoint_provenance=evaluation_provenance,evaluation_id=args.evaluation_id,
                action_omission=action_omission,
                compact_capture=args.compact_evaluation_capture,
                storage_guard=(DiskGuard([args.evaluation_artifact_root or folder,tempfile.gettempdir()],
                    minimum_free_bytes=int(args.evaluation_minimum_free_mib)*1024*1024)
                    if args.compact_evaluation_capture else None),
                artifact_guard=(ArtifactBudgetGuard(args.evaluation_artifact_root or folder,
                    maximum_bytes=int(remaining_budget)*1024*1024,check_every=1)
                    if args.compact_evaluation_capture else None))
            return
        except Exception as exc:
            error=repr(exc)
            raise
        finally:
            pool.close()
            ACTIVE_POOL=None
            write_json(folder/'activity.json',dict(active=False,frames=frames,updates=updates,
                error=error,jobs_active=False,stage='evaluation'))
    cases=copy.deepcopy(prior.get('cases',[]));states=[]
    scripted_prefix_frames=int(prior.get('scripted_prefix_native_frames',0))
    initial_scripted_prefix_frames=scripted_prefix_frames
    episode_prefix_frames=[0 for _ in range(c['environments'])]
    episode_prefix_rewards=[0. for _ in range(c['environments'])]
    episode_policy_frames=[0 for _ in range(c['environments'])]
    if prior.get('environment_states'):
        if len(prior['environment_states'])!=c['environments']:raise ValueError('Resume environment count differs')
        for i,s in enumerate(prior['environment_states']):pool.send(i,'load_snapshot',s)
        states=[pool.receive(i) for i in range(c['environments'])]
    else:
        cases=[training_case(curriculum,rng,c,i) for i in range(c['environments'])]
        if c.get('full_start_fraction',0)>0 and c.get('fixed_training_anchor_frame') is None:
            cases[0]=dict(frame=route.start,horizon=c['full_horizon'],stage=curriculum.stage,
                          frontier=curriculum.frontier,rehearsal=True,kind='full_start',training_segment=False)
        states=pool.reset_many(cases)
        for i,(case,state) in enumerate(zip(cases,states)):
            log(folder/'resets.jsonl',reset_record(i,case,state,prior.get('frames',0),prior.get('updates',0)))
        if c.get('fixed_training_anchor_noop_prefix_frames',0):
            states,scripted_prefix_frames=apply_fixed_anchor_prefix(pool,list(range(c['environments'])),c,cases,
                states,episode_prefix_frames,episode_prefix_rewards,scripted_prefix_frames,folder,
                int(prior.get('updates',0)))
    frames=int(prior.get('frames',0));updates=int(prior.get('updates',0));decisions=int(prior.get('decisions',0))
    grad_steps=int(prior.get('optimizer_steps',0));actor_steps=int(prior.get('actor_optimizer_steps',0))
    initial_updates=updates;initial_decisions=decisions;initial_grad_steps=grad_steps;initial_actor_steps=actor_steps
    initial=frames;target,target_updates=training_budget_targets(frames,updates,args.frames,args.updates)
    begun=time.perf_counter();elapsed_prior=prior.get('wall_seconds',0.)
    rollout_time=prior.get('rollout_seconds',0.);optimization_time=prior.get('optimization_seconds',0.)
    initial_compute_time=rollout_time+optimization_time
    auxiliary=load_human_auxiliary(c,device,c.get('bc_auxiliary_cache_root',args.output/'auxiliary_memmap_cache')) if args.stage=='train' else None
    bc_aux_exposures=int(prior.get('bc_auxiliary_exposures',0))
    returns=Counter(prior.get('episode_returns',{}));recent=prior.get('recent',[]);eval_frames=int(prior.get('evaluation_frames',0))
    last_eval=int(prior.get('last_evaluation_frames',frames));last_save=frames
    best_full=tuple(prior.get('best_full',(-1,-1,-1.)));best_return=float(prior.get('best_return',-1e9));best_frontier=prior.get('best_frontier',route.end)
    best_suffix={k:tuple(v) for k,v in prior.get('best_suffix',{}).items()}
    critic_rewarmup_until=int(prior.get('critic_rewarmup_until',0))
    if args.score_reward_adaptation and args.stage=='train':
        critic_rewarmup_until=max(critic_rewarmup_until,frames+c.get('critic_rewarmup_frames',16384))
    stop=[False];previous_signals={};frontier_transition_stop=None
    cuda_peak_allocated_highwater=int(torch.cuda.max_memory_allocated())
    cuda_peak_reserved_highwater=int(torch.cuda.max_memory_reserved())
    for name in ('SIGINT','SIGBREAK'):
        sig=getattr(signal,name,None)
        if sig is not None:previous_signals[sig]=signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    ck=folder/'checkpoints';ck.mkdir(exist_ok=True)
    write_json(folder/'config.json',c)
    def metadata():
        return dict(frames=frames,updates=updates,decisions=decisions,optimizer_steps=grad_steps,actor_optimizer_steps=actor_steps,
            scripted_prefix_native_frames=scripted_prefix_frames,
            cases=cases,episode_returns=dict(returns),recent=recent,route_sha256=route.meta['route_sha256'],
            wall_seconds=elapsed_prior+time.perf_counter()-begun,rollout_seconds=rollout_time,optimization_seconds=optimization_time,
            evaluation_frames=eval_frames,last_evaluation_frames=last_eval,best_full=best_full,best_return=best_return,best_frontier=best_frontier,best_suffix=best_suffix,
            critic_rewarmup_until=critic_rewarmup_until,reward_adaptation=prior.get('reward_adaptation'),
            score_reward_adaptation=prior.get('score_reward_adaptation'),
            scoped_kind18_reward_adaptation=prior.get('scoped_kind18_reward_adaptation'),
            matched_worker_reset=prior.get('matched_worker_reset'),
            curriculum_adaptation=prior.get('curriculum_adaptation'),
            normalization_adaptation=prior.get('normalization_adaptation'),
            timing_adaptation=prior.get('timing_adaptation'),
            history_adaptation=prior.get('history_adaptation'),
            exploration_adaptation=prior.get('exploration_adaptation'),
            forward_adaptation=prior.get('forward_adaptation'),
            stall_repair=prior.get('stall_repair'),
            critic_calibration=prior.get('critic_calibration'),
            critic_trained=bool(prior.get('critic_trained') or prior.get('critic_calibration') or updates>0),
            bc_initialization=bool(c.get('bc_initialization',False)),bc_initialization_details=prior.get('bc_initialization_details'),bc_auxiliary_exposures=bc_aux_exposures,
            fixed_task=dict(anchor_frame=c.get('fixed_training_anchor_frame'),
                noop_prefix_frames=int(c.get('fixed_training_anchor_noop_prefix_frames',0)),
                prefix_frames_used=scripted_prefix_frames,
                prefix_frame_budget=c.get('fixed_training_prefix_frame_budget'),
                curriculum_advancement_disabled=bool(c.get('disable_curriculum_advancement',False))),
            lineage=str(args.resume) if args.resume else 'fresh random weights; no BC; no normalization transfer')
    def evaluate_suffixes():
        nonlocal eval_frames
        for suffix_frame in c.get('evaluation_suffix_frames',[]):
            report=evaluate(model,pool,c,route,folder,frames,trials=c.get('evaluation_suffix_trials',10),reset_frame=suffix_frame)
            eval_frames+=report['evaluation_frames']
            sampled=[r for r in report['runs'] if r['mode']=='sampled']
            score=(report['sampled_completions']/report['sampled_trials'],float(np.mean([r['return_value'] for r in sampled])))
            key=str(suffix_frame)
            if score>best_suffix.get(key,(-1.,-1e9)):
                best_suffix[key]=score;save(ck/f'best_suffix_{suffix_frame}.pt',model,optimizer,pool,curriculum,rng,metadata())
    error=None
    try:
        if not prior or args.reward_adaptation or args.score_reward_adaptation or args.curriculum_adaptation or args.normalization_adaptation:
            save(ck/'initial.pt',model,optimizer,pool,curriculum,rng,metadata())
            if args.reward_adaptation:write_json(folder/'reward_adaptation.json',prior['reward_adaptation'])
            if args.score_reward_adaptation:write_json(folder/'score_reward_adaptation.json',prior['score_reward_adaptation'])
            if args.curriculum_adaptation:write_json(folder/'curriculum_adaptation.json',prior['curriculum_adaptation'])
            if args.normalization_adaptation:write_json(folder/'normalization_adaptation.json',prior['normalization_adaptation'])
            write_json(folder/'activity.json',dict(active=True,pid=os.getpid(),owners=pool.pids,target=target,target_updates=target_updates,frames=frames,stage='initial_evaluation'))
            if not args.skip_initial_evaluation:
                report=evaluate(model,pool,c,route,folder,frames);eval_frames+=report['evaluation_frames']
                rs=report['runs']
                best_full=task_score(report,c)
                best_return=float(np.mean([r['return_value'] for r in rs]))
                save(ck/'best_full_start.pt',model,optimizer,pool,curriculum,rng,metadata())
                save(ck/'best_fixed_return.pt',model,optimizer,pool,curriculum,rng,metadata())
                if args.reward_adaptation or args.normalization_adaptation:evaluate_suffixes()
        write_json(folder/'activity.json',dict(active=True,pid=os.getpid(),owners=pool.pids,target=target,target_updates=target_updates,frames=frames))
        while (updates<target_updates if target_updates is not None else frames<target) and not stop[0]:
            memory_phases=[]
            torch.cuda.reset_peak_memory_stats()
            rollout=[];scope_collection=empty_combat_scope_collection(len(states))
            score_scope_collection=empty_score_scope_collection(len(states))
            start=time.perf_counter();rollout_epsilon=exploration_epsilon(c,frames);rollout_start_frames=frames
            for _ in range(c['rollout_per_environment']):
                observations=[s['obs'] for s in states]
                if c.get('defer_critic',False):
                    with torch.no_grad():
                        x=window_tensors({k:np.stack([o[k] for o in observations]) for k in model.window_keys},device,batch=True)
                        dist=behavior_distribution(model.actor(x),rollout_epsilon,temperature=c.get('policy_temperature',1.));sampled=dist.sample()
                        actions=sampled.cpu().numpy();logs=dist.log_prob(sampled).cpu().numpy();values=np.zeros(len(actions))
                else:
                    with torch.no_grad():
                        x=window_tensors({k:np.stack([o[k] for o in observations]) for k in model.window_keys},device,batch=True)
                        logits,value=model(x);dist=behavior_distribution(logits,rollout_epsilon,temperature=c.get('policy_temperature',1.));sampled=dist.sample()
                        actions=sampled.cpu().numpy();logs=dist.log_prob(sampled).cpu().numpy();values=value.cpu().numpy()
                results=pool.step_many(actions)
                batch_end_frames=frames+sum(int(result['info']['transition_frames']) for result in results)
                for i,result in enumerate(results):
                    info=result['info'];frames+=info['transition_frames'];decisions+=1;returns[i]+=result['reward']
                    episode_policy_frames[i]+=int(info['transition_frames'])
                    accumulate_combat_scope_transition(scope_collection,i,info)
                    accumulate_score_scope_transition(score_scope_collection,i,info)
                    rollout.append(dict(obs=compact_window(observations[i]),next_obs=compact_window(result['obs']),action=int(actions[i]),log=float(logs[i]),
                        case_kind=cases[i].get('kind','legacy'),end_reason=info.get('end_reason'),
                        value=float(values[i]),next_value=0.,reward=result['reward'],duration=info['transition_frames'],
                        bootstrap=info['bootstrap_valid'],env_id=i,**{'continue':not(result['terminated'] or result['truncated'])}))
                    if result['terminated'] or result['truncated']:
                        row=dict(frames=frames,return_value=returns[i],worker_id=i,case=cases[i],**info)
                        if c.get('fixed_training_anchor_frame') is not None:
                            row.update(scripted_prefix_native_frames=episode_prefix_frames[i],
                                scripted_prefix_reward=episode_prefix_rewards[i],
                                policy_controlled_native_frames=episode_policy_frames[i],
                                total_episode_native_frames=int(info['elapsed_frames']),
                                prefix_excluded_from_ppo_rollout=True,
                                observer_score_prefix_validity='recorded_in_anchor_prefixes.jsonl')
                            if int(info['elapsed_frames'])!=episode_prefix_frames[i]+episode_policy_frames[i]:
                                raise RuntimeError('Episode prefix/policy frame accounting does not reconcile')
                        log(folder/'episodes.jsonl',row);recent.append(row);recent=recent[-60:];returns[i]=0.
                        observation_frames=batch_end_frames if args.stop_on_frontier_mastery else frames
                        if (not c.get('disable_curriculum_advancement',False)
                                and curriculum.observe(cases[i],info['completion'],observation_frames,info['elapsed_frames'])):
                            transition=curriculum.advancements[-1]
                            if args.stop_on_frontier_mastery and is_frontier_mastery_transition(transition):
                                transition['native_frame_boundary']=int(batch_end_frames)
                                transition['update_start_frames']=int(rollout_start_frames)
                                frontier_transition_stop=dict(transition)
                                stop[0]=True
                            log(folder/'curriculum.jsonl',transition,
                                f'FRONTIER {transition.get("kind","advance_earlier")}: {curriculum.frontier} | remaining {(route.end-curriculum.frontier)/60:.1f}s')
                if c.get('defer_critic',False):next_values=np.zeros(len(results))
                else:
                    with torch.no_grad():
                        x=window_tensors({k:np.stack([r['obs'][k] for r in results]) for k in model.window_keys},device,batch=True)
                        next_values=model.critic(x).squeeze(-1).cpu().numpy()
                for row,value in zip(rollout[-len(results):],next_values):row['next_value']=float(value) if row['bootstrap'] else 0.
                states=results
                ended=[i for i,r in enumerate(results) if r['terminated'] or r['truncated']]
                life_ended=[i for i in ended if results[i]['info'].get('training_task_terminal',False)]
                for i in life_ended:pool.send(i,'continue_life',None)
                for i in life_ended:states[i]=pool.receive(i)
                ended=[i for i in ended if i not in life_ended]
                if ended:
                    new_cases=[training_case(curriculum,rng,c,i) for i in ended];reset=pool.reset_many(new_cases,ended)
                    for i,case,result in zip(ended,new_cases,reset):
                        cases[i]=case;states[i]=result
                        log(folder/'resets.jsonl',reset_record(i,case,result,frames,updates))
                    for i in ended:
                        episode_prefix_frames[i]=0;episode_prefix_rewards[i]=0.;episode_policy_frames[i]=0
                    if c.get('fixed_training_anchor_noop_prefix_frames',0):
                        states,scripted_prefix_frames=apply_fixed_anchor_prefix(pool,ended,c,new_cases,states,
                            episode_prefix_frames,episode_prefix_rewards,scripted_prefix_frames,folder,updates+1)
                # Update-budget blocks always finish the current full rollout;
                # graceful interruption is observed at the next outer-cycle boundary.
                if (target is not None and frames>=target) or (target_updates is None and stop[0]):break
            if c.get('defer_critic',False):condition_critic(model,rollout,device,batch=c['batch_size'])
            memory_phases.append(training_memory_sample(pool,'after_collection_and_critic_conditioning'))
            cuda_peak_allocated_highwater=max(cuda_peak_allocated_highwater,memory_phases[-1]['gpu']['peak_allocated_bytes'])
            cuda_peak_reserved_highwater=max(cuda_peak_reserved_highwater,memory_phases[-1]['gpu']['peak_reserved_bytes'])
            rollout_time+=time.perf_counter()-start;start=time.perf_counter()
            update_config=dict(c,entropy_coefficient=entropy_weight(c,frames),bc_auxiliary_weight=auxiliary_weight(c,frames),exploration_epsilon=rollout_epsilon)
            torch.cuda.reset_peak_memory_stats()
            stats=optimize(model,optimizer,rollout,update_config,device,rng,critic_warmup_rollout(rollout_start_frames,c['critic_warmup_frames'],critic_rewarmup_until),updates+1,auxiliary=auxiliary)
            update_optimizer_steps=int(stats['optimizer_steps'])
            update_actor_optimizer_steps=int(stats['actor_optimizer_steps'])
            reset_kind_decisions=dict(Counter(r.get('case_kind','legacy') for r in rollout))
            # The update and its diagnostics no longer need transition windows.
            # Release them before logging/checkpoint snapshots are materialized.
            del rollout
            memory_phases.append(training_memory_sample(pool,'after_optimization_and_rollout_release'))
            cuda_peak_allocated_highwater=max(cuda_peak_allocated_highwater,memory_phases[-1]['gpu']['peak_allocated_bytes'])
            cuda_peak_reserved_highwater=max(cuda_peak_reserved_highwater,memory_phases[-1]['gpu']['peak_reserved_bytes'])
            bc_aux_exposures+=stats.get('bc_auxiliary_samples',0)
            optimization_time+=time.perf_counter()-start;updates+=1;grad_steps+=stats.pop('optimizer_steps');actor_steps+=stats.pop('actor_optimizer_steps')
            elapsed=time.perf_counter()-begun
            if frontier_transition_stop is not None:
                frontier_transition_stop['frames_collected_after_transition_in_rollout']=max(
                    0,int(frames)-int(frontier_transition_stop['native_frame_boundary']))
            stats.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_peak_allocated_bytes=cuda_peak_allocated_highwater,
                cuda_peak_reserved_bytes=cuda_peak_reserved_highwater,host_rss_bytes=resident_bytes(os.getpid()),
                memory_phase_snapshots=memory_phases)
            # Training ETA excludes initial/periodic evaluation wall time.
            fps=(frames-initial)/max(rollout_time+optimization_time-initial_compute_time,1e-6)
            recent_mean=float(np.mean([r['return_value'] for r in recent])) if recent else None
            remaining_budget_frames=(max(0,target-frames) if target is not None else
                max(0,target_updates-updates)*c['rollout_per_environment']*c['environments']*c['frame_skip'])
            row=dict(frames=frames,decisions=decisions,update=updates,rollout_start_frames=rollout_start_frames,
                frontier=curriculum.frontier,remaining_demo_seconds=(route.end-curriculum.frontier)/60,
                policy_controlled_native_frames=frames-initial,
                scripted_prefix_native_frames=scripted_prefix_frames,
                total_newly_advanced_training_frames=(frames-initial+scripted_prefix_frames-initial_scripted_prefix_frames),
                frontier_success_rate=float(np.mean(curriculum.outcomes)) if curriculum.outcomes else 0.,frontier_episodes=len(curriculum.outcomes),
                frontier_outcomes=list(curriculum.outcomes),learned_starts=list(curriculum.learned),
                curriculum_transitions=copy.deepcopy(curriculum.advancements),
                initial_failure_frames=int(getattr(curriculum,'initial_failure_frames',0)),
                initial_fallback_candidates=list(getattr(curriculum,'initial_fallback_candidates',[])),
                initial_fallback_cursor=int(getattr(curriculum,'initial_fallback_cursor',0)),
                initial_fallback_exhausted=bool(getattr(curriculum,'initial_fallback_exhausted',False)),
                recent_training_return=recent_mean,recent_suffix_success_rate=float(np.mean([r['completion'] for r in recent])) if recent else 0.,
                reward_components={k:float(np.mean([r['reward_totals'].get(k,0) for r in recent])) for k in c['weights']} if recent else {},
                reset_kind_decisions=reset_kind_decisions,
                combat_reward_scope=summarize_combat_scope_collection(scope_collection),
                score_gain_scope=summarize_score_scope_collection(score_scope_collection),
                update_optimizer_steps=update_optimizer_steps,
                update_actor_optimizer_steps=update_actor_optimizer_steps,
                frontier_transition_stop=copy.deepcopy(frontier_transition_stop),
                entropy_weight=update_config['entropy_coefficient'],exploration_epsilon=rollout_epsilon,policy_temperature=c.get('policy_temperature',1.),curriculum_mode=c.get('curriculum_mode','backward'),
                fps=fps,wall_seconds=elapsed_prior+elapsed,eta_seconds=remaining_budget_frames/max(fps,1),**stats)
            stage_label='CRITIC WARMUP' if stats['warmup'] else 'PPO'
            outcome_label=('local milestones evaluated at review' if c.get('forward_curriculum_version') else
                           f'suffix {row["frontier_success_rate"]:.0%}/{row["frontier_episodes"]}')
            log(folder/'updates.jsonl',row,f'{stage_label} {updates} | frames {frames:,} decisions {decisions:,} | frontier {curriculum.frontier} '
                f'remaining {row["remaining_demo_seconds"]:.1f}s ({c.get("curriculum_mode","backward")}) | {outcome_label} '
                f'train return {recent_mean} | entropy {stats["entropy"]:.3f} KL {stats["approximate_kl"]:.5f} '
                f'clip {stats["clip_fraction"]:.3f} EV {stats["explained_variance"]:.3f} | {fps:.0f}fps elapsed {elapsed/60:.1f}m training ETA {row["eta_seconds"]/60:.1f}m')
            if updates%5==0 and recent:
                print('TRAIN mixed-start last60 reward contributions '+json.dumps(row['reward_components']),flush=True)
                by_kind={}
                for episode in recent:
                    kind=episode['case'].get('kind','legacy')
                    group=by_kind.setdefault(kind,dict(episodes=0,completion=0,return_sum=0.,life_losses=0.))
                    group['episodes']+=1;group['completion']+=int(episode['completion']);group['return_sum']+=episode['return_value']
                    group['life_losses']+=episode['event_totals'].get('lives_lost',0)
                print('TRAIN reset groups '+json.dumps(by_kind),flush=True)
            if frames-last_eval>=c['evaluation_interval']:
                report=evaluate(model,pool,c,route,folder,frames,record=False);eval_frames+=report['evaluation_frames'];last_eval=frames
                rs=report['runs'];score=task_score(report,c)
                ret=float(np.mean([r['return_value'] for r in rs]))
                if score>best_full:best_full=score;save(ck/'best_full_start.pt',model,optimizer,pool,curriculum,rng,metadata())
                if ret>best_return:best_return=ret;save(ck/'best_fixed_return.pt',model,optimizer,pool,curriculum,rng,metadata())
                evaluate_suffixes()
                save(ck/f'eval_{frames:09d}.pt',model,optimizer,pool,curriculum,rng,metadata());curves(folder)
            if curriculum.frontier<best_frontier:
                best_frontier=curriculum.frontier;save(ck/'best_curriculum.pt',model,optimizer,pool,curriculum,rng,metadata())
            if frames-last_save>=c['checkpoint_interval']:
                save(ck/'latest.pt',model,optimizer,pool,curriculum,rng,metadata());last_save=frames
                print(f'CHECKPOINT latest at {frames:,} frames; best frontier {best_frontier}',flush=True)
            write_json(folder/'activity.json',dict(active=True,pid=os.getpid(),owners=pool.pids,target=target,target_updates=target_updates,frames=frames,updates=updates,frontier=curriculum.frontier))
            if (folder/'stop.request').exists():stop[0]=True
        final_save=save(ck/'latest.pt',model,optimizer,pool,curriculum,rng,metadata(),skip_if_identical=True)
        if not final_save['saved']:
            print('CHECKPOINT latest finalization skipped; complete saved resume state is identical',flush=True)
        curves(folder)
        block=dict(initial_frames=initial,frames=frames,updates=updates,target_updates=target_updates,additional_frames=frames-initial,
            additional_policy_controlled_native_frames=frames-initial,
            initial_scripted_prefix_native_frames=initial_scripted_prefix_frames,
            scripted_prefix_native_frames=scripted_prefix_frames,
            additional_scripted_prefix_native_frames=scripted_prefix_frames-initial_scripted_prefix_frames,
            additional_total_new_native_frames=(frames-initial+scripted_prefix_frames-initial_scripted_prefix_frames),
            scripted_prefix_budget=int(c.get('fixed_training_prefix_frame_budget',0)),
            additional_updates=updates-initial_updates,additional_decisions=decisions-initial_decisions,
            additional_optimizer_steps=grad_steps-initial_grad_steps,additional_actor_optimizer_steps=actor_steps-initial_actor_steps,
            rollout_seconds=rollout_time-prior.get('rollout_seconds',0.),
            optimization_seconds=optimization_time-prior.get('optimization_seconds',0.),
            config=str(args.config),resume=str(args.resume) if args.resume else None,
            frontier=curriculum.frontier,frontier_transition_stop=copy.deepcopy(frontier_transition_stop),
            frames_collected_after_transition_in_rollout=(frontier_transition_stop.get('frames_collected_after_transition_in_rollout')
                if frontier_transition_stop else 0),
            stop_reason=('frontier_mastery_transition' if frontier_transition_stop else
                ('external_stop_request' if stop[0] else None)),
            wall_seconds=time.perf_counter()-begun,interrupted=stop[0],jobs_active=False)
        write_json(folder/'last_block.json',block);log(folder/'blocks.jsonl',block)
    except Exception as exc:
        error=repr(exc)
        if args.stage=='train':
            try:save(ck/f'recovery_{frames:09d}.pt',model,optimizer,pool,curriculum,rng,metadata())
            except Exception as recovery:print('Recovery checkpoint failed: '+repr(recovery),flush=True)
        raise
    finally:
        pool.close()
        ACTIVE_POOL=None
        for sig,handler in previous_signals.items():signal.signal(sig,handler)
        write_json(folder/'activity.json',dict(active=False,frames=frames,updates=updates,frontier=curriculum.frontier,error=error,jobs_active=False))


if __name__=='__main__':
    try:main()
    except BaseException:
        active=globals().get('ACTIVE_POOL')
        if active is not None:
            active.close();ACTIVE_POOL=None
        raise
