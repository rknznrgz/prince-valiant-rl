"""Relabel the verified Level2 RAM cache with v4 score rewards and paired returns.

Offline only: no model, native library, optimizer, CUDA or policy execution.
The completed v3 replay/checkpoint/trainer remain unchanged.
"""
from _bootstrap import ROOT
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError('score replay output occupied')
    from prince_valiant_rl.training.level2_runtime import ten_step_arrays,SCHEMA
    from prince_valiant_rl.training.level2_score_reward import transition,REWARD_VERSION,WEIGHTS
    source=ROOT/'analysis/level2_semantics_v3_20261025/replay/replay.npz'
    source_receipt=json.loads(source.with_name('result.json').read_text())
    if digest(source)!=source_receipt['cache_sha256']:raise ValueError('source cache hash changed')
    current=json.loads((ROOT/'config/level2_reward_current.json').read_text())
    if current['reward_version']!=REWARD_VERSION or current['score_per_point']!=WEIGHTS['score']:
        raise ValueError('active reward selection mismatch')
    with np.load(source,allow_pickle=False) as z:
        data={k:z[k] for k in ['ram','actions','rewards','terminals','records']}
    n=len(data['actions']);old_rewards=data['rewards'].copy()
    rewards=np.zeros(n,np.float32);gains=np.zeros(n,np.int32);components={}
    for i in range(n):
        info=transition(data['ram'][i],data['ram'][i+1],level2_scope=True)
        if bool(info['terminated'])!=bool(data['terminals'][i]):raise ValueError('terminal label changed')
        rewards[i]=info['reward'];gains[i]=info['score_gain']
        for k,v in info['components'].items():components[k]=components.get(k,0.)+v
    if int(gains.sum())!=9150 or np.count_nonzero(gains)!=160:raise ValueError('score-route audit mismatch')
    paired=ten_step_arrays(rewards,data['terminals'],gamma=.99997)
    old_paired=ten_step_arrays(old_rewards,data['terminals'],gamma=.99997)
    data['rewards']=rewards
    args.output.mkdir(parents=True)
    with (args.output/'replay.npz').open('xb') as stream:
        np.savez_compressed(stream,**data,score_gains=gains,
            rewards10=paired['reward'],following10=paired['following'],discount10=paired['discount'],bootstrap10=paired['bootstrap'])
    result=dict(status='RELABELLED_NOT_TRAINED',schema=SCHEMA,reward_version=REWARD_VERSION,weights=WEIGHTS,
        source_cache=str(source.relative_to(ROOT)),source_cache_sha256=source_receipt['cache_sha256'],
        source_db_sha256=source_receipt['source_db_sha256'],cache_sha256=digest(args.output/'replay.npz'),
        rows=n,trainable_start_rows=source_receipt['trainable_start_rows'],unsupported_action_start_rows=source_receipt['unsupported_action_start_rows'],
        frame_start=20760,frame_end_exclusive=43209,action_masks=source_receipt['action_masks'],
        action_remapping=False,terminal_labels_unchanged=True,observations_unchanged=True,
        gamma=.99997,n_step=10,score_points_total=int(gains.sum()),positive_score_transitions=int(np.count_nonzero(gains)),
        score_reward_total=components['score'],component_totals=components,
        reward_sum_float32=float(rewards.astype(np.float64).sum()),old_reward_sum_float32=float(old_rewards.astype(np.float64).sum()),
        changed_one_step_rows=int(np.count_nonzero(rewards!=old_rewards)),
        changed_ten_step_rows=int(np.count_nonzero(paired['reward']!=old_paired['reward'])),
        legacy_trainer_compatible=False,
        legacy_trainer_note='The completed v3 one-shot trainer intentionally rejects the changed weights; a future launcher must explicitly bind v4 for both learning and evaluation.',
        training_updates=0,model_forwards=0,native_frames=0,CUDA_initialization=False)
    with (args.output/'result.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:result[k] for k in ['status','rows','score_points_total','score_reward_total','reward_sum_float32','changed_one_step_rows','changed_ten_step_rows']},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
