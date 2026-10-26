"""Stream selected Level2 human ancestry into a compact RAM-only replay cache."""
from _bootstrap import ROOT
from io import BytesIO
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError('replay preparation output occupied')
    from prince_valiant_rl.training.level2_runtime import Encoder,transition,SCHEMA,WEIGHTS
    db=ROOT/'captures/human/20261013_224343_044974/recording.sqlite3'
    expected='6223eeaa7fd4575e6bfbe85fc9be2b1fae20cf9a81abb386fb105ef7fcc541d4'
    if digest(db)!=expected:raise ValueError('source database drift')
    wal=db.with_name(db.name+'-wal')
    if wal.exists() and wal.stat().st_size:raise ValueError('immutable read requires empty WAL')
    seg_path=ROOT/'analysis/level23_human_segmentation_20261023/segmentation.json'
    seg=json.loads(seg_path.read_text());start=20760;end=43209;n=end-start
    if seg['source']['sqlite']['sha256']!=expected:raise ValueError('segmentation binding mismatch')
    ram=np.zeros((n+1,2048),np.uint8);actions=np.zeros(n,np.int64);rewards=np.zeros(n,np.float32)
    terminals=np.zeros(n,bool);records=np.zeros(n,np.int64);components=[];expected_frame=start
    masks=Encoder().actions;counts={str(a):0 for a in masks};previous=None;fragments=[]
    conn=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    try:
        for fragment in seg['selected_route']['ancestry']:
            lo=max(start,fragment['frame_start_inclusive']);hi=min(end,fragment['frame_end_exclusive'])
            if lo>=hi:continue
            if lo!=expected_frame:raise ValueError('selected ancestry gap/overlap')
            branch=fragment['branch_id'];fragments.append(dict(branch=branch,start=lo,end=hi))
            for (blob,) in conn.execute('select data from chunks where branch=? and first_frame<? and first_frame+count>? order by first_frame',(branch,hi,lo)):
                if len(blob)>16*1024*1024:raise ValueError('oversize source chunk')
                with np.load(BytesIO(blob),allow_pickle=False) as z:
                    frames=z['frame'];pre=z['pre_ram'];post=z['post_ram'];buttons=z['buttons'];rec=z['record']
                    if pre.dtype!=np.uint8 or pre.shape!=(len(frames),2048) or post.shape!=pre.shape or post.dtype!=np.uint8:
                        raise ValueError('RAM contract mismatch')
                    for j in np.flatnonzero((frames>=lo)&(frames<hi)):
                        frame=int(frames[j]);i=frame-start
                        if frame!=expected_frame:raise ValueError('source order mismatch')
                        if previous is not None and not np.array_equal(previous,pre[j]):raise ValueError('post/pre discontinuity')
                        mask=int(buttons[j])
                        info=transition(pre[j],post[j],level2_scope=True)
                        # Keep exact raw trajectory/history/returns. Unsupported
                        # actions get no invented label and cannot start a batch row.
                        ram[i]=pre[j];ram[i+1]=post[j];actions[i]=masks.index(mask) if mask in masks else -1;rewards[i]=info['reward'];terminals[i]=info['terminated'];records[i]=rec[j]
                        components.append(info['components']);counts[str(mask)]=counts.get(str(mask),0)+1
                        previous=post[j].copy();expected_frame+=1
        if expected_frame!=end or terminals.sum()!=1 or not terminals[-1]:raise ValueError('expected one final completion boundary')
        if not (np.diff(records)>0).all():raise ValueError('execution record order mismatch')
    finally:conn.close()
    args.output.mkdir(parents=True)
    with (args.output/'replay.npz').open('xb') as stream:
        np.savez_compressed(stream,ram=ram,actions=actions,rewards=rewards,terminals=terminals,records=records)
    result=dict(status='PREPARED',schema=SCHEMA,partial_observation_baseline=True,source_db_sha256=expected,
        segmentation_sha256=digest(seg_path),frame_start=start,frame_end_exclusive=end,rows=n,fragments=fragments,
        action_masks=masks,action_counts=counts,weights=WEIGHTS,
        trainable_start_rows=int((actions>=0).sum()),unsupported_action_start_rows=int((actions<0).sum()),
        unsupported_policy='No remapping. Retain exact trajectory for history/observed returns; exclude unsupported starts from sampling and margin labels.',
        component_totals={k:sum(c[k] for c in components) for k in components[0]},
        reward_sum=float(rewards.astype(np.float64).sum()),terminal_rows=int(terminals.sum()),
        terminal_outcome='level_complete',cache_sha256=digest(args.output/'replay.npz'),
        observation_scope='Raw control/render state, energy/retry bytes and eight opaque slot codes. No verified hostile/projectile positions, kills or hitboxes.',
        training_source='Existing successful human route only; rewinds follow selected ancestry with exact RAM continuity.',
        native_steps=0,policy_forwards=0,optimizer_steps=0)
    with (args.output/'result.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:result[k] for k in ['status','rows','action_counts','reward_sum','terminal_outcome']},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
