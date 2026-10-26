"""Audit human branch paths and replay a successful Level 1 backbone, read-only.

Derived artifacts are isolated from human recordings and all previous training.
"""
from _bootstrap import ROOT
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import time
import zlib

import numpy as np
from PIL import Image

from audit_manual_recording import audit
from prince_valiant_rl.recorder_native import RecorderNative
from prince_valiant_rl.navigation import completed, supported, live, map_section

OUT = ROOT / 'training/backward_level1_v1'


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def recording_identity(db):
    row=db.execute("SELECT value FROM metadata WHERE key='initial_checkpoint'").fetchone()
    if row is None:raise ValueError('Recording lacks an initial complete checkpoint')
    chunks=db.execute('SELECT COUNT(*),COALESCE(SUM(count),0),COALESCE(MAX(first_record+count),0) FROM chunks').fetchone()
    digest=hashlib.sha256()
    for table,key in (('metadata','key'),('branches','id'),('chunks','id'),('checkpoints','id'),('resets','id')):
        digest.update(table.encode('ascii'))
        for values in db.execute(f'SELECT * FROM {table} ORDER BY {key}'):
            for value in values:
                payload=value if isinstance(value,bytes) else repr(value).encode('utf-8')
                digest.update(type(value).__name__.encode('ascii'));digest.update(len(payload).to_bytes(8,'little'));digest.update(payload)
    return (json.loads(row[0]),*chunks,db.execute('SELECT COUNT(*) FROM branches').fetchone()[0],
            db.execute('SELECT COUNT(*) FROM checkpoints').fetchone()[0],digest.hexdigest())


def assert_same_snapshot(source,snapshot):
    original=recording_identity(source);saved=recording_identity(snapshot)
    if original!=saved:
        raise ValueError('Output snapshot belongs to another recording/version. Use --output with a NEW directory; existing data is preserved.')
    return saved


def branch_path(branches, checkpoints, leaf, end):
    """Iterative ancestry traversal: some human paths have >1000 forks."""
    path, seen = [], set()
    while True:
        if leaf in seen:
            raise ValueError('Branch ancestry cycle')
        seen.add(leaf)
        branch = branches[leaf]
        if not branch['start'] <= end <= branch['end']:
            raise ValueError('Invalid branch cutoff')
        path.append(dict(branch=leaf, start=branch['start'], end=end))
        parent = branch['parent']
        if parent is None:
            break
        source = checkpoints[parent]
        if source['frame'] != branch['start']:
            raise ValueError('Fork frame differs from source checkpoint')
        end, leaf = source['frame'], source['branch']
    return list(reversed(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, default=ROOT/'captures/human/20261013_224343_044974')
    p.add_argument('--scan-only', action='store_true')
    p.add_argument('--candidate', type=int, default=0)
    p.add_argument('--output',type=Path,default=OUT,help='Immutable data version directory; use a NEW directory for a new/extended session')
    args = p.parse_args()
    out=args.output if args.output.is_absolute() else ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    if (out/'route.json').exists() and not args.scan_only:
        raise ValueError('Verified route already exists. Use --scan-only to inspect or --output with a NEW directory to rebuild.')
    source = (args.session/'recording.sqlite3').resolve()
    # Stable SQLite backup includes WAL without touching the human database.
    frozen = out/'human_session.sqlite3'
    if not frozen.exists():
        original = sqlite3.connect(source.as_uri()+'?mode=ro', uri=True)
        copy = sqlite3.connect(frozen)
        original.backup(copy); copy.close(); original.close()
    db = sqlite3.connect(frozen.resolve().as_uri()+'?mode=ro', uri=True)
    original=sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)
    try:source_identity=assert_same_snapshot(original,db)
    finally:original.close()
    meta = {k: json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
    if meta['interventions'] != []:
        raise ValueError('Demonstration has interventions')
    branches = {b:dict(trajectory=t,parent=p,start=s,end=e,mark=m)
        for b,t,p,s,e,m in db.execute('SELECT id,trajectory,parent_checkpoint,start_frame,final_frame,mark FROM branches')}
    checkpoints = {i:dict(branch=b,frame=f,kind=k) for i,b,f,k in db.execute('SELECT id,branch,frame,kind FROM checkpoints')}
    successes, masks, boundaries = [], Counter(), []
    for b,data in db.execute('SELECT branch,data FROM chunks ORDER BY first_record'):
        with np.load(io.BytesIO(data),allow_pickle=False) as c:
            pre,post=c['pre_ram'],c['post_ram']
            masks.update(map(int,c['buttons']))
            candidates=np.flatnonzero((pre[:,0x33]==7)&(post[:,0x33]==4))
            for i in candidates:
                verified=completed(pre[i],post[i])
                boundaries.append(dict(branch=b,frame=int(c['frame'][i]+1),record=int(c['record'][i]),verified=verified,
                    old_mode=int(pre[i,0x33]),new_mode=int(post[i,0x33]),row=int(pre[i,0x50E])+256*int(pre[i,0x50F]),
                    gate=int(pre[i,0x204]),player_dispatch=int(pre[i,0x98])))
                if verified:
                    successes.append(boundaries[-1])
    report=dict(source=str(source),frozen_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),
        logical_snapshot_identity=source_identity,
        frames=sum(masks.values()),branches=len(branches),checkpoints=len(checkpoints),session_meta=meta,
        controller_counts=dict(masks),successes=successes,completion_candidates=boundaries)
    dump(out/'human_scan.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('session_meta','frozen_sha256')},indent=2),flush=True)
    if args.scan_only:
        return
    if not successes:
        raise ValueError('No verified Level 1 completion; inspect transitions before training')
    selected=successes[args.candidate]
    path=branch_path(branches,checkpoints,selected['branch'],selected['frame'])
    pre,post,buttons,record_ids=[],[],[],[]
    origin=[]
    for segment in path:
        start,end=segment['start'],segment['end']
        cursor=start
        for (data,) in db.execute('SELECT data FROM chunks WHERE branch=? ORDER BY first_frame',(segment['branch'],)):
            with np.load(io.BytesIO(data),allow_pickle=False) as c:
                ix=np.flatnonzero((c['frame']>=start)&(c['frame']<end))
                if not len(ix):continue
                if int(c['frame'][ix[0]])!=cursor:raise ValueError('Missing frame within branch')
                pre.extend(c['pre_ram'][ix]);post.extend(c['post_ram'][ix]);buttons.extend(c['buttons'][ix]);record_ids.extend(c['record'][ix])
                origin.extend([segment['branch']]*len(ix));cursor+=len(ix)
        if cursor!=end:raise ValueError('Incomplete branch tail')
    pre=np.asarray(pre,np.uint8);post=np.asarray(post,np.uint8);buttons=np.asarray(buttons,np.uint8)
    np.testing.assert_array_equal(post[:-1],pre[1:])
    if path[0]['start']!=0:raise ValueError('No initial state ancestry')
    root_cp=meta['initial_checkpoint']
    initial=zlib.decompress(db.execute('SELECT state FROM checkpoints WHERE id=?',(root_cp,)).fetchone()[0])
    (out/'initial.state').write_bytes(initial)
    native=RecorderNative(ROOT/'roms/prince_valiant.nes');native.load_state(initial)
    states_dir=out/'states';states_dir.mkdir(exist_ok=True)
    # Native cached reset states every second. Earlier checkpoint replay validates
    # every intervening RAM transition, and saved checkpoints verify screen too.
    checkpoint_at={}
    for segment in path:
        for ident,cp in checkpoints.items():
            if cp['branch']==segment['branch'] and segment['start']<=cp['frame']<=segment['end']:
                checkpoint_at[cp['frame']]=ident
    valid=[];state_refs={};native_checks=0;start=time.perf_counter();level_start=None
    try:
        for i in range(len(buttons)+1):
            ram=native.ram_buffer().copy()
            expected=pre[i] if i<len(buttons) else post[-1]
            np.testing.assert_array_equal(ram,expected,err_msg=f'Coherent replay mismatch before frame {i}')
            if i in checkpoint_at:
                blob,thumb=db.execute('SELECT state,thumbnail FROM checkpoints WHERE id=?',(checkpoint_at[i],)).fetchone()
                saved=RecorderNative(ROOT/'roms/prince_valiant.nes')
                try:
                    saved.load_state(zlib.decompress(blob))
                    np.testing.assert_array_equal(saved.ram_buffer(),ram)
                    np.testing.assert_array_equal(saved.copy_screen_rgb(),native.copy_screen_rgb())
                    # Branch restore identity: exact serialized machine state, not RAM alone.
                    if saved.dump_state()!=native.dump_state():raise ValueError(f'Full state differs at checkpoint frame {i}')
                finally:saved.close()
                native_checks+=1
            if live(ram) and supported(ram):
                if level_start is None and map_section(ram)==0:level_start=i
                valid.append(i)
            if i%60==0 or i==level_start or i==len(buttons):
                filename=f'{i:07d}.state.z'
                (states_dir/filename).write_bytes(zlib.compress(native.dump_state(),3))
                state_refs[str(i)]=filename
            if i==len(buttons):break
            native.frame_advance(int(buttons[i]))
            np.testing.assert_array_equal(native.ram_buffer(),post[i],err_msg=f'Post-RAM frame {i}')
            if i%3000==0:print(f'Verified replay {i}/{len(buttons)} native frames',flush=True)
        Image.fromarray(native.copy_screen_rgb()).save(out/'human_completion.png')
    finally:native.close()
    np.savez_compressed(out/'route.npz',ram=np.concatenate((pre,post[-1:])),buttons=buttons,
                        record=np.asarray(record_ids,np.int64),branch=np.asarray(origin),valid_starts=np.asarray(valid))
    lengths=[];last=None
    for m in buttons:
        if last==int(m):lengths[-1][1]+=1
        else:lengths.append([int(m),1]);last=int(m)
    route=dict(selected=selected,path=path,initial_checkpoint=root_cp,level_start=level_start,
        logical_snapshot_identity=source_identity,
        completion_frame=len(buttons),full_state_checks=native_checks,replay_frames=len(buttons),replay_seconds=time.perf_counter()-start,
        states=state_refs,action_counts=dict(Counter(map(int,buttons))),hold_duration_counts=dict(Counter(n for m,n in lengths)),
        available_successes=len(successes),independent_trajectories=len(set(b['trajectory'] for b in branches.values())),
        interventions=[],route_sha256=hashlib.sha256((out/'route.npz').read_bytes()).hexdigest(),verified=True)
    dump(out/'route.json',route)
    print(json.dumps({k:v for k,v in route.items() if k not in ('states','path','hold_duration_counts')},indent=2),flush=True)
    db.close()


if __name__=='__main__':main()
