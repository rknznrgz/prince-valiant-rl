"""Prepare exact selected-ancestry Level3 RAM/action supervision and visual contracts."""
from _bootstrap import ROOT
from io import BytesIO
import hashlib
import json
import sqlite3
import zlib
import numpy as np
from PIL import Image
from level2_v12_rollouts import digest,write,resource_gate
from prince_valiant_rl.recorder_native import RecorderNative

NAME='level3_imitation_v1_20261026'


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False);resource_gate(out,'launch',launch=True)
    db=ROOT/'captures/human/20261013_224343_044974/recording.sqlite3';expected='6223eeaa7fd4575e6bfbe85fc9be2b1fae20cf9a81abb386fb105ef7fcc541d4';assert digest(db)==expected
    wal=db.with_name(db.name+'-wal');assert not wal.exists() or wal.stat().st_size==0
    sp=ROOT/'analysis/level23_human_segmentation_20261023/segmentation.json';ip=sp.with_name('checkpoint_evidence_index.json');seg=json.loads(sp.read_text());index=json.loads(ip.read_text());assert seg['source']['sqlite']['sha256']==expected
    start,end=44700,93800;n=end-start
    ram=np.empty((n+1,2048),np.uint8);masks=np.empty(n,np.uint8);records=np.empty(n,np.int64);expected_frame=start;previous=None;fragments=[]
    con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);con.execute('PRAGMA query_only=ON');assert con.execute('PRAGMA query_only').fetchone()[0]==1
    try:
        for frag in seg['selected_route']['ancestry']:
            lo=max(start,frag['frame_start_inclusive']);hi=min(end,frag['frame_end_exclusive'])
            if lo>=hi:continue
            assert lo==expected_frame;fragments.append(dict(branch=frag['branch_id'],start=lo,end=hi))
            for (blob,) in con.execute('select data from chunks where branch=? and first_frame<? and first_frame+count>? order by first_frame',(frag['branch_id'],hi,lo)):
                assert len(blob)<=16*1024*1024
                with np.load(BytesIO(blob),allow_pickle=False) as d:
                    for j in np.flatnonzero((d['frame']>=lo)&(d['frame']<hi)):
                        frame=int(d['frame'][j]);assert frame==expected_frame;i=frame-start
                        pre,post=d['pre_ram'][j],d['post_ram'][j]
                        assert pre.shape==post.shape==(2048,) and pre.dtype==post.dtype==np.uint8
                        if previous is not None:np.testing.assert_array_equal(previous,pre)
                        ram[i]=pre;ram[i+1]=post;masks[i]=int(d['buttons'][j]);records[i]=int(d['record'][j]);previous=post.copy();expected_frame+=1
        assert expected_frame==end and np.all(np.diff(records)>0)
        checkpoints=[]
        for frame in [44700,44820,90000,92400,93720]:
            cp=next(c for c in index['checkpoints'] if c['frame']==frame)
            row=con.execute('select branch,frame,record_end,state from checkpoints where id=?',(cp['checkpoint_id'],)).fetchone();assert tuple(row[:3])==(cp['branch_id'],frame,cp['record_end'])
            state=zlib.decompress(row[3]);path=out/f'frame_{frame}.state';path.write_bytes(state)
            native=RecorderNative(ROOT/'roms/prince_valiant.nes')
            try:
                native.load_state(state);np.testing.assert_array_equal(native.ram_buffer(),ram[frame-start]);Image.fromarray(native.copy_screen_rgb()).save(out/f'frame_{frame}.png')
            finally:native.close()
            checkpoints.append(dict(frame=frame,state=str(path.relative_to(ROOT)),state_sha256=digest(path),screenshot_sha256=digest(out/f'frame_{frame}.png'),source=cp))
    finally:con.close()
    vocabulary=np.unique(masks);mapping={int(mask):i for i,mask in enumerate(vocabulary)}
    labels=np.array([mapping[int(mask)] for mask in masks],np.int64)
    keys={};conflicts=[]
    for i,row in enumerate(ram[:-1]):
        key=hashlib.sha256(row.tobytes()).digest()
        if key in keys and masks[keys[key]]!=masks[i]:conflicts.append(dict(a=keys[key],b=i,first=int(masks[keys[key]]),second=int(masks[i])))
        keys[key]=i
    np.savez_compressed(out/'demonstration.npz',ram=ram,masks=masks,labels=labels,records=records)
    # Raw changes are measurements only, not transferred Level2 terminal semantics.
    raw_F3_changes=[dict(frame=start+int(i)+1,before=int(ram[i,0xF3]),after=int(ram[i+1,0xF3])) for i in np.flatnonzero(ram[:-1,0xF3]!=ram[1:,0xF3])]
    receipt=dict(status='prepared',study=NAME,source_db_sha256=expected,segmentation_sha256=digest(sp),evidence_index_sha256=digest(ip),
        frame_start=start,frame_end=end,rows=n,fragments=fragments,checkpoints=checkpoints,vocabulary=vocabulary.tolist(),
        dataset_sha256=digest(out/'demonstration.npz'),raw_F3_changes=raw_F3_changes,unique_current_ram=len(keys),current_ram_action_conflicts=conflicts,
        observation_schema='level3_raw_ram_imitation_v1',observation_shape=[2048],observation_dtype='uint8',observation_contract='All current NES CPU RAM bytes as opaque numeric observation. Train-only normalization. No invented enemy/hitbox/score/resource labels.',
        human_supervision=True,record_ids_policy_input=False,frame_indices_policy_input=False,future_ram_policy_input=False,
        terminal_contract='Require visible LEVEL3 COMPLETE and following Level4 title; byte flags only candidate evidence until visually bound.',
        native_frames=0,optimizer_steps=0,all_source_rows_training=True,scope='Known selected-ancestry human route, including title/completion transitions. No unseen-data validation claim.')
    write(out/'preparation.json',receipt)
    print(json.dumps(dict(status='prepared',rows=n,vocabulary=vocabulary.tolist(),conflicts=len(conflicts),F3_changes=raw_F3_changes)))


if __name__=='__main__':main()
