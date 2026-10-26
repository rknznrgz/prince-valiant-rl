"""Train RAM behavioral cloning and test neural versus explicit state-memory policies."""
from _bootstrap import ROOT
from io import BytesIO
import gzip
import json
import sqlite3
import time
import numpy as np
import torch
from torch.nn import functional as F
import imageio_ffmpeg
from level2_v12_rollouts import digest,write,resource_gate
from prince_valiant_rl.recorder_native import RecorderNative
from prince_valiant_rl.training.level2_score_reward import transition
from prince_valiant_rl.training.level2_demonstration_policy import DemonstrationActor,KnownStatePolicy,SCHEMA
from prince_valiant_rl.training.temporal_models import configure_precision

NAME='level2_title_bc_v57_20261026'


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--report',required=True);args=p.parse_args()
    out=ROOT/args.output;report=ROOT/args.report
    if out.exists() or report.exists():raise FileExistsError('Choose fresh output and report paths')
    out.mkdir(parents=True);report.mkdir(parents=True);resource_gate(out,'launch',launch=True)
    source=ROOT/'analysis/level2_score_reward_v4_20261025/replay_v4/replay.npz';assert digest(source)=='159834da0f4c8a53fd7429000806b6c0cac5e00838b973748887e8db6bc818df'
    with np.load(source,allow_pickle=False) as d:ram=d['ram'];records=d['records'];actions=d['actions']
    intro_root=ROOT/'analysis/level2_title_bc_v57_20261026_preparation'
    intro_authority=json.loads((intro_root/'authority.json').read_text())
    assert digest(intro_root/'intro.npz')==intro_authority['intro_sha256']
    with np.load(intro_root/'intro.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(data['ram'][-1],ram[0]);ram=np.concatenate([data['ram'][:-1],ram]);records=np.concatenate([data['records'],records]);actions=np.concatenate([np.full(60,-2),actions])
    db=ROOT/'captures/human/20261013_224343_044974/recording.sqlite3';assert digest(db)=='6223eeaa7fd4575e6bfbe85fc9be2b1fae20cf9a81abb386fb105ef7fcc541d4'
    conn=sqlite3.connect(db.resolve().as_uri()+'?mode=ro&immutable=1',uri=True);ids={int(value):i for i,value in enumerate(records)};masks=np.full(len(records),-1,np.int16)
    try:
        for (blob,) in conn.execute('select data from chunks where first_record<=? and first_record+count>? order by first_record',(int(records[-1]),int(records[0]))):
            with np.load(BytesIO(blob),allow_pickle=False) as d:
                for j,record in enumerate(d['record']):
                    if int(record) not in ids:continue
                    i=ids[int(record)];np.testing.assert_array_equal(d['pre_ram'][j],ram[i]);np.testing.assert_array_equal(d['post_ram'][j],ram[i+1]);mask=int(d['buttons'][j])
                    assert masks[i] in [-1,mask];masks[i]=mask
    finally:conn.close()
    assert np.all(masks>=0)
    vocabulary=np.unique(masks).astype(np.int64);labels=np.searchsorted(vocabulary,masks).astype(np.int64)
    memory=None  # Selected arm is neural only.
    np.savez_compressed(out/'demonstration.npz',ram=ram,masks=masks,records=records,vocabulary=vocabulary)
    configure_precision({'network_precision':'strict_fp32'});torch.set_num_threads(2);torch.manual_seed(20261128)
    actor=DemonstrationActor(len(vocabulary)).to('cuda:0');x=torch.as_tensor(ram[:-1].copy(),device='cuda:0');y=torch.as_tensor(labels,device='cuda:0')
    actor.mean.copy_((x.float()/255.).mean(0));actor.scale.copy_((x.float()/255.).std(0,unbiased=False).clamp_min(.02));opt=torch.optim.Adam(actor.parameters(),lr=.001)
    parent=ROOT/'training/level2_full_bc_refinement_v56_20261026/actor.pt'
    actor.load_state_dict(torch.load(parent,map_location='cpu',weights_only=True)['state'])
    for group in opt.param_groups:group['lr']=.00001
    hard=torch.arange(len(y),device='cuda:0')
    started=time.monotonic();rng=np.random.default_rng(20261128);steps=0;attempted=completed=0;episodes=[];status='failed';error=None
    write(out/'authority.json',dict(schema=SCHEMA,human_action_labels=True,training_rows=len(masks),unsupported_legacy_rows=int((actions<0).sum()),raw_action_vocabulary=vocabulary.tolist(),
        source_cache_sha256=digest(source),db_sha256=digest(db),model_parameters=sum(p.numel() for p in actor.parameters()),
        native_frame_cap=90000,optimizer_cap=8000,parent_bc_sha256=digest(parent),scope='One completed human route; whole route is training data; no unseen-route claim.',
        source_hashes={p:digest(ROOT/p) for p in ['scripts/run_level2_title_bc.py','src/prince_valiant_rl/training/level2_demonstration_policy.py','src/prince_valiant_rl/training/level2_score_reward.py','instrumentation/recorder_build/recorder_native.dll']}))
    try:
        with (out/'fit.jsonl').open('x') as log:
            for steps in range(1,8001):
                uniform=torch.as_tensor(rng.integers(len(y),size=256),device='cuda:0')
                difficult=hard[torch.as_tensor(rng.integers(len(hard),size=256),device='cuda:0')]
                ids_batch=torch.cat([uniform,difficult]);loss=F.cross_entropy(actor(x[ids_batch]),y[ids_batch])
                if steps in [2000,5000]:
                    for group in opt.param_groups:group['lr']*=.3
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),10.);opt.step()
                if steps%100==0:
                    with torch.no_grad():
                        prediction=torch.cat([actor(x[i:i+512]).argmax(-1) for i in range(0,len(y),512)])
                        hard=torch.where(prediction!=y)[0];correct=len(y)-len(hard)
                        if not len(hard):
                            # Exact runtime batch size removes a batching-dependent success claim.
                            prediction=torch.cat([actor(x[i:i+1]).argmax(-1) for i in range(len(y))]);hard=torch.where(prediction!=y)[0];correct=len(y)-len(hard)
                        if not len(hard):hard=torch.arange(len(y),device='cuda:0')
                    row=dict(step=steps,loss=float(loss.detach()),training_correct=correct,rows=len(y));log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                    resource_gate(out,'bc_fit',launch=False)
                    if correct==len(y):break
                if time.monotonic()-started>1200:raise TimeoutError('Behavior cloning wall cap')
        actor.eval();torch.save(dict(state=actor.state_dict(),vocabulary=vocabulary.tolist(),schema=SCHEMA,steps=steps),out/'actor.pt')
        assert digest(intro_root/'title.state')==intro_authority['state_sha256']
        span=dict(state=(intro_root/'title.state').read_bytes());(out/'entry.state').write_bytes(span['state'])
        for arm in ['neural']:
            for variant in [0]:
                folder=out/f'{arm}_{variant}';folder.mkdir();native=RecorderNative(ROOT/'roms/prince_valiant.nes');native.load_state(span['state']);rows=[native.ram_buffer().copy()];executed=[];proposals=[];hits=[];score=0;end='time_limit';entered=False;entry_frame=None
                writer=imageio_ffmpeg.write_frames(str(folder/'episode.mp4'),(256,240),fps=30,codec='libx264',pix_fmt_in='rgb24',output_params=['-crf','24'],ffmpeg_log_level='error');writer.send(None);writer.send(native.copy_screen_rgb())
                try:
                    for frame in range(36000):
                        if attempted>=90000 or time.monotonic()-started>2400:raise TimeoutError('Full imitation study cap')
                        if frame%300==0:resource_gate(out,'evaluation',launch=False)
                        pre=rows[-1]
                        with torch.inference_mode():predicted=int(vocabulary[int(actor(torch.as_tensor(pre.copy()[None],device='cuda:0')).argmax(-1))])
                        known=memory.action(pre) if arm=='state_memory' else None;mask=known if known is not None else predicted
                        attempted+=1;native.frame_advance(mask);completed+=1;post=native.ram_buffer().copy()
                        if not entered and int(pre[0xF3])==0:entered=True;entry_frame=frame
                        game=transition(pre,post,level2_scope=True) if entered else dict(score_gain=0,terminated=False,end_reason=None)
                        rows.append(post);executed.append(mask);proposals.append(predicted);hits.append(known is not None);score+=game['score_gain']
                        if (frame+1)%2==0 or game['terminated']:writer.send(native.copy_screen_rgb())
                        if frame%3000==0:print(json.dumps(dict(arm=arm,frame=frame,energy=int(post[0xFC]),lives=int(post[0xF7]),score=score)),flush=True)
                        if game['terminated']:end=game['end_reason'];break
                finally:native.close();writer.close()
                np.savez_compressed(folder/'episode.npz',ram=np.stack(rows),masks=np.array(executed,np.uint8),neural_masks=np.array(proposals,np.uint8),memory_hits=np.array(hits,bool))
                result=dict(arm=arm,entry_frame=entry_frame,title_start_frame=20700,variant=variant,frames=len(executed),score_gain=score,end_reason=end,complete=end=='level_complete',memory_hits=sum(hits),
                    dataset=str((folder/'episode.npz').relative_to(ROOT)),dataset_sha256=digest(folder/'episode.npz'),video=str((folder/'episode.mp4').relative_to(ROOT)),video_sha256=digest(folder/'episode.mp4'))
                write(folder/'result.json',result);episodes.append(result);print(json.dumps(result),flush=True)
        status='completed'
    except BaseException as exc:error=repr(exc);raise
    finally:write(out/'result.json',dict(status=status,error=error,episodes=episodes,optimizer_steps=steps,native_attempted=attempted,native_completed=completed,native_uncertain=attempted-completed,elapsed_seconds=time.monotonic()-started))
    return 0


if __name__=='__main__':raise SystemExit(main())
