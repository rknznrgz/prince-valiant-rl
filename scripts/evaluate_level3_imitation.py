"""Standalone current-RAM neural Level3 play, retaining the real ending and next title."""
from _bootstrap import ROOT
import argparse
import json
import time
import numpy as np
import torch
from PIL import Image
import imageio_ffmpeg
from level2_v12_rollouts import digest,write,resource_gate
from prince_valiant_rl.recorder_native import RecorderNative
from prince_valiant_rl.training.level3_imitation import Level3ImitationActor,observe_ram,SCHEMA
from prince_valiant_rl.training.temporal_models import configure_precision


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',required=True);parser.add_argument('--output',required=True);parser.add_argument('--max-frames',type=int,default=53000);args=parser.parse_args(argv)
    assert 1<=args.max_frames<=65000
    out=ROOT/args.output;out.mkdir(exist_ok=False);resource_gate(out,'launch',launch=True)
    prep=ROOT/'analysis/level3_imitation_v1_20261026_preparation';receipt=json.loads((prep/'preparation.json').read_text());cp=next(x for x in receipt['checkpoints'] if x['frame']==44700)
    state=ROOT/cp['state'];assert digest(state)==cp['state_sha256']
    checkpoint=ROOT/args.checkpoint;payload=torch.load(checkpoint,map_location='cpu',weights_only=True);assert payload['schema']==SCHEMA and payload['vocabulary']==receipt['vocabulary']
    configure_precision({'network_precision':'strict_fp32'});torch.set_num_threads(2);actor=Level3ImitationActor(len(payload['vocabulary']));actor.load_state_dict(payload['state']);actor.to('cuda:0').eval().requires_grad_(False)
    native=RecorderNative(ROOT/'roms/prince_valiant.nes');native.load_state(state.read_bytes());rows=[native.ram_buffer().copy()];masks=[];entered=False;candidate=None;attempted=completed=0;status='failed';error=None;snapshots=[];started=time.monotonic()
    write(out/'authority.json',dict(schema=SCHEMA,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest(checkpoint),entry_state=str(state.relative_to(ROOT)),entry_state_sha256=digest(state),
        model_parameters=sum(p.numel() for p in actor.parameters()),vocabulary=payload['vocabulary'],current_RAM_only=True,human_data_loaded_at_inference=False,
        action_retrieval=False,sequence_cursor=False,frame_index_policy_input=False,mid_episode_resets=0,optimizer_steps=0,native_cap=args.max_frames,
        source_hashes={p:digest(ROOT/p) for p in ['scripts/evaluate_level3_imitation.py','src/prince_valiant_rl/training/level3_imitation.py','instrumentation/recorder_build/recorder_native.dll']}))
    writer=imageio_ffmpeg.write_frames(str(out/'episode.mp4'),(256,240),fps=30,codec='libx264',pix_fmt_in='rgb24',output_params=['-crf','22'],ffmpeg_log_level='error');writer.send(None);writer.send(native.copy_screen_rgb())
    try:
        for frame in range(args.max_frames):
            if time.monotonic()-started>900:raise TimeoutError('Level3 evaluation900s cap')
            if frame%300==0:resource_gate(out,'evaluation',launch=False)
            pre=observe_ram(rows[-1])
            with torch.inference_mode():mask=payload['vocabulary'][int(actor(torch.as_tensor(pre[None],device='cuda:0')).argmax(-1))]
            if int(pre[0xF3])==0:entered=True
            attempted+=1;native.frame_advance(int(mask));completed+=1;post=native.ram_buffer().copy();rows.append(post);masks.append(mask)
            if entered and candidate is None and int(post[0xF3])==127:
                candidate=frame+1;(out/'candidate_flag.state').write_bytes(native.dump_state());print(json.dumps(dict(candidate_flag_frame=candidate)),flush=True)
            relative=completed-candidate if candidate is not None else None
            if completed in [60,120] or relative in [0,30,60,120,240,300,600,900,1200,1500,1800,2100]:
                file=out/f'frame_{completed:06d}.png';Image.fromarray(native.copy_screen_rgb()).save(file);snapshots.append(dict(frame=completed,relative_to_candidate=relative,path=file.name,sha256=digest(file)))
            if completed%2==0:writer.send(native.copy_screen_rgb())
            if frame%6000==0:print(json.dumps(dict(frame=frame,raw_F3=int(post[0xF3]),raw_energy=int(post[0xFC]),raw_lives=int(post[0xF7]))),flush=True)
            if relative==2100:break
        status='completed'
    except BaseException as exc:error=repr(exc);raise
    finally:
        native.close();writer.close();np.savez_compressed(out/'episode.npz',ram=np.stack(rows),masks=np.array(masks,np.uint8))
        write(out/'result.json',dict(status=status,error=error,candidate_flag_frame=candidate,snapshots=snapshots,native_attempted=attempted,native_completed=completed,native_uncertain=attempted-completed,
            dataset_sha256=digest(out/'episode.npz'),video_sha256=digest(out/'episode.mp4'),completion_confirmed=False,classification='Requires Level3 complete card and subsequent Level4 evidence; no Level2 byte semantics assumed.',elapsed_seconds=time.monotonic()-started))
    print(json.dumps(dict(status=status,frames=completed,candidate_flag_frame=candidate)))


if __name__=='__main__':main()
