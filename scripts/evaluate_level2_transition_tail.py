"""Continue neural play beyond F3, saving actual completion/next-level visual evidence."""
from _bootstrap import ROOT
import json
import time
import numpy as np
import torch
import imageio_ffmpeg
from PIL import Image
from level2_v12_rollouts import digest,write,resource_gate
from prince_valiant_rl.recorder_native import RecorderNative
from prince_valiant_rl.training.level2_demonstration_policy import DemonstrationActor
from prince_valiant_rl.training.temporal_models import configure_precision

NAME='level2_transition_tail_v59_20261026'


def main(output=None):
    if output is None:raise ValueError('Use evaluate_level2_confirmed_completion --output')
    out=output;out.mkdir(parents=True,exist_ok=False);resource_gate(out,'launch',launch=True)
    original=ROOT/'analysis/level2_standalone_completion_v58_20261026';a=json.loads((original/'authority.json').read_text())
    checkpoint=ROOT/a['checkpoint'];state=ROOT/a['state'];assert digest(checkpoint)==a['checkpoint_sha256'] and digest(state)==a['state_sha256']
    configure_precision({'network_precision':'strict_fp32'});torch.set_num_threads(2)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True);actor=DemonstrationActor(len(payload['vocabulary']));actor.load_state_dict(payload['state']);actor.to('cuda:0').eval()
    native=RecorderNative(ROOT/'roms/prince_valiant.nes');native.load_state(state.read_bytes())
    video=out/'full_transition.mp4';writer=imageio_ffmpeg.write_frames(str(video),(256,240),fps=30,codec='libx264',pix_fmt_in='rgb24',output_params=['-crf','20'],ffmpeg_log_level='error');writer.send(None);writer.send(native.copy_screen_rgb())
    rows=[native.ram_buffer().copy()];masks=[];entered=False;flag_frame=None;snapshots=[];attempted=completed=0;status='failed';error=None;started=time.monotonic()
    try:
        for frame in range(26000):
            if time.monotonic()-started>600:raise TimeoutError('Transition audit wall cap')
            if frame%300==0:resource_gate(out,'evaluation',launch=False)
            pre=rows[-1]
            if int(pre[0xF3])==0:entered=True
            if flag_frame is None:
                with torch.inference_mode():mask=payload['vocabulary'][int(actor(torch.as_tensor(pre.copy()[None],device='cuda:0')).argmax(-1))]
            else:
                tail=frame-flag_frame
                # Keep the card visible, then ordinary confirm-button presses.
                mask=1 if tail>=600 and tail%120<8 else 0
            attempted+=1;native.frame_advance(int(mask));completed+=1;post=native.ram_buffer().copy();rows.append(post);masks.append(mask)
            if entered and flag_frame is None and int(post[0xF3])==127:
                flag_frame=frame+1;(out/'flag.state').write_bytes(native.dump_state())
                print(json.dumps(dict(candidate_flag_frame=flag_frame)),flush=True)
            relative=completed-flag_frame if flag_frame is not None else None
            if relative in [0,120,300,600,900,1200,1500,1800,2100,2400]:
                file=out/f'after_flag_{relative:04d}.png';Image.fromarray(native.copy_screen_rgb()).save(file)
                snapshots.append(dict(tail_frame=relative,path=str(file.relative_to(ROOT)),sha256=digest(file),F3=int(post[0xF3]),energy=int(post[0xFC]),lives=int(post[0xF7])))
                print(json.dumps(snapshots[-1]),flush=True)
            if completed%2==0:writer.send(native.copy_screen_rgb())
            if relative==2400:break
        status='completed'
    except BaseException as exc:error=repr(exc);raise
    finally:
        native.close();writer.close();np.savez_compressed(out/'episode.npz',ram=np.stack(rows),masks=np.array(masks,np.uint8))
        write(out/'result.json',dict(status=status,error=error,candidate_flag_frame=flag_frame,snapshots=snapshots,native_attempted=attempted,native_completed=completed,native_uncertain=attempted-completed,
            checkpoint_sha256=digest(checkpoint),state_sha256=digest(state),dataset_sha256=digest(out/'episode.npz'),video_sha256=digest(video),optimizer_steps=0,
            completed=False,classification='pending independent visual and terminal audit; F3 alone not accepted'))


if __name__=='__main__':main()
