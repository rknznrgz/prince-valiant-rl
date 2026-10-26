"""Independent neural/observation/continuity/visual ending audit for Level3 play."""
from _bootstrap import ROOT
import argparse
import json
import subprocess
import numpy as np
import torch
from PIL import Image
import imageio_ffmpeg
from level2_v12_rollouts import digest,write
from prince_valiant_rl.training.level3_imitation import Level3ImitationActor,observe_ram,SCHEMA
from prince_valiant_rl.training.temporal_models import configure_precision
from prince_valiant_rl.training.level3_evidence import match_reference_region

CARD_BOX=(65,124,238,146)
NEXT_BOX=(30,43,235,70)


def crop(image,box):
    left,top,right,bottom=box;return image[top:bottom,left:right]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args(argv);out=ROOT/args.output
    r=json.loads((out/'result.json').read_text());a=json.loads((out/'authority.json').read_text())
    assert r['status']=='completed' and r['native_attempted']==r['native_completed'] and r['native_uncertain']==0
    for path,sha in a['source_hashes'].items():assert digest(ROOT/path)==sha,path
    assert digest(ROOT/a['checkpoint'])==a['checkpoint_sha256'] and digest(ROOT/a['entry_state'])==a['entry_state_sha256']
    assert digest(out/'episode.npz')==r['dataset_sha256'] and digest(out/'episode.mp4')==r['video_sha256']
    payload=torch.load(ROOT/a['checkpoint'],map_location='cpu',weights_only=True);assert payload['schema']==SCHEMA
    prep=ROOT/'analysis/level3_imitation_v1_20261026_preparation';receipt=json.loads((prep/'preparation.json').read_text());assert payload['preparation_sha256']==digest(prep/'preparation.json')
    assert payload['dataset_sha256']==receipt['dataset_sha256']==digest(prep/'demonstration.npz')
    configure_precision({'network_precision':'strict_fp32'});torch.set_num_threads(2);actor=Level3ImitationActor(len(payload['vocabulary']));actor.load_state_dict(payload['state']);actor.to('cuda:0').eval()
    with np.load(out/'episode.npz',allow_pickle=False) as d:ram=d['ram'];masks=d['masks']
    assert ram.shape==(len(masks)+1,2048) and ram.dtype==np.uint8 and len(masks)==r['native_completed']
    with np.load(prep/'demonstration.npz',allow_pickle=False) as d:reference=d['ram'];reference_masks=d['masks']
    np.testing.assert_array_equal(ram[0],reference[0]);first_divergence=None;entered=False;candidate=None
    for i,mask in enumerate(masks):
        pre=observe_ram(ram[i])
        with torch.inference_mode():expected=payload['vocabulary'][int(actor(torch.as_tensor(pre[None],device='cuda:0')).argmax(-1))]
        assert int(mask)==expected
        if first_divergence is None and i<len(reference) and not np.array_equal(ram[i],reference[i]):first_divergence=i
        if pre[0xF3]==0:entered=True
        if entered and candidate is None and ram[i+1,0xF3]==127:candidate=i+1
    assert candidate==r['candidate_flag_frame']
    ref=ROOT/'analysis/level3_terminal_references_v2_20261026'
    reference_card=ref/'anchor_92400_after300.png';reference_title=ref/'anchor_93720_after8.png'
    card=np.array(Image.open(reference_card).convert('RGB'));next_title=np.array(Image.open(reference_title).convert('RGB'))
    assert len(np.unique(crop(card,CARD_BOX).reshape(-1,3),axis=0))>=2 and len(np.unique(crop(next_title,NEXT_BOX).reshape(-1,3),axis=0))>=2
    card_frame=next_frame=None;title_shift=None
    for snapshot in r['snapshots']:
        p=out/snapshot['path'];assert digest(p)==snapshot['sha256'];image=np.array(Image.open(p).convert('RGB'));frame=snapshot['frame']
        if candidate is not None and frame>=candidate and np.array_equal(crop(image,CARD_BOX),crop(card,CARD_BOX)):card_frame=frame if card_frame is None else card_frame
        matched_shift=match_reference_region(image,next_title,NEXT_BOX,max_vertical_shift=4)
        if card_frame is not None and frame>card_frame and matched_shift is not None:
            if next_frame is None:next_frame=frame;title_shift=matched_shift
    confirmed=card_frame is not None and next_frame is not None
    resources=[json.loads(line) for line in (out/'resources.jsonl').read_text().splitlines()];assert all(not row['gate_failures'] for row in resources)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(out/'episode.mp4'),'-f','null','-'],check=True,capture_output=True)
    write(out/'verification.json',dict(status='verified',level3_complete=confirmed,card_frame=card_frame,level4_title_frame=next_frame,
        candidate_flag_frame=candidate,flag_alone_accepted=False,all_neural_actions_recomputed=True,model_parameters=a['model_parameters'],
        first_reference_ram_divergence=first_divergence,reference_frames_compared=min(len(masks),len(reference)-1),
        exact_reference_prefix_frames=min(len(masks),len(reference)-1) if first_divergence is None else first_divergence,
        level4_title_vertical_reference_shift=title_shift,
        entry_state_sha256=a['entry_state_sha256'],checkpoint_sha256=a['checkpoint_sha256'],native_frames=len(masks),native_uncertain=0,
        observations_current_RAM_only=True,human_data_loaded_by_policy=False,action_retrieval=False,frame_counter_policy_input=False,
        mid_episode_resets=0,optimizer_steps_during_evaluation=0,video_decoded=True,resource_samples=len(resources),
        reference_card_sha256=digest(reference_card),reference_level4_title_sha256=digest(reference_title),
        scope='Human-supervised known-route completion only; whole route is training data. No unseen start/generalization or RL-only claim. No Level4 competence claim.'))
    print(json.dumps(dict(status='verified',level3_complete=confirmed,card_frame=card_frame,level4_title_frame=next_frame,native_frames=len(masks),first_divergence=first_divergence)))


if __name__=='__main__':main()
