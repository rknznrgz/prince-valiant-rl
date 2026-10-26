"""Fresh extended inference recording; accept completion only after card+Level3 evidence."""
from _bootstrap import ROOT
import argparse
import json
import numpy as np
from PIL import Image
from level2_v12_rollouts import digest,write
from prince_valiant_rl.training.level2_completion_evidence import CompletionEvidence


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args(argv)
    out=(ROOT/args.output).resolve();assert ROOT.resolve() in out.parents,'Output must be within project';assert not out.exists()
    from evaluate_level2_transition_tail import main as record
    record(out)
    result=json.loads((out/'result.json').read_text());assert result['status']=='completed' and result['native_uncertain']==0
    historical=ROOT/'analysis/level2_preparation_v2_20261025/completion_boundary_trace/frame_43500.png'
    next_title=ROOT/'analysis/level2_transition_tail_v59_20261026/after_flag_0900.png'
    evidence=CompletionEvidence(np.array(Image.open(historical).convert('RGB')),np.array(Image.open(next_title).convert('RGB')))
    with np.load(out/'episode.npz',allow_pickle=False) as d:ram=d['ram']
    # Establish observed fresh entry independently of the stale incoming flag.
    zero=np.flatnonzero(ram[:,0xF3]==0)
    if len(zero):evidence.observe(int(zero[0]),ram[zero[0]],np.zeros((240,256,3),np.uint8))
    for row in result['snapshots']:
        p=ROOT/row['path'];assert digest(p)==row['sha256'];frame=result['candidate_flag_frame']+row['tail_frame']
        evidence.observe(frame,ram[frame],np.array(Image.open(p).convert('RGB')))
    write(out/'completion_evidence.json',dict(real_level2_completion=evidence.confirmed,flag_frame=evidence.flag,complete_card_frame=evidence.card,next_level_title_frame=evidence.next_level,
        flag_alone_accepted=False,classification='Corroborated visual transition' if evidence.confirmed else 'Unconfirmed; inspect recorded ending',
        historical_template_sha256=digest(historical),next_title_template_sha256=digest(next_title)))
    print(json.dumps(dict(real_level2_completion=evidence.confirmed,complete_card_frame=evidence.card,next_level_title_frame=evidence.next_level)))


if __name__=='__main__':main()
