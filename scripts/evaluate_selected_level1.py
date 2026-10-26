"""Greedy selected update-3000 evaluation under the original reset-389 contract."""
from _bootstrap import ROOT
import argparse
import json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    from prince_valiant_rl.training.dqfd_production import production_reset389_episode
    selection=json.loads((ROOT/'analysis/level1_dqfd_demo_continuation_20261025/selected_checkpoint.json').read_text())
    out=ROOT/a.output
    out.mkdir(parents=True,exist_ok=False)
    meta=selection['metadata']
    pins={k:meta[k] for k in ['manifest_sha256','READY_sha256','identity_sha256','actor_sha256','protocol_sha256','code_sha256','budget_sha256']}
    result=production_reset389_episode(root=ROOT,output_dir=out,receipt_path=out/'receipts.jsonl',
        attempt_id='submission-reset389',update=3000,checkpoint_path=ROOT/selection['checkpoint'],
        checkpoint_sha256=selection['checkpoint_sha256'],pins=pins,inference_device=a.device)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
