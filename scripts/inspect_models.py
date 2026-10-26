"""Validate selected checkpoint hashes and model state shapes on CPU; no gameplay."""
from _bootstrap import ROOT
import argparse
import hashlib
import json

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    import torch
    from prince_valiant_rl.training.dqfd_pretraining import load_committed_milestone
    from prince_valiant_rl.training.level2_demonstration_policy import DemonstrationActor,SCHEMA as S2
    from prince_valiant_rl.training.level3_imitation import Level3ImitationActor,SCHEMA as S3
    selection=json.loads((ROOT/'analysis/level1_dqfd_demo_continuation_20261025/selected_checkpoint.json').read_text())
    path=ROOT/selection['checkpoint']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==selection['checkpoint_sha256']
    meta=selection['metadata']
    pins={k:v for k,v in meta.items() if k.endswith('_sha256') and k!='config_sha256'}
    model,receipt=load_committed_milestone(path,expected_update=3000,expected_pins=pins,expected_config_hash=meta['config_sha256'])
    results=[dict(level=1,sha256=receipt['checkpoint_sha256'],parameters=sum(p.numel() for p in model.parameters()),optimizer_steps=0)]
    for level,rel,ctor,schema in [
        (2,'training/level2_title_bc_v57_20261026/actor.pt',DemonstrationActor,S2),
        (3,'training/level3_imitation_refine_v2_20261026/best.pt',Level3ImitationActor,S3)]:
        path=ROOT/rel;payload=torch.load(path,map_location='cpu',weights_only=True)
        expected={2:'900c098cbe0376cfbc5aabc4c145b83fcd96b377772c6528e1a42beb9e9efbbd',
            3:'1481dd27a35a04e97bcb9dd25717171d16d25fa9063e18cda69e404f9f5afbcb'}
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[level]
        assert payload['schema']==schema
        actor=ctor(len(payload['vocabulary']));actor.load_state_dict(payload['state'],strict=True)
        assert all(torch.isfinite(t).all() for t in actor.state_dict().values())
        results.append(dict(level=level,schema=schema,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parameters=sum(p.numel() for p in actor.parameters()),optimizer_steps=0))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
