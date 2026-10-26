"""Train a separate Level3 raw-RAM human-imitation model; never run the emulator here."""
from _bootstrap import ROOT
import argparse
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from level2_v12_rollouts import digest,write,resource_gate
from prince_valiant_rl.training.level3_imitation import Level3ImitationActor,SCHEMA
from prince_valiant_rl.training.temporal_models import configure_precision


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--resume');parser.add_argument('--steps',type=int,default=20000);parser.add_argument('--lr',type=float,default=.001);args=parser.parse_args(argv)
    assert 1<=args.steps<=30000 and 0<args.lr<=.001
    out=ROOT/args.output;out.mkdir(exist_ok=False);resource_gate(out,'launch',launch=True)
    prep=ROOT/'analysis/level3_imitation_v1_20261026_preparation';receipt=json.loads((prep/'preparation.json').read_text());assert receipt['status']=='prepared' and not receipt['current_ram_action_conflicts']
    dataset=prep/'demonstration.npz';assert digest(dataset)==receipt['dataset_sha256']
    with np.load(dataset,allow_pickle=False) as d:ram=d['ram'][:-1].copy();labels=d['labels'].copy()
    configure_precision({'network_precision':'strict_fp32'});torch.set_num_threads(2);torch.manual_seed(20261130)
    device='cuda:0';x=torch.as_tensor(ram,device=device);y=torch.as_tensor(labels,device=device)
    actor=Level3ImitationActor(len(receipt['vocabulary'])).to(device)
    actor.mean.copy_((x.float()/255.).mean(0));actor.scale.copy_((x.float()/255.).std(0,unbiased=False).clamp_min(.02))
    if args.resume:
        parent=ROOT/args.resume;payload=torch.load(parent,map_location='cpu',weights_only=True);assert payload['schema']==SCHEMA and payload['vocabulary']==receipt['vocabulary'];actor.load_state_dict(payload['state'])
    opt=torch.optim.Adam(actor.parameters(),lr=args.lr);rng=np.random.default_rng(20261130);hard=np.arange(len(y));best=-1;best_step=0;steps=0;status='failed';error=None;started=time.monotonic();runtime_exact=False
    authority=dict(schema=SCHEMA,preparation_sha256=digest(prep/'preparation.json'),dataset_sha256=digest(dataset),observation='Current2048 uint8 RAM bytes, no source indices, no future observations.',
        vocabulary=receipt['vocabulary'],human_demonstration_rows=len(y),model_parameters=sum(p.numel() for p in actor.parameters()),optimizer_cap=args.steps,
        parent=args.resume,parent_sha256=digest(ROOT/args.resume) if args.resume else None,initial_lr=args.lr,native_frames=0,
        source_hashes={p:digest(ROOT/p) for p in ['scripts/train_level3_imitation.py','src/prince_valiant_rl/training/level3_imitation.py']})
    write(out/'authority.json',authority)
    def save(path,step,correct,exact):
        temp=path.with_suffix('.tmp');torch.save(dict(state=actor.state_dict(),schema=SCHEMA,vocabulary=receipt['vocabulary'],optimizer_steps=step,
            training_correct=correct,training_rows=len(y),runtime_single_row_exact=exact,preparation_sha256=authority['preparation_sha256'],dataset_sha256=authority['dataset_sha256']),temp);temp.replace(path)
    try:
        with (out/'fit.jsonl').open('x') as log:
            for steps in range(1,args.steps+1):
                if time.monotonic()-started>2400:raise TimeoutError('Level3 cloning2400s cap')
                ids=rng.integers(len(y),size=1024)
                if steps>3000 or args.resume:ids[:512]=hard[rng.integers(len(hard),size=512)]
                batch=torch.as_tensor(ids,device=device);loss=F.cross_entropy(actor(x[batch]),y[batch])
                opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),10.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Nonfinite Level3 update')
                opt.step()
                if steps in [5000,10000,15000]:
                    for group in opt.param_groups:group['lr']*=.3
                if steps%250==0:
                    with torch.no_grad():
                        predictions=torch.cat([actor(x[i:i+512]).argmax(-1) for i in range(0,len(y),512)])
                        wrong=torch.where(predictions!=y)[0];correct=len(y)-len(wrong);runtime_exact=False
                        if not len(wrong):
                            predictions=torch.cat([actor(x[i:i+1]).argmax(-1) for i in range(len(y))]);wrong=torch.where(predictions!=y)[0];correct=len(y)-len(wrong);runtime_exact=not len(wrong)
                    hard=wrong.cpu().numpy() if len(wrong) else np.arange(len(y))
                    row=dict(step=steps,loss=float(loss.detach()),correct=correct,rows=len(y),runtime_exact=runtime_exact,lr=opt.param_groups[0]['lr']);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True);resource_gate(out,'fitting',launch=False)
                    if correct>best or (correct==best and runtime_exact):best=correct;best_step=steps;save(out/'best.pt',steps,correct,runtime_exact)
                    if runtime_exact:break
            save(out/'last.pt',steps,correct,runtime_exact);status='completed'
    except BaseException as exc:error=repr(exc);raise
    finally:write(out/'result.json',dict(status=status,error=error,optimizer_steps=steps,best_step=best_step,best_correct=best,rows=len(y),runtime_single_row_exact=runtime_exact,native_frames=0,elapsed_seconds=time.monotonic()-started))
    return 0


if __name__=='__main__':main()
