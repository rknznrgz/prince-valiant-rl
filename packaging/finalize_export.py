"""Finalize and verify a newly created export; invoked by prepare_export.py."""
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
import zipfile
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def finalize(src,out):
    out=Path(out);src=Path(src)
    for p in (BASE/'templates').glob('*.py'):shutil.copyfile(p,out/'scripts'/p.name)
    # Complete the selected Level2 preprocessing chain (v3 RAM -> v4 rewards).
    for rel in ['scripts/prepare_level2_score_replay.py','config/level2_reward_current.json']:
        shutil.copyfile(src/rel,out/rel)
    allow=json.loads((out/'manifests/allowlist.json').read_text())
    if 'prepare_level2_score_replay' not in allow['commands']:
        allow['commands'].append('prepare_level2_score_replay')
        launcher=out/'run.py';text=launcher.read_text();text=text.replace("COMMANDS=[", "COMMANDS=['prepare_level2_score_replay', ")
        launcher.write_text(text,encoding='utf-8')
    dump(out/'manifests/allowlist.json',allow)
    # Retain the frozen protocol hash, without shipping private agent prompts.
    for rel in ['scripts/level1_dqfd_pilot.py','src/prince_valiant_rl/training/dqfd_pretrain_entrypoint.py']:
        p=out/rel;s=p.read_text(encoding='utf-8')
        s=s.replace('actual_protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()',
            'actual_protocol_hash = "74a7f1744688c7c7c6b6f57c059a266f8329d0d14a08c2544163552e9a4880ed"')
        s=s.replace('if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_sha256:',
            'if PROMPT11_SHA256 != protocol_sha256:')
        p.write_text(s,encoding='utf-8')
    p=out/'scripts/level1_dqfd_pilot.py';s=p.read_text()
    s=s.replace('help="show bounded future phase contracts"','help=argparse.SUPPRESS')
    p.write_text(s,encoding='utf-8')
    # Drop obsolete source-rewriting reviewer; fresh completion evidence is kept.
    obsolete=out/'scripts/review_level2_title_bc.py'
    if obsolete.exists():obsolete.unlink()
    # Absolute paths in copied video metadata are historical locations, replace
    # with source-relative identifiers and preserve original-file hash separately.
    for p in (out/'deliverables/videos').glob('*.json'):
        text=p.read_text(encoding='utf-8')
        text=text.replace(str(src).replace('\\','\\\\')+'\\\\','').replace(str(src).replace('\\','/')+'/','')
        p.write_text(text,encoding='utf-8')
    # Recompute closure AFTER portability changes and helper reduction.
    candidates={}
    for folder in ['src','scripts']:
        for p in (out/folder).rglob('*.py'):
            parts=list(p.relative_to(out).with_suffix('').parts)[1:]
            if parts[-1]=='__init__':parts.pop()
            candidates['.'.join(parts)]=p
    roots=[out/'scripts'/f'{name}.py' for name in json.loads((out/'manifests/allowlist.json').read_text())['commands']]
    roots += [out/'src/prince_valiant_rl/rl_env.py',out/'src/prince_valiant_rl/training/dqfd_level1_adapter.py',
        out/'src/prince_valiant_rl/training/dqfd_continuation.py']
    visited=set();edges={};pending=list(roots);symbols={};third_party=set()
    while pending:
        p=pending.pop()
        if p in visited:continue
        visited.add(p);rel=p.relative_to(out).as_posix();tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        mod=next(k for k,v in candidates.items() if v==p);package=mod if p.name=='__init__.py' else mod.rpartition('.')[0]
        deps=set();symbols[rel]=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:base=importlib.util.resolve_name('.'*node.level+base,package)
                names=[base]+[base+'.'+x.name for x in node.names]
            for name in names:
                if name in candidates:deps.add(candidates[name])
                elif name:third_party.add(name.split('.')[0])
        for parent in p.parents:
            if parent==out:break
            init=parent/'__init__.py'
            if init.is_file():deps.add(init)
        edges[rel]=sorted(d.relative_to(out).as_posix() for d in deps);pending.extend(deps-visited)
    removed=[]
    for p in candidates.values():
        if p not in visited:removed.append(p.relative_to(out).as_posix());p.unlink()
    dump(out/'manifests/dependency_graph.json',dict(roots=[p.relative_to(out).as_posix() for p in roots],edges=edges,
        dynamic_dependencies={'prepare_level1_dqfd_transitions --adapter':'prince_valiant_rl.training.dqfd_level1_adapter:create_adapter',
        'evaluate_level2_confirmed_completion':'scripts/evaluate_level2_transition_tail.py',
        'Windows process child':'run.py -> level1_dqfd_pilot / process_envs.level1_child',
        'native':'build_recorder.py -> nes_emu include/src + recorder_api.cpp; then build_kind18_observer.py -> sidecar source + trace_hooks.hpp'},removed_after_adaptation=removed))
    # Minimize artifact manifest to workflow-relevant inputs (not old studies).
    rows=json.loads((out/'manifests/artifacts.json').read_text())
    rows=[r for r in rows if not r['source'].startswith('training/backward_level1_v1/') or
        r['source'].startswith('training/backward_level1_v1/states/') or
        Path(r['source']).name in {'route.json','route.npz','human_session.sqlite3','initial.state'}]
    # No private prompt/session files are needed by the adapted frozen protocol.
    rows=[r for r in rows if 'worker_prompts' not in r['source'] and not r['source'].endswith(('-shm','-wal'))]
    additions=['training/level2_title_bc_v57_20261026/demonstration.npz',
        'instrumentation/recorder_build/recorder_native.dll','instrumentation/recorder_build/build_id.txt',
        'instrumentation/kind18_observer_build/kind18_observer.dll','instrumentation/kind18_observer_build/provenance.json',
        'analysis/level1_dqfd_demo_continuation_20261025/execution_budget.json']
    additions += ['analysis/level2_semantics_v3_20261025/replay/replay.npz',
        'analysis/level2_semantics_v3_20261025/replay/result.json']
    for rel in additions:
        p=src/rel
        if p.is_file() and not any(r['source']==rel for r in rows):
            rows.append(dict(filename=p.name,source=rel,destination=rel,bytes=p.stat().st_size,sha256=sha(p),included=False,
                role='Exact historical runtime/data input',access_condition='Local only; rights/provenance review required',
                acquisition='Original owner: submission_assets restore --source <checkout> --destination '+rel+'; public access unresolved',download_url=None))
    for p in [out/'docs/case_study_draft/REPORT_DRAFT.pdf',*(out/'deliverables/videos').glob('*.mp4')]:
        rel=p.relative_to(out).as_posix()
        if not any(r['destination']==rel for r in rows):
            rows.append(dict(filename=p.name,source=rel,destination=rel,bytes=p.stat().st_size,sha256=sha(p),included=True,
                role='Draft report' if p.suffix=='.pdf' else 'Submission video; see video manifest for playback/metadata',
                access_condition='Contains game artwork; publication rights review pending; no external hosting plan',
                acquisition='Included in local review export',download_url=None))
    dump(out/'manifests/artifacts.json',rows)
    # Explicitly remove a byte-identical, non-runtime contact sheet duplicate.
    duplicate=out/'analysis/level1_dqfd_demo_continuation_20261025/contact_4000.png'
    if duplicate.exists():
        assert sha(duplicate)==sha(duplicate.with_name('contact_3000.png'));duplicate.unlink()
    # Export selected authentic scalar diagnostics, without private process data.
    journal=src/'analysis/level1_dqfd_demo_continuation_20261025/production/child_receipts.jsonl'
    fields=['successful_step','loss','td1_abs_mean','td10_abs_mean','margin_mean','l2','gradient_norm','demo_fraction','q_abs_mean']
    log=out/'analysis/level1_dqfd_demo_continuation_20261025/training_metrics.csv'
    with log.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['receipt_sequence']+fields);writer.writeheader();count=0
        for line in journal.read_text().splitlines():
            row=json.loads(line);u=row['payload']
            if u.get('kind')=='update':
                writer.writerow(dict(receipt_sequence=row['sequence'],**{k:u[k] for k in fields}));count+=1
    assert count==3000
    dump(out/'manifests/log_provenance.json',dict(source=journal.relative_to(src).as_posix(),source_sha256=sha(journal),
        exported_log=log.relative_to(out).as_posix(),exported_sha256=sha(log),rows=count,
        units='successful_step=global optimizer update; loss/gradient scalars; demo_fraction=sample fraction; not episode reward'))
    # Include original training configuration, logs and plots, never executable old launchers.
    for rel in ['analysis/level1_dqfd_demo_continuation_20261025/learning_trends.png',
        'analysis/level1_dqfd_demo_continuation_20261025/candidate_comparison.png']:
        p=src/rel
        if p.is_file():shutil.copyfile(p,out/rel)
    for p in (src/'docs/case_study_draft/assets').glob('*'):
        if p.suffix in {'.png','.csv'}:
            dest=out/p.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    # All pins below are observed installed versions, not claims of online availability.
    packages=['numpy','gymnasium','pillow','imageio-ffmpeg','matplotlib','nes-py']
    (out/'requirements.txt').write_text('# Observed Windows CPython 3.13 installed versions; registry availability not checked.\n'+
        '\n'.join(f'{n}=={importlib.metadata.version(n)}' for n in packages)+'\n',encoding='utf-8')
    (out/'requirements-cuda.txt').write_text('--index-url https://download.pytorch.org/whl/cu130\n'+
        f'torch=={importlib.metadata.version("torch")}\n',encoding='utf-8')
    licenses=[]
    for name in packages+['torch']:
        dist=importlib.metadata.distribution(name)
        licenses.append(dict(package=name,version=dist.version,license=dist.metadata.get('License-Expression') or dist.metadata.get('License'),
            license_files=[str(p) for p in dist.files or [] if 'license' in str(p).lower() or 'copying' in str(p).lower()]))
    dump(out/'manifests/dependency_licenses.json',licenses)
    (out/'.gitignore').write_text('''.venv*/
__pycache__/
*.py[cod]
roms/
captures/
savestates/
instrumentation/*build/
*.dll
*.obj
*.lib
*.exp
*.state*
*.sqlite3*
training/submission_*/
analysis/submission_*/
.env*
''',encoding='utf-8')
    (out/'THIRD_PARTY_NOTICES.md').write_text('''# Redistribution review

No project license has been selected. The owner must decide before publication.
The narrowly retained nes-py native include/src files preserve their notices;
see instrumentation/nes-py/LICENSE, LICENSING.md and CITATION.cff.
The checked-in LICENSING.md explicitly records an unresolved MIT versus upstream
SimpleNES GPLv3 provenance discrepancy. This local review copy is NOT cleared
for publication; establish applicable imported-revision terms first. No entire
vendor checkout, tests, build outputs or DLLs are exported.

Python dependencies are installed separately, not vendored. Observed version and
license metadata is in manifests/dependency_licenses.json. Those metadata do not
resolve upstream native licensing. No ROM is included. Reset states, human SQLite
recordings and derived training datasets are withheld pending game-content and
recording redistribution review. Screenshots/videos/report contain game artwork;
their publication rights also need review. Exact local asset hashes are provided,
not fabricated public download links.
''',encoding='utf-8')
    write_docs(src,out,symbols,rows)
    # Remove machine-local path prefixes in staged review copies only.
    for p in out.rglob('*'):
        if p.is_file() and p.suffix in {'.py','.json','.md','.txt','.csv'}:
            text=p.read_text(encoding='utf-8-sig');original_text=text
            for prefix in [str(src).replace('\\','\\\\')+'\\\\',str(src)+'\\',str(src).replace('\\','/')+'/']:
                text=text.replace(prefix,'')
            if text!=original_text:p.write_text(text,encoding='utf-8')
    guide=out/'docs/LEVEL3_IMITATION.md'
    guide.write_text('\n'.join(line for line in guide.read_text().splitlines() if 'run.py test ' not in line)+'\n',encoding='utf-8')
    # Rebuild implementation is bundled; default output must be a fresh directory.
    tools=out/'packaging';tools.mkdir(exist_ok=True)
    for name in ['prepare_export.py','finalize_export.py']:
        shutil.copyfile(BASE/name,tools/name)
    shutil.copytree(BASE/'templates',tools/'templates',dirs_exist_ok=True)
    # Preserve source hashes of copied files and account for every adaptation.
    provenance=json.loads((out/'manifests/provenance.json').read_text())
    provenance['files']=[r for r in provenance['files'] if (out/r['destination']).is_file() and r['destination'] not in {'README.md','github.md'}]
    known={r['destination'] for r in provenance['files']}
    for p in out.rglob('*'):
        if p.is_file():
            rel=p.relative_to(out).as_posix();original=src/rel
            if rel not in known and rel not in {'README.md','github.md'} and original.is_file():provenance['files'].append(dict(source=rel,destination=rel,source_bytes=original.stat().st_size,source_sha256=sha(original)))
    for r in provenance['files']:r['export_sha256']=sha(out/r['destination']);r['adapted']=r['export_sha256']!=r['source_sha256']
    provenance['source_location']='Original checkout supplied to exporter; no absolute personal path retained'
    dump(out/'manifests/provenance.json',provenance)
    allow=json.loads((out/'manifests/allowlist.json').read_text())
    allow['initial_dependency_allowlist']=allow.get('initial_dependency_allowlist',allow['source_files'])
    allow['source_files']=sorted(r['source'] for r in provenance['files'] if not r['source'].startswith('../'))
    allow['generated_files']='See exact manifests/inventory.json; exporter/finalizer/templates define generated output contents'
    dump(out/'manifests/allowlist.json',allow)
    verify(src,out)

def write_docs(src,out,symbols,rows):
    commands='''```powershell
python run.py --list
python run.py inspect_models
python run.py submission_assets audit
# After supplying assets and building native code; these commands were NOT executed:
python run.py evaluate_selected_level1 --output analysis/submission_l1_eval
python run.py evaluate_level2_confirmed_completion --output analysis/submission_l2_eval
python run.py evaluate_level3_imitation --checkpoint training/level3_imitation_refine_v2_20261026/best.pt --output analysis/submission_l3_eval
python run.py review_level3_imitation --output analysis/submission_l3_eval
python run.py train_level3_imitation --output training/submission_l3_initial --steps 20000
python run.py train_level3_imitation --output training/submission_l3_refine --resume training/submission_l3_initial/best.pt --steps 15500 --lr 0.00001
python run.py run_level2_title_bc --output training/submission_l2_title --report analysis/submission_l2_title
python run.py prepare_level2_title_bc --output analysis/submission_l2_intro
python run.py prepare_level2_replay --output analysis/submission_l2_replay
python run.py prepare_level2_score_replay --output analysis/submission_l2_score_replay
python run.py prepare_level3_imitation --output analysis/submission_l3_preparation
python run.py prepare_backward_level1 --session captures/human/20261013_224343_044974 --output training/submission_route
python run.py train_backward_level1 --stage train --config config/backward_level1_v1.json --output training/submission_ppo --updates 1
python run.py train_backward_level1 --stage evaluate --config config/backward_level1_v1.json --output analysis/submission_ppo_eval --resume training/submission_ppo/checkpoints/latest.pt --trials 10 --evaluation-mode greedy
python run.py level1_dqfd_pilot preflight --output training/submission_dqfd
python run.py level1_dqfd_pilot pretrain --output training/submission_dqfd --execution-budget config/submission_execution_budget.json --report-dir analysis/submission_dqfd_receipts --dry-run
python run.py prepare_level1_dqfd_transitions --mode preflight --output training/submission_transitions --frame-cap 100000 --artifact-cap-bytes 12884901888 --wall-cap-seconds 1800
python run.py prepare_level1_dqfd_transitions --mode replay --output training/submission_transitions --frame-cap 100000 --artifact-cap-bytes 12884901888 --wall-cap-seconds 1800 --adapter prince_valiant_rl.training.dqfd_level1_adapter:create_adapter --resource-receipt analysis/submission_resource_gate.json --receipt-dir analysis/submission_replay_receipts
```
'''
    # Readiness table deliberately distinguishes code existence from usable assets.
    readiness='''| Workflow | Included functionality | Current readiness |
|---|---|---|
| Level 1 DQN/DQfD | Exact update-3000 checkpoint, committed receipt, DuelingQ, losses/PER, transition builder, supervised trainer and exact continuation loader, reset-389 evaluator | CPU checkpoint load verified; native evaluation blocked on route/states, identity-pinned DLL/provenance and ROM; training additionally needs pinned replay/actor/budget chain |
| Level 1 PPO | Backward PPO trainer/evaluator, process workers, curricula, rewards, temporal models | Source/help checked; route, normalization/BC/config-specific inputs required; no selected PPO result replaced by this package |
| Level 2 imitation | Title-inclusive fitting from v56 parent; neural completion runner and independent visual card/title check | Model load verified; exact data/parent/reset and native inputs withheld; not training/evaluation ready as distributed |
| Level 3 imitation | Raw-RAM fitting/refinement, selected actor, current-RAM evaluator and independent all-action/reference review | Model load verified; normalization embedded; exact preparation/dataset/reset/reference images needed for evaluation/review |
| General environment | Gymnasium RAM wrapper, ROM identity, extraction, actions and state loading | Source/help checked; nes-py wheel and gameplay state required; legacy wrapper differs from selected RecorderNative workflows |
'''
    (out/'README.md').write_text('''# Prince Valiant: curated local submission preparation

Prepared 26 October 2026. This is a reviewable local export, not a published
repository or a fully compliant assignment submission. See [github.md](github.md)
for implementation review, dependency maps, verification and blockers.

Use Windows x64 and CPython 3.13. Selected training and gameplay runners require
NVIDIA CUDA (observed torch 2.14.1+cu130); CPU is supported for checkpoint inspection.
Other platforms are unverified. The installed package versions were read locally;
their availability on a fresh package index has not been verified.

```powershell
py -3.13 -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda.txt
```

The native source route requires Visual Studio Build Tools with MSVC x64 C++14,
Windows SDK and the standard Visual Studio Installer vswhere.exe. Review
[third-party notices](THIRD_PARTY_NOTICES.md) before redistributing native code.
Build the recorder first, then its ABI2 observer:

```powershell
python instrumentation/build_recorder.py
python instrumentation/build_kind18_observer.py
```

No generated DLL is shipped. The state codec build ID must match saved states.
The Level 1 production evaluator also requires the original DLL/provenance SHA256
identities: a new compiler build may fail that check. Do not remove identity guards;
restore exact locally held binaries (rights permitting) or perform a separately
reviewed compatibility study. Builds were not run during packaging.

Supply a legally obtained ROM at `roms/prince_valiant.nes`. SHA256 must be
`96ad0a41a34fccc98bb7255890b131f77d4131cc69594c77e3ff33cd3a8fe6a5`.
The recorder verifies this identity; `submission_assets audit` checks it without
loading native code. No cartridge bytes are included.

Required local-only inputs are listed with size, hash and exact destination in
[artifact manifest](manifests/artifacts.json). There are no public download URLs.
For each reviewed input, the original owner can restore it explicitly:

```powershell
python run.py submission_assets restore --source <research-checkout> --destination <exact-manifest-destination>
```

This is an optional local preparation step, not an instruction to publish
recordings or reset states. It rejects occupied destinations and wrong hashes.
New recordings can be captured using another recorder, but do not satisfy the
selected studies' pinned data identities; exact training reproduction currently
requires owner-provided inputs. The preparation tools document extraction from
the original selected ancestry. Do not treat all copied checkpoints as proof
that the package is training-ready.

'''+readiness+'\n'+commands+'''
Commands above are actual parser examples, not results from new gameplay.
`config/submission_execution_budget.json` and `analysis/submission_resource_gate.json`
are required future reviewed inputs, NOT supplied or fabricated artifacts. The
DQfD production training command is the shown pretrain command with `--dry-run`
removed, once its input/authority/resource blockers are resolved. Transition
replay advances the emulator; it was not executed during packaging. The100000
frame cap is a bounded example and may need a reviewed budget appropriate to
all selected replay sources. Exact acceptance is not inferred from parser help.
Always choose fresh output directories. Preparation outputs use fresh names but
historical consumers still require their pinned artifact destinations; validate
the generated identity before installing it there. The PPO `latest.pt` example
must be replaced with the actual checkpoint filename written by the chosen run.
DQfD `pretrain` requires a fresh valid `--execution-budget` and `--report-dir`;
see `python run.py level1_dqfd_pilot pretrain --help`. Historical receipts are not
new authorization. Generic unchecked resume/online/final-eval phases are not
offered. Level 3 `--resume` loads actor state, then starts a fresh Adam optimizer;
it is refinement, not exact optimizer continuation. Level 2 requires its v56 parent.

The 10-page [report PDF](docs/case_study_draft/REPORT_DRAFT.pdf) is explicitly a
draft. No revised LaTeX PDF was found in the documented report location.
[Presentation](deliverables/videos/progress_presentation.mp4) is 4m23.67s,
1280x960, 30fps H.264, with three real-time full-run companions. Early/middle
runs are 24x and selected runs 8x; separate resets/models are used. No voiceover
or game audio is present. Level 2/3 episode rewards are unavailable, shown N/A.
The brief does not settle accelerated playback acceptance.

Tests, private recordings, ROMs, caches, virtual environments, obsolete studies
and agent tooling are excluded. Originals remain in the research workspace.
Models are each below GitHub's 100 MiB file limit; two companion videos exceed
50 MiB and need a deliberate storage decision before publication. No LFS pointers
or fabricated release links are included.

Rebuild into an unused directory from the original workspace with the original
Python environment (needed only to record installed dependency metadata):

```powershell
python packaging/prepare_export.py --source <research-checkout> --output <new-empty-path>
```

This uses an explicit entry-point dependency closure and artifact allowlist,
never broad source/artifact tree copies. No training or evaluation is run.
''',encoding='utf-8')
    table='''| Assignment requirement (direct DOCX read, including tables) | Evidence | Status/gap |
|---|---|---|
| Complete Levels 1–3 in at least 7/10 deterministic/low-temperature episodes | Recorded known-route clears | NOT established; no 10-episode campaign |
| Level 1 evaluation from normal game start; no lives/invincibility/level-skip manipulation | Selected reset-389 study | Normal-start compliance unaudited; do not count controlled probes |
| Separate level models allowed; documented imitation/curricula allowed | Separate DQfD / Level2 and Level3 imitation | Disclosed; not one generalizing model, not all online RL |
| Environment, training, evaluation source | Curated dependency graph and source map | Included; native/data blockers below |
| Setup/commands | README and CLI help captures | Prepared; no clean-machine run |
| Pinned dependencies and seeds | requirements files; experiment/config/model metadata | Installed versions verified; package-index availability unverified; L1 DQfD 20261024, base PPO config 20261015 (selected lineage 20261022), L2 20261128, L3 20261130; see exact retained config for each branch |
| Trained checkpoints/download | Three selected models, receipt and artifact manifest | Included; no public data downloads |
| Logs/TensorBoard | Authentic L2/L3 fit.jsonl, results; L1 learning plots and original diagnostics ledger listed below | Prepared with units; not all requested episode curves available |
| PDF 5–10 pages, optional appendices | REPORT_DRAFT.pdf (10 pages) | Draft only; revised LaTeX replacement pending |
| Report algorithm/reward/exploration/hurdles/results/time/hardware/next steps | Draft report and retained configs/logs | Draft review required; no invented training time or missing curves |
| Stable 3–5 minute 720p+ MP4, early/middle/final/full runs/steps/rewards | Four hash-verified videos | 263.67s presentation; accelerated, separate runs; Level2/3 reward N/A |
| Private Git repository with Alphabots access and one submission link | No publication attempted | Future step; access NOT granted |
| Due date | Assignment prose says 13.10.2026 (change); table says 16.10.2026 | Conflicting source dates, no deadline inferred |
| Bonus comparisons/ablation/Level4/generalization | Scoped PPO/DQfD history; no controlled reward ablation included | No extra-credit acceptance claimed; Level4 title is not competence |
'''
    maplines=['| File | Exact top-level symbols | Retained role |','|---|---|---|']
    for rel,names in sorted(symbols.items()):
        if not names:continue
        role='Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json'
        if 'dqfd' in rel:role='Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract'
        elif 'level3' in rel:role='Raw-RAM imitation or independent Level3 evidence'
        elif 'level2' in rel:role='Level2 data/actor/reward/completion evidence dependency'
        elif any(t in rel for t in ['native','game_state','scene','navigation','ram_']):role='Native state, semantic/RAM observation or action/reset pipeline'
        elif any(t in rel for t in ['backward','temporal','history','process_envs']):role='Level1 observation/history, models, PPO curricula or worker lifecycle'
        maplines.append(f'| [{rel}]({rel}) | '+', '.join('`'+n+'`' for n in names)+' | '+role+' |')
    (out/'github.md').write_text('''# GitHub preparation and report implementation review

Prepared **26 October 2026**, local package `github_submission/prince-valiant-rl`
beside original `prince-valiant-rl`. No research-root Git metadata exists: no
revision identity, clean diff or commit history is claimed. Original tests,
experiments, recordings and models are preserved. Only the research root github.md
and status handoff are updated. No publication/upload/access change occurred.

This file is the concrete retained-code handoff for the research
`docs/REPORT_PREPARATION_PROMPT.md`. The file/function map below describes the
staging implementation. [Provenance](manifests/provenance.json) records original
and exported SHA256 values; do not confuse adaptation hashes with historical
experiment code identities.

## Supported implementation and readiness

'''+readiness+'''
Exact commands are in [README](README.md), with safe `--help` captures and results
in [verification.json](manifests/verification.json). Implemented means retained
source; static/help checks do not mean runtime gameplay. Only CPU checkpoint
deserialization/state shape and integrity checks were executed. No optimizer,
training, native build/load, emulator gameplay or fresh evaluation was run.

## Dependency and required artifact map

| Workflow | Code closure and artifact/schema relationships |
|---|---|
| Native/environment | recorder_native.RecorderNative -> recorder_api.cpp -> nes_emu include/src; ROM identity and build-bound PVSTATE1 reset payload. kind18_observer.Kind18ObserverNative adds ABI2 sidecar and provenance. rl_env.PrinceValiantRamEnv is a separate legacy nes_py wrapper requiring gameplay.json. |
| Level1 | prepare_level1_dqfd_transitions -> lazy dqfd_level1_adapter:create_adapter -> Level1Env/HumanRoute -> route.json/route.npz/states; TransitionCache -> manifest/READY/identity/outcomes/transitions/windows. DuelingQ -> WindowPath -> schema/shapes/model_config inside committed checkpoint. dqfd_execution -> paired TD1/TD10, margin, PER; continuation restores Q/target/Adam/RNG/PER only from accepted update2000. Selected eval uses committed update3000 + metadata pins + original reset389 identity. |
| PPO | train_backward_level1 -> process_envs -> backward_level1.level1_child; TemporalActorCritic/FrameHistory/window_storage/control_history -> same versioned feature groups, curriculum/reward configuration. Selected score-quarantine config retained for DQfD lineage; base backward config is an additional supported trainer example, not a reproduction of every report variant. |
| Level2 | prepare_level2_replay -> selected ancestry/SQLite -> RAM replay; title preparer -> index checkpoint20700/title.state/intro.npz; run_level2_title_bc -> v4 replay + intro + SQLite labels + v56 parent -> DemonstrationActor schema level2_full_ram_bc_v55, normalization buffers inside actor.pt. Evaluation -> frozen v57 actor + entry.state + original authority; transition tail uses ordinary confirm-button presses after flag, not neural Level3 play. CompletionEvidence requires complete card then Level3 title. |
| Level3 | prepare_level3_imitation -> immutable SQLite + segmentation/index -> 49100 rows44700..93800 + frame44700 reset + preparation.json; Level3ImitationActor schema level3_raw_ram_imitation_v1 -> 2048/512/256/22, mean/scale embedded. Refinement uses same vocabulary, starts fresh Adam. evaluate -> checkpoint/preparation/reset/ROM/DLL only; review -> full saved actions/RAM + training dataset + actively rendered independent terminal references. |

Full transitive edges and explicit dynamic dispatch/source dependencies:
[dependency graph](manifests/dependency_graph.json). Missing inputs have actual
size/hash/destination and owner-restoration instructions in
[artifacts](manifests/artifacts.json). No alternative recordings are asserted
hash-compatible. Where acquisition is owner-only, public reproduction is blocked.
Historical budget receipts bind old source hashes; staged code changes invalidate
reuse for a new DQfD training launch. A fresh reviewed budget/code/protocol chain is
required, not a copied receipt or bypassed guard. This is a training-readiness gap.

## Assignment evidence and gaps

'''+table+'''
## Checkpoints, logs, report and videos

Level1: `training/level1_dqfd_demo_continuation_20261025/checkpoint_3000.pt`, SHA256
`7e44f1bc067b8cb827dfa8594a15fe7a7a51896a7bf0ab9ff104977c6f97d7c6`.
Selection and exact configuration/shape/seed/state hashes are retained in
`analysis/level1_dqfd_demo_continuation_20261025/selected_checkpoint.json`.
Updates3000 and4000 completed;5000 regressed. Only3000 is exported. This is
demo-only DQfD continuation, not online RL. Configured episode reward at3000 is
+108.09179112744023 over18273 native frames. No global best/latest substitution.

Level2: `training/level2_title_bc_v57_20261026/actor.pt`, SHA256
`900c098cbe0376cfbc5aabc4c145b83fcd96b377772c6528e1a42beb9e9efbbd`.
Level3: `training/level3_imitation_refine_v2_20261026/best.pt`, SHA256
`1481dd27a35a04e97bcb9dd25717171d16d25fa9063e18cda69e404f9f5afbcb`.
All checkpoint byte sizes/hashes are in artifacts.json; normalization lives in
the two imitation state dictionaries. Vocabulary and schema are preserved.

Authentic `fit.jsonl` logs use optimizer update indices, cross-entropy and fitted
row counts; they are not native-frame reward curves or held-out accuracy. L3
initial and refined fits are separate, total35500 applied supervised updates.
L1 training_metrics.csv contains3000 authentic update rows extracted from the
source receipt journal; manifests/log_provenance.json pins the original journal
and derived table hashes. Learning plots/README show milestone outcomes. Evidence remains scoped
to each original experiment. File inventory lists exactly which logs are included.

Draft10-page report: [PDF](docs/case_study_draft/REPORT_DRAFT.pdf), [text](docs/case_study_draft/REPORT_DRAFT.md).
Video: [presentation](deliverables/videos/progress_presentation.mp4), plus three
full-run real-time companions; actual source/output metadata in
[video manifest](deliverables/videos/manifest.original.json). Export inventory
adds current hashes. Original video manifest hash is tracked separately.
Presentation263.67s,1280x960,30fps H.264;24x early/middle,8x final playback;
no internal source-run cuts, separate resets/policies, no recorded L2/3 reward,
no voiceover/game audio. Re-decoding is not a new gameplay verification.

## Included/excluded and portability

The explicit exported inventory is [inventory.json](manifests/inventory.json).
Only import closure files, narrow native include/src + build wrappers, selected
models/evidence/report/videos and packaging tools are included. Tests, fixtures,
fake harnesses, archive copies, agent/session tooling, private notes, generated
builds, ROMs, recordings, reset states, virtual environments and duplicate models
are excluded; they remain in the source workspace. .gitignore is supplementary.

Portability changes: explicit CLI catalog with no test command; fresh output
parsers for one-shot Level2/title and Level3 preparers; removal of generated-script
rewriting and unused Level2 rollout code; direct Level2 completion runner import;
new strict selected-model inspector, local asset restorer and Level1 evaluator.
Model architectures, reward coefficients, schemas and committed models unchanged.
Original experiment hashes stay in provenance; new execution hashes differ.
The frozen DQfD protocol hash is retained directly instead of distributing private
agent prompts. Validation still requires that exact hash and a valid budget chain.
Native build identity originally hashed the entire copied nes_emu tree, including
unretained native tests/benchmarks. The minimal source build therefore has a new
tree identity: historical reset compatibility requires a separately reviewed codec
identity procedure or access to the exact original locally held binary. A new
build is a build route for new states, not proof of selected-state compatibility.

## Verification and publication blockers

See verification.json for exact subprocess argv, return codes and captured help,
CPU model integrity results, static closure audit and metadata checks. Interpreter
is the author's existing CPython environment, with isolated mode and explicit
staged paths. This is NOT a clean-machine install test. No native smoke/evaluation
check was performed. Native compiler compatibility and published wheel availability
remain unverified. DLL hash pins/build-bound states may reject clean rebuilds.
Private inputs have no public delivery plan. No general Level2/3 environment
semantic validity or unseen-route competence is asserted.

Resolve native license provenance, project license choice, dataset/reset/media
redistribution, fresh DQfD authority chain, input access, fresh-machine dependency
install/native compatibility, final report, normal-start/7-of10 evaluation and
video-format acceptance before claiming a compliant submission. Two video files
exceed50MiB; all are below100MiB, but cumulative history/storage needs a decision.
No LFS pointers, remote repository, release links or granted access are fabricated.

## Scientific limits

One known human-route clear is not a7/10 campaign. Level1 normal-start compliance
remains unaudited. Level2/3 are imitation-based and fitted on the full route,
without held-out generalization. Controlled probes remain separate from normal
game evaluation. Visible Level4 title does not demonstrate Level4 competence.

## Exact retained file/function map

'''+ '\n'.join(maplines)+'\n',encoding='utf-8')

def verify(src,out):
    refusal=subprocess.run([sys.executable,'-B',str(out/'packaging/prepare_export.py'),'--source',str(src),'--output',str(out)],capture_output=True,text=True,timeout=15)
    assert refusal.returncode!=0 and 'Refusing occupied output' in refusal.stderr
    assignment=src.parent.parent/'RL_Developer_task_Orkun.docx'
    if assignment.is_file():
        xml=ET.fromstring(zipfile.ZipFile(assignment).read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        dump(out/'manifests/assignment_source.json',dict(filename=assignment.name,bytes=assignment.stat().st_size,sha256=sha(assignment),
            paragraphs_read=len(xml.findall('.//w:p',ns)),tables_read=len(xml.findall('.//w:tbl',ns)),
            extraction='All document.xml paragraphs including those nested in tables; requirements transcribed in github.md'))
    # -I excludes cwd/PYTHONPATH and user site. Only the staging launcher injects
    # staged src/scripts; installed binary dependencies remain available.
    commands=json.loads((out/'manifests/allowlist.json').read_text())['commands']
    previous=json.loads((out/'manifests/verification.json').read_text()) if (out/'manifests/verification.json').is_file() else None
    inventory=json.loads((out/'manifests/inventory.json').read_text()) if (out/'manifests/inventory.json').is_file() else []
    old_hashes={r['path']:r['sha256'] for r in inventory}
    runtime=[p for folder in ['src','scripts'] for p in (out/folder).rglob('*.py')]+[out/'run.py']
    reuse=bool(previous and all(old_hashes.get(p.relative_to(out).as_posix())==sha(p) for p in runtime)
        and all(r['exit_code']==0 for r in previous['executions']))
    runs=previous['executions'] if reuse else []
    if not reuse:
        for args in [['--list']]+[[name,'--help'] for name in commands]+[['inspect_models']]:
            proc=subprocess.run([sys.executable,'-I','-B','-X','utf8',str(out/'run.py'),*args],cwd=out,capture_output=True,text=True,encoding='utf-8',timeout=60)
            runs.append(dict(argv=['python','-I','-B','-X','utf8','run.py',*args],exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
    compile_errors=[]
    for p in out.rglob('*.py'):
        try:compile(p.read_text(encoding='utf-8-sig'),str(p.relative_to(out)),'exec')
        except Exception as exc:compile_errors.append(dict(file=p.relative_to(out).as_posix(),error=repr(exc)))
    import_check=None if reuse else subprocess.run([sys.executable,'-I','-B','-X','utf8','-c',
        "import sys,runpy,pathlib,json; root=pathlib.Path.cwd();sys.path[:0]=[str(root/'src'),str(root/'scripts')];sys.argv=['inspect_models'];runpy.run_path(str(root/'scripts/inspect_models.py'),run_name='__main__'); modules={n:str(pathlib.Path(m.__file__).resolve()) for n,m in sys.modules.items() if n.startswith('prince_valiant_rl') and getattr(m,'__file__',None)};assert all(pathlib.Path(p).is_relative_to(root) for p in modules.values());print(json.dumps({'all_project_modules_from_staging':True,'module_count':len(modules)}))"],cwd=out,capture_output=True,text=True,encoding='utf-8',timeout=60)
    import_result=previous['isolated_project_imports'] if reuse else dict(exit_code=import_check.returncode,stdout=import_check.stdout,stderr=import_check.stderr)
    # ffmpeg -i reads metadata only; no frames are advanced or re-encoded.
    import imageio_ffmpeg
    video=[]
    video_manifest=json.loads((out/'deliverables/videos/manifest.original.json').read_text())
    for item in video_manifest['outputs']:
        p=out/'deliverables/videos'/item['file']
        proc=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-i',str(p.relative_to(out))],cwd=out,capture_output=True,text=True,timeout=30)
        assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
        assert '1280x960' in proc.stderr and '30 fps' in proc.stderr and 'h264' in proc.stderr
        video.append(dict(file=p.name,sha256=sha(p),metadata=proc.stderr,decode_performed=False))
    pdf=out/'docs/case_study_draft/REPORT_DRAFT.pdf';pdfdata=pdf.read_bytes()
    pages=len(re.findall(rb'/Type\s*/Page\b',pdfdata));assert pages==10,pages
    config=out/'training/level1_condition_B_score_quarantine_v1_20261024/ppo/config.json'
    sel=json.loads((out/'analysis/level1_dqfd_demo_continuation_20261025/selected_checkpoint.json').read_text())
    assert sha(config)==sel['metadata']['config_sha256']
    source_checks=[dict(file=r['source'],unchanged=sha(src/r['source'])==r['source_sha256'])
        for r in json.loads((out/'manifests/provenance.json').read_text())['files'] if not r['source'].startswith('../')]
    assert all(r['unchanged'] for r in source_checks)
    excluded=[];absolute=[];large=[];links=[]
    for p in out.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(out).as_posix()
        if any(part in {'tests','test','__pycache__','node_modules','.git','.venv','.venv-gpu','roms','captures'} for part in p.relative_to(out).parts) or p.name.startswith('test_') or p.name.endswith('_test.py') or p.suffix in {'.nes','.dll','.sqlite3','.pyc'}:excluded.append(rel)
        if p.stat().st_size>50*1024*1024:large.append(dict(path=rel,bytes=p.stat().st_size))
        if p.suffix in {'.py','.json','.md','.txt','.cpp','.hpp','.inc'}:
            for number,line in enumerate(p.read_text(encoding='utf-8-sig',errors='replace').splitlines(),1):
                if re.search(r'[A-Za-z]:[/\\]+(?:Users|home)[/\\]+',line):absolute.append(dict(path=rel,line=number,text=line[:250]))
        if p.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
                if target.startswith(('https:','http:','#','mailto:')):continue
                resolved=(p.parent/target.split('#')[0]).resolve()
                if not resolved.exists() and target!='manifests/inventory.json':links.append(dict(document=rel,target=target,classification='Upstream documentation reference; non-runtime file deliberately excluded' if rel.startswith('instrumentation/') else 'Historical source-only reference' if rel.startswith(('analysis/','docs/LEVEL3','deliverables/videos/README.original')) else 'Review needed'))
    dump(out/'manifests/link_audit.json',links)
    dump(out/'manifests/verification.json',dict(date='2026-10-26',environment='Existing Windows CPython environment, isolated -I/-B; no original checkout on Python import path',
        executions=runs,compile_errors=compile_errors,excluded_files_found=excluded,absolute_personal_paths=absolute,large_files_over_50MiB=large,
        isolated_project_imports=import_result,video_metadata=video,pdf_pages=pages,selected_config_hash_verified=True,
        original_source_hashes_rechecked=len(source_checks),original_sources_unchanged=True,
        cli_checks_reused_for_identical_runtime_bytes=reuse,
        exporter_refuses_occupied_output=True,
        training_executed=False,native_loaded=False,emulator_frames=0,clean_machine_verified=False))
    dump(out/'manifests/inventory.json',[dict(path=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='inventory.json'])
    print(json.dumps(dict(help_failures=[r['argv'] for r in runs if r['exit_code']],compile_errors=len(compile_errors),excluded=len(excluded),absolute_personal_paths=len(absolute))))

if __name__=='__main__':
    finalize(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
