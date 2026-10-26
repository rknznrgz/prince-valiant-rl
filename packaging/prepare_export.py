"""Create a new, explicitly allowlisted submission. Never overwrite an export.

Usage: <python> prepare_export.py --source <research checkout> --output <new folder>
No training, native loading, network, or emulator execution is performed.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

ROOTS = [
    'scripts/level1_dqfd_pilot.py', 'scripts/prepare_level1_dqfd_transitions.py',
    'scripts/train_backward_level1.py', 'scripts/prepare_backward_level1.py',
    'scripts/prepare_level2_replay.py', 'scripts/prepare_level2_title_bc.py',
    'scripts/prepare_level2_score_replay.py',
    'scripts/run_level2_title_bc.py', 'scripts/evaluate_level2_confirmed_completion.py',
    'scripts/evaluate_level2_transition_tail.py', 'scripts/review_level2_title_bc.py',
    'scripts/prepare_level3_imitation.py', 'scripts/train_level3_imitation.py',
    'scripts/evaluate_level3_imitation.py', 'scripts/review_level3_imitation.py',
    'src/prince_valiant_rl/rl_env.py',
    'src/prince_valiant_rl/training/dqfd_level1_adapter.py',
    'src/prince_valiant_rl/training/dqfd_continuation.py',
    'src/prince_valiant_rl/training/dqfd_production.py',
]
COMMANDS = ['level1_dqfd_pilot', 'prepare_level1_dqfd_transitions',
    'train_backward_level1', 'prepare_backward_level1', 'prepare_level2_replay',
    'prepare_level2_score_replay',
    'prepare_level2_title_bc', 'run_level2_title_bc',
    'evaluate_level2_confirmed_completion', 'prepare_level3_imitation',
    'train_level3_imitation', 'evaluate_level3_imitation', 'review_level3_imitation',
    'submission_assets', 'evaluate_selected_level1', 'inspect_models']

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def module(p):
    parts=list(p.with_suffix('').parts)
    if parts[0]=='src': parts=parts[1:]
    elif parts[0]=='scripts': parts=parts[1:]
    if parts[-1]=='__init__': parts.pop()
    return '.'.join(parts)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); src=a.source.resolve(); out=a.output.resolve()
    if out.exists(): raise SystemExit('Refusing occupied output: '+str(out))
    # Index names, but copy only the graph reachable from selected entry points.
    index={module(p.relative_to(src)):p.relative_to(src) for folder in ['src','scripts']
           for p in (src/folder).rglob('*.py') if '__pycache__' not in p.parts}
    selected=set(); edges={}; external=set(); pending=list(ROOTS)
    while pending:
        rel=Path(pending.pop()); key=rel.as_posix()
        if key in selected: continue
        if not (src/rel).is_file(): raise FileNotFoundError(rel)
        selected.add(key); deps=set(); tree=ast.parse((src/rel).read_text(encoding='utf-8-sig'))
        mod=module(rel); package=mod if rel.name=='__init__.py' else mod.rpartition('.')[0]
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import): names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    base=importlib.util.resolve_name('.'*node.level+base,package)
                names=[base]+[base+'.'+x.name for x in node.names]
            for name in names:
                if name in index: deps.add(index[name].as_posix())
                elif name and name.split('.')[0] not in {'prince_valiant_rl'}:
                    external.add(name.split('.')[0])
        # Include package initializers even when Python loads them implicitly.
        for parent in rel.parents:
            init=parent/'__init__.py'
            if parent!=Path('.') and (src/init).is_file(): deps.add(init.as_posix())
        edges[key]=sorted(deps); pending.extend(deps-selected)
    # Runtime source reads are explicit; this generated-script preparer is
    # adapted below so it no longer writes or reads historical launcher variants.
    native=['instrumentation/build_recorder.py','instrumentation/recorder_api.cpp',
        'instrumentation/build_kind18_observer.py','instrumentation/kind18_observer.hpp',
        'instrumentation/kind18_sidecar_api.inc','instrumentation/trace_hooks.hpp',
        'instrumentation/nes-py/LICENSE','instrumentation/nes-py/LICENSING.md',
        'instrumentation/nes-py/CITATION.cff']
    for folder in ['include','src']:
        native += [p.relative_to(src).as_posix() for p in
            (src/'instrumentation/nes-py/nes_emu'/folder).rglob('*') if p.is_file()]
    selected.update(native)
    # Curated configs are expanded only by literal config references in closure.
    configs={'config/ram_map.json','config/backward_level1_v1.json',
        'config/dqfd_demo_continuation_20261025.json',
        'config/dqfd_demonstration_extension_20261025.json'}
    for rel in list(selected):
        if rel.endswith('.py'):
            text=(src/rel).read_text(encoding='utf-8-sig')
            configs.update(re.findall(r'config/[A-Za-z0-9_./-]+\.(?:json|npz)',text))
    selected.update(p for p in configs if (src/p).is_file())
    evidence=[
        'analysis/level1_dqfd_demo_continuation_20261025',
        'analysis/level1_dqfd_extended_study_20261025',
        'analysis/level2_title_bc_v57_20261026',
        'analysis/level2_standalone_completion_v58_20261026',
        'analysis/level2_transition_tail_v59_20261026',
        'analysis/level3_imitation_evaluation_v2_20261026',
        'training/level2_title_bc_v57_20261026',
        'training/level3_imitation_v1_20261026',
        'training/level3_imitation_refine_v2_20261026',
    ]
    for folder in evidence:
        for p in (src/folder).glob('*'):
            if p.is_file() and (p.name in ['README.md','selected_checkpoint.json','fit.jsonl',
                'result.json','verification.json','authority.json','postrun_budget.json']
                or p.suffix=='.png') and p.stat().st_size<5*1024*1024:
                selected.add(p.relative_to(src).as_posix())
    models=['training/level1_dqfd_demo_continuation_20261025/checkpoint_3000.pt',
        'training/level1_dqfd_demo_continuation_20261025/checkpoint_3000.pt.receipt.json',
        'training/level2_title_bc_v57_20261026/actor.pt',
        'training/level3_imitation_refine_v2_20261026/best.pt']
    selected.update(models)
    docs=['docs/LEVEL3_IMITATION.md','docs/case_study_draft/REPORT_DRAFT.pdf',
        'docs/case_study_draft/REPORT_DRAFT.md','docs/case_study_draft/assets/source_manifest.json']
    selected.update(docs)
    selected.add('training/level1_condition_B_score_quarantine_v1_20261024/ppo/config.json')
    out.mkdir(parents=True)
    provenance=[]
    for rel in sorted(selected):
        p=src/rel; dest=out/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        if any(x in p.parts for x in ['tests','test','benchmark']) or p.name.startswith('test_'):
            raise ValueError('Excluded test reached closure: '+rel)
        shutil.copyfile(p,dest)
        provenance.append(dict(source=rel,destination=rel,source_bytes=p.stat().st_size,source_sha256=sha(p)))
    # Portable dispatcher: helper scripts are retained without advertising them.
    (out/'run.py').write_text('''"""Curated, explicit submission CLI; no research-tree fallback."""
from pathlib import Path
import os,runpy,sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
COMMANDS='''+repr(COMMANDS)+'''
def main():
    if len(sys.argv)<2 or sys.argv[1] in ('--help','-h','--list'):
        print('Usage: python run.py COMMAND [arguments]\\nCommands: '+', '.join(COMMANDS));return
    command=sys.argv[1]
    if command not in COMMANDS:raise SystemExit('Unknown command; use --list')
    os.chdir(ROOT);sys.argv=[str(ROOT/'scripts'/f'{command}.py'),*sys.argv[2:]]
    runpy.run_path(sys.argv[0],run_name='__main__')
if __name__=='__main__':main()
''',encoding='utf-8')
    # Historical one-shot tools get actual argument parsers before any effects.
    def edit(rel,old,new):
        p=out/rel;s=p.read_text(encoding='utf-8');assert old in s,(rel,old);p.write_text(s.replace(old,new),encoding='utf-8')
    edit('scripts/run_level2_title_bc.py','def main():\n    out=ROOT/\'training\'/NAME;out.mkdir(exist_ok=False);report=ROOT/\'analysis\'/NAME;report.mkdir(exist_ok=False);',
        "def main():\n    import argparse\n    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--report',required=True);args=p.parse_args()\n    out=ROOT/args.output;report=ROOT/args.report\n    if out.exists() or report.exists():raise FileExistsError('Choose fresh output and report paths')\n    out.mkdir(parents=True);report.mkdir(parents=True);")
    # Remove unused memory/retrieval and span loader dependency from selected training.
    edit('scripts/run_level2_title_bc.py','from prince_valiant_rl.level2_validation_env import _load_span\n','')
    edit('scripts/run_level2_title_bc.py','    memory=KnownStatePolicy(ram[:-1],masks)','    memory=None  # Selected arm is neural only.')
    edit('scripts/prepare_level2_title_bc.py','def main():\n    prep=ROOT/\'analysis\'/f\'{NAME}_preparation\';assert not prep.exists()',
        "def main():\n    import argparse\n    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()\n    prep=ROOT/args.output;assert not prep.exists()")
    p=out/'scripts/prepare_level2_title_bc.py';s=p.read_text();s=s[:s.index("    old='level2_full_bc_refinement")]+"    print(str(prep))\n\nif __name__=='__main__':main()\n";p.write_text(s,encoding='utf-8')
    edit('scripts/evaluate_level2_transition_tail.py','def main():\n    out=ROOT/\'analysis\'/NAME;out.mkdir(exist_ok=False);',
        "def main(output=None):\n    if output is None:raise ValueError('Use evaluate_level2_confirmed_completion --output')\n    out=output;out.mkdir(parents=True,exist_ok=False);")
    p=out/'scripts/evaluate_level2_confirmed_completion.py';s=p.read_text();start=s.index('    source=(ROOT/');end=s.index('    result=json.loads',start)
    s=s[:start]+"    from evaluate_level2_transition_tail import main as record\n    record(out)\n"+s[end:];p.write_text(s,encoding='utf-8')
    edit('scripts/prepare_level3_imitation.py',"    out=ROOT/'analysis'/f'{NAME}_preparation';out.mkdir(exist_ok=False);",
        "    import argparse\n    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()\n    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False);")
    # Shared helper reduced to the exact used functions; no obsolete rollout driver.
    p=out/'scripts/level2_v12_rollouts.py';s=p.read_text();tree=ast.parse(s);lines=s.splitlines(keepends=True)
    keep=''.join(lines[n.lineno-1:n.end_lineno] and ''.join(lines[n.lineno-1:n.end_lineno])+'\n\n'
        for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and n.name in {'digest','write','resource_gate'})
    p.write_text(keep,encoding='utf-8')
    # Do not advertise unimplemented pilot phases.
    edit('scripts/level1_dqfd_pilot.py','choices=("preflight", "acceptance", "pretrain", "retention", "online", "final-eval")','choices=("preflight", "pretrain")')
    # Every implicit historical output is replaced by an explicit user choice.
    edit('scripts/train_level3_imitation.py',"parser.add_argument('--output',default='training/level3_imitation_v1_20261026')","parser.add_argument('--output',required=True)")
    # Workflows still use hash-bound artifact destinations. Assets are restored there,
    # never looked up in the original checkout.
    dump(out/'manifests/dependency_graph.json',dict(roots=ROOTS,edges=edges,external_import_roots=sorted(external)))
    # Record restricted/local-only artifacts without copying states or private DBs.
    requested=set()
    for rel in selected:
        if rel.endswith('.py') or rel.endswith('.json'):
            text=(out/rel).read_text(encoding='utf-8-sig')
            for value in re.findall(r'[\"\']([^\"\'\n]+)[\"\']',text):
                value=value.replace('\\\\','/').replace('\\','/')
                if value.startswith(('training/','analysis/','captures/','savestates/','config/','instrumentation/')):
                    p=src/value
                    if p.is_file() and value not in selected:requested.add(value)
    # Explicit runtime/data/build and independently reviewed reference closure.
    for folder in ['training/backward_level1_v1',
        'training/level1_dqfd_transitions_reconstruction_v4_20261024',
        'analysis/level3_imitation_v1_20261026_preparation',
        'analysis/level3_terminal_references_v2_20261026',
        'analysis/level2_title_bc_v57_20261026_preparation',
        'analysis/level23_human_segmentation_20261023']:
        for p in (src/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:requested.add(p.relative_to(src).as_posix())
    requested.update(['training/level2_title_bc_v57_20261026/entry.state',
        'training/level2_full_bc_refinement_v56_20261026/actor.pt',
        'analysis/level2_score_reward_v4_20261025/replay_v4/replay.npz',
        'analysis/level2_preparation_v2_20261025/completion_boundary_trace/frame_43500.png',
        'captures/human/20261013_224343_044974/recording.sqlite3'])
    artifacts=[]
    for rel in sorted(requested):
        p=src/rel
        if not p.is_file():continue
        artifacts.append(dict(filename=p.name,source=rel,destination=rel,bytes=p.stat().st_size,sha256=sha(p),
            included=False,role='Pinned workflow input; see dependency graph and owning guide',
            access_condition='Local research data/state: redistribution and game-content rights unresolved; do not publish automatically',
            acquisition='Original owner can restore this exact local file with submission_assets restore --source <research checkout>. No public download exists. New recordings do not reproduce pinned hashes.',download_url=None))
    for rel in models:
        p=out/rel
        artifacts.append(dict(filename=p.name,source=rel,destination=rel,bytes=p.stat().st_size,sha256=sha(p),included=True,role='Selected model or committed model receipt',access_condition='Project license decision pending',acquisition='Included',download_url=None))
    video_root=src.parent/'submission_demo_videos'
    video_manifest=json.loads((video_root/'manifest.json').read_text())
    for item in video_manifest['outputs']:
        p=video_root/item['file'];assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
        dest=out/'deliverables/videos'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        provenance.append(dict(source='../submission_demo_videos/'+p.name,destination=dest.relative_to(out).as_posix(),source_bytes=p.stat().st_size,source_sha256=sha(p)))
    shutil.copyfile(video_root/'README.md',out/'deliverables/videos/README.original.md')
    shutil.copyfile(video_root/'manifest.json',out/'deliverables/videos/manifest.original.json')
    dump(out/'manifests/artifacts.json',artifacts)
    for row in provenance:
        row['export_sha256']=sha(out/row['destination']);row['adapted']=row['export_sha256']!=row['source_sha256']
    dump(out/'manifests/provenance.json',dict(date='2026-10-26',source_revision=None,git_metadata='Absent at research root',files=provenance))
    dump(out/'manifests/allowlist.json',dict(source_files=sorted(selected),commands=COMMANDS))
    from finalize_export import finalize
    finalize(src,out)
    print(json.dumps(dict(output=str(out),source_files=len(selected),external_artifacts=len(artifacts),bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))))

if __name__=='__main__':main()
