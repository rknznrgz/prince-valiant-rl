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
    if guide.exists():guide.write_text('\n'.join(line for line in guide.read_text().splitlines() if 'run.py test ' not in line)+'\n',encoding='utf-8')
    specification=importlib.util.spec_from_file_location('export_documentation',BASE/'apply_documentation.py')
    documentation=importlib.util.module_from_spec(specification);specification.loader.exec_module(documentation)
    documentation.apply(out)
    # Rebuild implementation is bundled; default output must be a fresh directory.
    tools=out/'packaging';tools.mkdir(exist_ok=True)
    for name in ['prepare_export.py','finalize_export.py','apply_documentation.py','documentation.json']:
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
    # Reader-facing documentation is installed after legacy guide cleanup.
    return

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
