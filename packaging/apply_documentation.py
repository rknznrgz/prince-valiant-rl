"""Install the repository's reader-facing documentation and validate local links."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

REMOVED=[
    'github.md','docs/LEVEL3_IMITATION.md','deliverables/videos/README.original.md',
    'analysis/level1_dqfd_demo_continuation_20261025/README.md',
    'analysis/level1_dqfd_extended_study_20261025/README.md',
    'analysis/level2_title_bc_v57_20261026/README.md',
    'analysis/level2_standalone_completion_v58_20261026/README.md',
    'analysis/level2_transition_tail_v59_20261026/README.md',
    'analysis/level3_imitation_evaluation_v2_20261026/README.md',
]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def apply(root):
    root=Path(root).resolve()
    text=json.loads(Path(__file__).with_name('documentation.json').read_text(encoding='utf-8'))
    retired=[]
    for rel in REMOVED:
        path=(root/rel).resolve()
        assert path.is_relative_to(root) and path.suffix=='.md'
        if path.is_file():
            retired.append(dict(path=rel,previous_sha256=sha(path),reason='Replaced by consolidated reader-facing documentation'))
            path.unlink()
    for rel,content in text.items():
        path=root/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8')
    # Third-party attribution is preserved; clarify the current publication status.
    (root/'THIRD_PARTY_NOTICES.md').write_text('''# Third-party notices

No project-wide license has been selected. Publication of this repository does
not resolve the licensing and rights questions below.

The retained nes-py native sources preserve their original notices. See
`instrumentation/nes-py/LICENSE`, `LICENSING.md`, and `CITATION.cff`. That licensing
guide records an unresolved discrepancy between nes-py's MIT declaration and
upstream SimpleNES GPLv3 provenance. Applicable imported-revision terms and any
required permissions have not been established; the MIT notice alone is not a
clearance for all native source material.

Python dependencies are installed separately and retain their own licenses.
Observed version/license metadata is in `manifests/dependency_licenses.json`.

No ROM is included. Game artwork appears in screenshots, videos, and the draft
report; no rights to that artwork are granted by this project. Private recordings,
reset states, and derived training datasets are withheld pending redistribution
review. Required local inputs and their access conditions are recorded in
`manifests/artifacts.json`.
''',encoding='utf-8')
    symbols={}
    for folder in ['src','scripts']:
        for p in sorted((root/folder).rglob('*.py')):
            tree=ast.parse(p.read_text(encoding='utf-8-sig'))
            symbols[p.relative_to(root).as_posix()]=[
                dict(name=n.name,line=n.lineno,kind=type(n).__name__)
                for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
    dump(root/'manifests/source_symbols.json',symbols)
    provenance_path=root/'manifests/provenance.json'
    if provenance_path.exists():
        provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
        files=provenance.get('files',[])
        retired_provenance=[r for r in files if not (root/r['destination']).exists()]
        provenance['files']=[r for r in files if (root/r['destination']).exists()]
        provenance['retired_documentation']=provenance.get('retired_documentation',[])+retired_provenance
        dump(provenance_path,provenance)
        allowlist_path=root/'manifests/allowlist.json'
        if allowlist_path.exists():
            allowlist=json.loads(allowlist_path.read_text())
            allowlist['source_files']=sorted(r['source'] for r in provenance['files'] if not r['source'].startswith('../'))
            allowlist['documentation_files']=sorted(text)
            dump(allowlist_path,allowlist)
    verification_path=root/'manifests/documentation_review.json'
    old=json.loads(verification_path.read_text()) if verification_path.exists() else {}
    dump(verification_path,dict(date='2026-10-26',removed=old.get('removed',[])+retired,
        runtime_code_changed=False,models_or_evidence_changed=False,original_research_documents_preserved=True,
        replacements=sorted(text),third_party_notices_preserved=True))

def validate(root):
    root=Path(root).resolve();broken=[];external_references=[]
    for p in root.rglob('*.md'):
        if '.git' in p.relative_to(root).parts:continue
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#','mailto:')):continue
            path=p.parent/target.split('#')[0]
            if not path.exists():
                row=dict(document=p.relative_to(root).as_posix(),target=target)
                if p==root/'instrumentation/nes-py/LICENSING.md':external_references.append(row)
                else:broken.append(row)
    assert not broken,broken
    dump(root/'manifests/link_audit.json',dict(broken_local_links=broken,upstream_documentation_references=external_references))
    files=[p for p in sorted(root.rglob('*')) if p.is_file() and '.git' not in p.relative_to(root).parts
        and p!=root/'manifests/inventory.json']
    dump(root/'manifests/inventory.json',[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    return dict(files=len(files)+1,markdown_files=len([p for p in files if p.suffix=='.md']),broken_links=len(broken))

if __name__=='__main__':
    root=Path(sys.argv[1]);apply(root);print(json.dumps(validate(root)))
