"""Audit or explicitly restore hash-pinned local inputs. Does not fetch a ROM."""
from _bootstrap import ROOT
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['audit','restore'])
    p.add_argument('--source',type=Path,help='Original research checkout; restore only')
    p.add_argument('--destination',help='Restore only this exact manifest destination; repeat for other inputs')
    a=p.parse_args()
    rows=json.loads((ROOT/'manifests/artifacts.json').read_text())
    if a.action=='restore':
        if a.source is None or not a.destination:p.error('restore requires --source and --destination; review access conditions first')
        rows=[r for r in rows if r['destination']==a.destination and not r['included']]
        if len(rows)!=1:p.error('destination must identify one excluded artifact in the manifest')
        row=rows[0];source=a.source.resolve()/row['source'];dest=(ROOT/row['destination']).resolve()
        if ROOT.resolve() not in dest.parents:raise ValueError('Destination escapes package')
        if dest.exists():raise FileExistsError(dest)
        if sha(source)!=row['sha256']:raise ValueError('Source hash mismatch')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    statuses=[dict(destination=r['destination'],status='missing' if not (ROOT/r['destination']).is_file()
        else 'verified' if sha(ROOT/r['destination'])==r['sha256'] else 'HASH_MISMATCH') for r in rows]
    print(json.dumps(statuses,indent=2))
    rom=ROOT/'roms/prince_valiant.nes'
    print(json.dumps(dict(rom_present=rom.is_file(),rom_identity_verified=rom.is_file() and sha(rom)=='96ad0a41a34fccc98bb7255890b131f77d4131cc69594c77e3ff33cd3a8fe6a5')))

if __name__=='__main__':main()
