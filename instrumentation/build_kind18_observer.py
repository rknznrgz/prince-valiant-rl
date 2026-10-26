"""Build experimental observer DLL; preserve production recorder and save codec."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent


def main():
    base=ROOT/'recorder_build';out=ROOT/'kind18_observer_build';out.mkdir(exist_ok=True)
    original=(base/'recorder_native.dll').read_bytes()
    original_hash=hashlib.sha256(original).hexdigest()
    shutil.copytree(base/'nes_emu',out/'nes_emu',dirs_exist_ok=True)
    # TLS declaration must match in every sidecar TU. Original headers/sources
    # and their serializers remain untouched.
    hooks=(ROOT/'trace_hooks.hpp').read_text()
    assert hooks.count('extern bool debug_trace_enabled;')==1
    (out/'trace_hooks.hpp').write_text(hooks.replace('extern bool debug_trace_enabled;',
                                                   'extern thread_local bool debug_trace_enabled;'))
    api=(base/'recorder_api.cpp').read_text()
    stubs='''bool debug_trace_enabled=false;
void debug_instruction(CPU&,MainBus&){}
void debug_access(char,NES_Address,NES_Byte){}
void debug_rom_read(NES_Address,NES_Byte,const NES_Byte*){}'''
    assert api.count(stubs)==1
    # Observer header declares namespace NES itself; include before its block.
    api=api.replace(stubs,'').replace('namespace NES {', '#include "kind18_observer.hpp"\nnamespace NES {',1)
    anchor='    void transfer(Emulator& e) {'
    assert api.count(anchor)==1
    api=api.replace(anchor,'    static const NES_Byte* prg_base(Emulator& e){return e.cartridge.getROM().data();}\n'+anchor)
    cut=api.index('using NES::Emulator;')
    build_id=(base/'build_id.txt').read_text().strip()
    api=api[:cut]+(ROOT/'kind18_sidecar_api.inc').read_text().replace('BASE_RECORDER_BUILD_ID',build_id)
    (out/'recorder_api.cpp').write_text(api)
    vswhere=Path(r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe')
    vs=Path(subprocess.check_output([str(vswhere),'-latest','-products','*','-property','installationPath'],text=True).strip())
    args=['cl','/nologo','/O2','/EHsc','/std:c++14','/LD','/I'+str(out),'/I'+str(ROOT),
          '/I'+str(out/'nes_emu/include'),str(out/'recorder_api.cpp'),
          *map(str,sorted((out/'nes_emu/src/nes_emu').rglob('*.cpp'))),
          '/Fe:'+str(out/'kind18_observer.dll')]
    batch=out/'compile.cmd'
    batch.write_text('@echo off\ncall "'+str(vs/'VC/Auxiliary/Build/vcvars64.bat')+'" >nul\n'
                     +subprocess.list2cmdline(args)+'\n')
    subprocess.run(['cmd','/c',str(batch)],cwd=out,check=True)
    assert (base/'recorder_native.dll').read_bytes()==original
    (out/'provenance.json').write_text(json.dumps(dict(
        base_build_id=build_id,base_dll_sha256=original_hash,
        observer_dll_sha256=hashlib.sha256((out/'kind18_observer.dll').read_bytes()).hexdigest(),
        generated_api_sha256=hashlib.sha256(api.encode()).hexdigest(),observer_abi=2,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                       ('kind18_observer.hpp','kind18_sidecar_api.inc','build_kind18_observer.py')},
        compatibility='Base RecorderState serialization unchanged; rec_build denotes base codec identity only. Experimental combat event schema v2 uses 48-byte records and separate SHA256/ABI; no production replacement.'),indent=2))
    print(out/'kind18_observer.dll')


if __name__=='__main__':main()
