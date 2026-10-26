"""Build vanilla nes-py with versioned, pointer-free MMC1 disk snapshots.

Only copies in recorder_build are modified. No navigation hooks are linked.
MSVC x64 and the checked-in nes-py sources are required.
"""
from pathlib import Path
import hashlib
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / 'recorder_build'
SOURCE = BUILD / 'nes_emu'


def main():
    BUILD.mkdir(exist_ok=True)
    shutil.copytree(ROOT / 'nes-py/nes_emu', SOURCE, dirs_exist_ok=True)
    for filename, cls in [('emulator.hpp', 'Emulator'), ('cpu.hpp', 'CPU'),
                          ('picture_bus.hpp', 'PictureBus'), ('controller.hpp', 'Controller'),
                          ('mapper.hpp', 'Mapper'), ('mappers/mapper_SxROM.hpp', 'MapperSxROM')]:
        p = SOURCE / 'include/nes_emu' / filename
        s = p.read_text().replace('class ' + cls + ' {', 'class ' + cls + ' {\n    friend struct RecorderState;')
        if cls == 'MapperSxROM':
            s = s.replace('class MapperSxROM : public Mapper {',
                          'class MapperSxROM : public Mapper {\n    friend struct RecorderState;')
        p.write_text(s)
    # Serialize each PPU field individually; never write callbacks or object pointers.
    ppu = (SOURCE / 'include/nes_emu/ppu.hpp').read_text().split('struct Snapshot {')[1].split('};')[0]
    names = re.findall(r'\b(\w+)\s*;', ppu)
    api = (ROOT / 'recorder_api.cpp').read_text().replace('/*PPU_FIELDS*/',
            '\n'.join('        io(p.' + n + ');' for n in names))
    fingerprint = hashlib.sha256()
    for p in sorted(SOURCE.rglob('*')):
        if p.is_file():
            fingerprint.update(p.relative_to(SOURCE).as_posix().encode())
            fingerprint.update(p.read_bytes())
    fingerprint.update(api.encode())
    build_id = fingerprint.hexdigest()
    api = api.replace('RECORDER_BUILD_ID', build_id)
    (BUILD / 'recorder_api.cpp').write_text(api)
    vswhere = Path(r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe')
    vs = Path(subprocess.check_output([str(vswhere), '-latest', '-products', '*',
                                      '-property', 'installationPath'], text=True).strip())
    args = ['cl', '/nologo', '/O2', '/EHsc', '/std:c++14', '/LD', '/I' + str(ROOT),
            '/I' + str(SOURCE / 'include'), str(BUILD / 'recorder_api.cpp'),
            *map(str, sorted((SOURCE / 'src/nes_emu').rglob('*.cpp'))),
            '/Fe:' + str(BUILD / 'recorder_native.dll')]
    batch = BUILD / 'compile.cmd'
    batch.write_text('@echo off\ncall "' + str(vs / 'VC/Auxiliary/Build/vcvars64.bat') +
                     '" >nul\n' + subprocess.list2cmdline(args) + '\n')
    subprocess.run(['cmd', '/c', str(batch)], cwd=BUILD, check=True)
    (BUILD / 'build_id.txt').write_text(build_id)


if __name__ == '__main__':
    main()
