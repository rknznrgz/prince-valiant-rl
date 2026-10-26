"""Vanilla native emulator with ROM/build-bound full disk states (no pickle)."""
import ctypes as C
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from .paths import ROOT

_dll = None
MAGIC = b'PVSTATE1'
SUPPORTED_ROM = '96ad0a41a34fccc98bb7255890b131f77d4131cc69594c77e3ff33cd3a8fe6a5'


def library(path=None):
    global _dll
    if path is not None:
        return configure_library(C.CDLL(str(path)))
    if _dll is None:
        _dll = configure_library(C.CDLL(str(ROOT / 'instrumentation/recorder_build/recorder_native.dll')))
    return _dll


def configure_library(dll):
    signatures = {'rec_build': ([], C.c_char_p), 'rec_create': ([C.c_char_p], C.c_void_p),
            'rec_destroy': ([C.c_void_p], None), 'rec_reset': ([C.c_void_p], None),
            'rec_ram': ([C.c_void_p], C.POINTER(C.c_uint8)),
            'rec_screen': ([C.c_void_p], C.POINTER(C.c_uint32)),
            'rec_step': ([C.c_void_p, C.c_int], C.c_int),
            'rec_save': ([C.c_void_p, C.c_void_p, C.c_int], C.c_int),
            'rec_load': ([C.c_void_p, C.c_void_p, C.c_int], C.c_int)}
    for name, (args, ret) in signatures.items():
        f = getattr(dll, name)
        f.argtypes, f.restype = args, ret
    return dll


class RecorderNative:
    def __init__(self, rom, *, api=None):
        self.api=library() if api is None else api
        self.rom = Path(rom)
        self.rom_hash = hashlib.sha256(self.rom.read_bytes()).hexdigest()
        if self.rom_hash != SUPPORTED_ROM:
            raise ValueError('Recorder and event schema are verified only for the supplied Prince Valiant ROM')
        self.build = self.api.rec_build().decode()
        self.pointer = self.api.rec_create(str(self.rom).encode())
        if not self.pointer:
            raise RuntimeError('Native ROM load failed')
        self._buffer = C.create_string_buffer(1024 * 1024)
        self.api.rec_reset(self.pointer)

    def ram_buffer(self):
        return np.ctypeslib.as_array(self.api.rec_ram(self.pointer), shape=(2048,))

    def frame_advance(self, mask):
        if not 0 <= mask <= 255 or self.api.rec_step(self.pointer, mask):
            raise RuntimeError('Native frame failed')

    def copy_screen_rgb(self):
        p = np.ctypeslib.as_array(self.api.rec_screen(self.pointer), shape=(240, 256))
        return np.stack(((p >> 16) & 255, (p >> 8) & 255, p & 255), axis=-1).astype(np.uint8)

    def dump_state(self):
        """Fast uncompressed capture on emulator thread; compress on writer thread."""
        n = self.api.rec_save(self.pointer, self._buffer, len(self._buffer))
        if n < 0:
            raise RuntimeError('Complete save state failed (MMC1/CHR ROM required)')
        data = self._buffer.raw[:n]
        header = json.dumps(dict(schema=1, rom_sha256=self.rom_hash, build=self.build,
                                payload_sha256=hashlib.sha256(data).hexdigest())).encode()
        return MAGIC + struct.pack('<I', len(header)) + header + data

    def load_state(self, data):
        if data[:8] != MAGIC:
            raise ValueError('Not a recorder full state')
        n, = struct.unpack('<I', data[8:12])
        meta, payload = json.loads(data[12:12+n]), data[12+n:]
        if meta['rom_sha256'] != self.rom_hash or meta['build'] != self.build:
            raise ValueError('ROM or emulator build differs from checkpoint')
        if hashlib.sha256(payload).hexdigest() != meta['payload_sha256']:
            raise ValueError('State checksum mismatch')
        # Load into a fresh owner: a rejected state cannot corrupt current gameplay.
        candidate = self.api.rec_create(str(self.rom).encode())
        if not candidate:
            raise RuntimeError('Restore owner creation failed')
        if self.api.rec_load(candidate, payload, len(payload)):
            self.api.rec_destroy(candidate)
            raise ValueError('Native state rejected')
        self.api.rec_destroy(self.pointer)
        self.pointer = candidate

    def close(self):
        if self.pointer:
            self.api.rec_destroy(self.pointer)
            self.pointer = None
