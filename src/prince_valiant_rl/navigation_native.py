"""Owner-local ctypes adapter for the isolated navigation damage guard DLL."""
import ctypes as C
import numpy as np
from .paths import ROOT

_dll=None
def library():
    global _dll
    if _dll is None:
        _dll=C.CDLL(str(ROOT/'instrumentation/navigation_build/navigation_native.dll'))
        signatures={'nav_create':([C.c_char_p],C.c_void_p),'nav_destroy':([C.c_void_p],None),'nav_reset':([C.c_void_p],None),
            'nav_ram':([C.c_void_p],C.POINTER(C.c_uint8)),'nav_screen':([C.c_void_p],C.POINTER(C.c_uint32)),
            'nav_step':([C.c_void_p,C.c_int,C.c_int],C.c_int),'nav_save':([C.c_void_p],C.c_void_p),
            'nav_load':([C.c_void_p,C.c_void_p],None),'nav_free_save':([C.c_void_p],None),'nav_damage_blocks':([],C.c_uint64),'nav_spawn_filters':([],C.c_uint64)}
        for name,(args,ret) in signatures.items():f=getattr(_dll,name);f.argtypes=args;f.restype=ret
    return _dll

class Snapshot:
    def __init__(self,owner):self.owner=owner;self.pointer=library().nav_save(owner.pointer)
    def __del__(self):
        if getattr(self,'pointer',None):library().nav_free_save(self.pointer);self.pointer=None

class NavigationNative:
    def __init__(self,rom):
        self.pointer=library().nav_create(str(rom).encode());self.damage_guard=False
        if not self.pointer:raise RuntimeError('Native ROM load failed')
    def reset(self):library().nav_reset(self.pointer)
    def ram_buffer(self):return np.ctypeslib.as_array(library().nav_ram(self.pointer),shape=(2048,))
    def frame_advance(self,mask):
        if library().nav_step(self.pointer,int(mask),int(self.damage_guard)):raise RuntimeError('Native frame failed')
        self.damage_blocks=int(library().nav_damage_blocks());self.spawn_filters=int(library().nav_spawn_filters())
    def dump_state(self):return Snapshot(self)
    def load_state(self,snapshot):
        if snapshot.owner is not self:raise ValueError('Snapshot belongs to another native owner')
        library().nav_load(self.pointer,snapshot.pointer)
    def copy_screen_rgb(self):
        p=np.ctypeslib.as_array(library().nav_screen(self.pointer),shape=(240,256));return np.stack(((p>>16)&255,(p>>8)&255,p&255),axis=-1).astype(np.uint8)
    def close(self):
        if self.pointer:library().nav_destroy(self.pointer);self.pointer=None
