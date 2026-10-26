"""Opt-in experimental native observer owner; never used by production Level1Env."""
import ctypes as C
import hashlib
import json
from .paths import ROOT
from .recorder_native import RecorderNative, library

BUILD=ROOT/'instrumentation/kind18_observer_build'
EVENT_ACCEPTED_HIT=1
EVENT_DEFEAT=2


class Kind18Event(C.Structure):
    _fields_=[('native_frame',C.c_uint64),('target_slot',C.c_uint32),
              ('projectile_slot',C.c_uint32),('enemy_kind',C.c_uint32),
              ('handler_bank',C.c_uint32),('commit_pc',C.c_uint32),('lethal_pc',C.c_uint32),
              ('event_type',C.c_uint32),('counter_before',C.c_uint32),('counter_after',C.c_uint32)]


class Kind18ObserverNative(RecorderNative):
    """Base codec identity is preserved; experimental DLL provenance is separate.

    Events are observer state, not serialized game state. Loads/resets clear
    queues, pending context, overflow and local frame counter. Restoring inside
    a handler without observing its dispatch cannot emit an event.
    """
    def __init__(self,rom):
        metadata=json.loads((BUILD/'provenance.json').read_text())
        for path,expected in [(BUILD/'kind18_observer.dll',metadata['observer_dll_sha256']),
                (ROOT/'instrumentation/recorder_build/recorder_native.dll',metadata['base_dll_sha256'])]:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Observer/base DLL differs from recorded build provenance')
        api=library(BUILD/'kind18_observer.dll')
        for name,args,ret in [('rec_observer_abi',[],C.c_uint),
                ('rec_observer_counter_commit_valid',[C.c_uint,C.c_uint],C.c_uint),
                ('rec_observer_overflow',[C.c_void_p],C.c_uint),
                ('rec_observer_read',[C.c_void_p,C.POINTER(Kind18Event),C.c_int],C.c_int),
                ('rec_observer_capacity',[C.c_void_p,C.c_uint],C.c_int)]:
            f=getattr(api,name);f.argtypes=args;f.restype=ret
        if api.rec_observer_abi()!=2 or C.sizeof(Kind18Event)!=48:
            raise ValueError('Observer ABI mismatch')
        if api.rec_build().decode()!=metadata['base_build_id']:
            raise ValueError('Observer base codec identity mismatch')
        super().__init__(rom,api=api)
        self.abi_identity=dict(abi=2,record_size=C.sizeof(Kind18Event),
            observer_dll_sha256=metadata['observer_dll_sha256'],
            base_dll_sha256=metadata['base_dll_sha256'],base_build_id=metadata['base_build_id'],
            provenance_path=str(BUILD/'provenance.json'),observer_dll_path=str(BUILD/'kind18_observer.dll'))

    def read_events(self):
        events=[]
        while True:
            event=Kind18Event()
            count=self.api.rec_observer_read(self.pointer,C.byref(event),1)
            if count<0:raise RuntimeError('Observer read failed')
            if not count:return events
            events.append(dict((name,int(getattr(event,name))) for name,_ in event._fields_))

    def overflow(self):return bool(self.api.rec_observer_overflow(self.pointer))

    def set_observer_capacity(self,capacity):
        """0,1 or2 records; resets observer only. Zero tests explicit overflow."""
        if capacity not in (0,1,2):raise ValueError('Prototype capacity must be0,1 or2')
        if self.api.rec_observer_capacity(self.pointer,capacity):raise RuntimeError('Observer capacity failed')
