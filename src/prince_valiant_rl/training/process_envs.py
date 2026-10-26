"""Spawned independent CPU emulator owners; CUDA inference stays in the parent."""
import multiprocessing as mp
import os,time,traceback,sys
import numpy as np

def child_main(pipe,worker,offsets):
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    from .scene_pool import SequentialScenePool
    from .scenes import ControlledScene
    from .temporal import TemporalEnv
    from .interface import TrainingEnv
    from .starts import CachedStart
    from .events import section,live_position
    pool=SequentialScenePool(worker);env=None;custom={};normal={};stashed=None
    try:
        import sys
        pipe.send(dict(ready=True,pid=os.getpid(),torch_imported='torch' in sys.modules))
        while True:
            command,payload=pipe.recv()
            if command=='close':break
            try:
                if command=='reset':
                    prior_sources=set(pool.sources);prior_normal=set(normal)
                    scene=payload['scene'];configuration=payload.get('config')
                    if scene=='normal':
                        source=payload.get('source','gameplay')
                        if source not in normal:normal[source]=CachedStart(source,task='manager',horizon=payload.get('horizon',600))
                        cache=normal[source];base,obs,info=cache.reset();base.core.max_episode_frames=payload.get('horizon',600)
                        from .input_features import configure_encoder
                        configure_encoder(base,payload.get('input_latches',False));obs=base.encoder(base.ram,base._previous,base.tracker)
                        env=TemporalEnv(base,offsets);env.base=base;env.history.reset(obs);base.frame_observer=env.history.push
                        obs=env.history.window();mask=np.ones(12,bool)
                        if payload.get('mask_start'):mask[10]=False
                    else:
                        if configuration is not None:
                            import json
                            key=(scene,json.dumps(configuration,sort_keys=True))
                            if key not in custom:
                                base=ControlledScene(scene,config=configuration,allow_unvalidated=payload.get('allow_unvalidated',False));base.cache=pool.sources;custom[key]=base
                            base=custom[key]
                        else:base=pool.get(scene)
                        env=TemporalEnv(base,offsets);obs,info=env.reset(seed=payload['seed'],options={'source':payload['source']});mask=base.allowed.copy()
                    prelude=payload.get('prelude_actions',[])
                    if len(prelude)>240:raise ValueError('Bounded development prelude exceeds240 native decisions')
                    for action in prelude:
                        if not mask[int(action)]:raise ValueError('Masked prelude control')
                        obs,_,t,u,_=env.step(int(action))
                        if t or u:raise ValueError('Evaluation prelude crossed a boundary; condition is invalid')
                    import hashlib
                    ram=env.base.ram
                    info.update(initial_ram_sha256=hashlib.sha256(ram.tobytes()).hexdigest(),health=int(ram[0xFC]),
                        player_x=int(ram[0x56])+256*int(ram[0x57]),section=section(ram),position_valid=live_position(ram),pid=os.getpid(),evaluation_prelude_frames=len(prelude),initialization_replay_frames=
                        sum(pool.sources[s].info.get('checkpoint_replay_frames',0) for s in set(pool.sources)-prior_sources)+
                        sum(normal[s].info.get('checkpoint_replay_frames',0) for s in set(normal)-prior_normal))
                    pipe.send(dict(obs=obs,info=info,mask=mask))
                elif command=='stash':
                    import copy
                    if stashed is not None:raise RuntimeError('A live rollout is already stashed')
                    base=env.base;controller=env.env
                    fields=('tracker','_previous','_last','ram','decisions','encoder','observation_space')
                    core_fields=('frontier','elapsed_frames','_done','_ram','max_episode_frames')
                    scene_fields=('reached','dead','intervention_failure','hits','defeats','terminal','frame_log','edits','_pre')
                    stashed=dict(env=env,base=base,controller=controller,mask=mask.copy(),history=copy.deepcopy(env.history),snapshot=base.core.native.dump_state(),
                        state=copy.deepcopy({k:getattr(base,k) for k in fields}),
                        core=copy.deepcopy({k:getattr(base.core,k) for k in core_fields}),
                        scene=copy.deepcopy({k:getattr(controller,k) for k in scene_fields if hasattr(controller,k)}))
                    pipe.send(dict(stashed=True))
                elif command=='restore':
                    if stashed is None:raise RuntimeError('No stashed rollout')
                    s=stashed;env=s['env'];base=s['base'];controller=s['controller']
                    base.core.native.load_state(s['snapshot'])
                    for k,v in s['state'].items():setattr(base,k,v)
                    for k,v in s['core'].items():setattr(base.core,k,v)
                    for k,v in s['scene'].items():setattr(controller,k,v)
                    if isinstance(controller,ControlledScene):
                        controller.env=base;base.core.before_native_frame=controller._before;base.core.after_native_frame=controller._after
                    else:base.core.before_native_frame=None;base.core.after_native_frame=None
                    env.history=s['history'];base.frame_observer=env.history.push;env.base=base;mask=s['mask'];stashed=None
                    ram=base.ram
                    pipe.send(dict(obs=env.history.window(),mask=mask.copy(),
                        info=dict(ram_sha256=hashlib.sha256(ram.tobytes()).hexdigest(),elapsed_frames=base.core.elapsed_frames)))
                elif command=='step':
                    start=time.perf_counter();obs,reward,t,u,info=env.step(int(payload));ram=env.base.ram
                    compact={k:info.get(k) for k in ('transition_frames','elapsed_frames','bootstrap_valid','events','reward_components','scene_success','end_reason')}
                    compact.update(health=int(ram[0xFC]),player_x=int(ram[0x56])+256*int(ram[0x57]),player_y=int(ram[0x58])+256*int(ram[0x59]),
                        camera_x=int(ram[0x6E])+256*int(ram[0x6F]),section=section(ram),position_valid=live_position(ram),raw_types=ram[0xC1:0xCB].tolist(),cpu_step_seconds=time.perf_counter()-start)
                    compact['ram_sha256']=hashlib.sha256(ram.tobytes()).hexdigest()
                    compact['torch_imported']='torch' in sys.modules
                    pipe.send(dict(obs=obs,reward=reward,terminated=t,truncated=u,info=compact))
                else:raise ValueError('Unknown CPU emulator command')
            except Exception:pipe.send(dict(error=traceback.format_exc()))
    except (EOFError,BrokenPipeError):pass
    finally:
        pool.close()
        for cache in normal.values():cache.close()
        pipe.close()

class ProcessEnvs:
    def __init__(self,count,worker,offsets=(-10,-5,0),child_target=None):
        if count<1:raise ValueError('Positive process count required')
        context=mp.get_context('spawn');self.parents=[];self.processes=[];self.pids=[];self.owner_metadata=[]
        # run.py uses runpy: Windows would otherwise reexecute the GPU training
        # tool's top-level imports in EVERY CPU owner. The stable dispatcher has
        # no neural imports and child_main is a normal module-level pickle target.
        from ..paths import ROOT
        main=sys.modules.get('__main__');previous_file=getattr(main,'__file__',None);previous_spec=getattr(main,'__spec__',None)
        try:
            if main is not None:main.__file__=str(ROOT/'run.py');main.__spec__=None
            for _ in range(count):
                parent,child=context.Pipe();process=context.Process(target=child_target or child_main,args=(child,worker,offsets),daemon=True)
                process.start();child.close();self.parents.append(parent);self.processes.append(process)
        except Exception:self.close();raise
        finally:
            if main is not None:main.__file__=previous_file;main.__spec__=previous_spec
        try:
            for i in range(count):
                ready=self.receive(i);self.pids.append(ready['pid']);self.owner_metadata.append(ready)
        except Exception:self.close();raise
    def send(self,i,command,payload):self.parents[i].send((command,payload))
    def receive(self,i):
        if not self.parents[i].poll(180):raise TimeoutError(f'Emulator process {i} did not respond')
        result=self.parents[i].recv()
        if 'error' in result:raise RuntimeError(f'Emulator process {i} failed:\n{result["error"]}')
        return result
    def reset_many(self,cases,indices=None):
        indices=list(range(len(cases))) if indices is None else list(indices)
        if len(indices)!=len(cases):raise ValueError('Reset assignments must align with process indices')
        for i,case in zip(indices,cases):self.send(i,'reset',case)
        return [self.receive(i) for i in indices]
    def step_many(self,actions,indices=None):
        indices=list(range(len(actions))) if indices is None else list(indices)
        if len(indices)!=len(actions):raise ValueError('Action assignments must align with process indices')
        for i,action in zip(indices,actions):self.send(i,'step',int(action))
        return [self.receive(i) for i in indices]
    def stash_many(self):
        for i in range(len(self.parents)):self.send(i,'stash',None)
        return [self.receive(i) for i in range(len(self.parents))]
    def restore_many(self):
        for i in range(len(self.parents)):self.send(i,'restore',None)
        return [self.receive(i) for i in range(len(self.parents))]
    def close(self):
        for parent in self.parents:
            try:parent.send(('close',None))
            except (EOFError,BrokenPipeError,OSError):pass
        for process in self.processes:
            process.join(5)
            if process.is_alive():process.terminate();process.join(5)
        for parent in self.parents:parent.close()

def grouped_advantages(rollout,gamma,lam):
    """GAE per independent stream; duration stays in original base-step units."""
    from .events import duration_gae
    advantage=np.empty(len(rollout),np.float32);returns=advantage.copy()
    for stream in sorted({r.get('env_id',0) for r in rollout}):
        ix=[i for i,r in enumerate(rollout) if r.get('env_id',0)==stream];rows=[rollout[i] for i in ix]
        a,v=duration_gae([r['reward'] for r in rows],[r['value'] for r in rows],[r['next_value'] for r in rows],
            [r['duration'] for r in rows],[r['bootstrap'] for r in rows],[r['continue'] for r in rows],gamma,lam)
        advantage[ix]=a;returns[ix]=v
    return advantage,returns

def condition_critic(model,rollout,device,batch=128):
    """Batch unchanged critic predictions AFTER fresh actor-only collection.

    Complete inputs are stored directly. A boundary's actual following window
    is retained; it is never replaced by the next episode's reset observation.
    No running normalization/dropout exists in the supported fixed-window model.
    """
    import torch
    from .temporal_models import window_tensors
    windows=[];lookup={};pairs=[]
    for row in rollout:
        pair=[]
        for key in ('obs','next_obs'):
            observation=row[key];identity=id(observation)
            if identity not in lookup:lookup[identity]=len(windows);windows.append(observation)
            pair.append(lookup[identity])
        pairs.append(pair)
    values=[]
    with torch.no_grad():
        for start in range(0,len(windows),batch):
            x=window_tensors({k:np.stack([w[k] for w in windows[start:start+batch]]) for k in model.window_keys},device,batch=True)
            values.extend(model.critic(x).squeeze(-1).cpu().numpy())
    for row,(current,following) in zip(rollout,pairs):
        row['value']=float(values[current]);row['next_value']=float(values[following]) if row['bootstrap'] else 0.
    return len(windows)

def resident_bytes(pid):
    if os.name!='nt':return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=wintypes.HANDLE
    handle=kernel.OpenProcess(0x410,False,pid)
    if not handle:return None
    try:
        value=Counters();value.cb=ctypes.sizeof(value)
        api=ctypes.WinDLL('psapi');api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        return int(value.WorkingSetSize) if api.GetProcessMemoryInfo(handle,ctypes.byref(value),value.cb) else None
    finally:
        kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle(handle)
