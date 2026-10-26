"""Fresh, versioned navigation task. No legacy policy weights or scene hints.

Native snapshots stay with their owning emulator. Portable starts replay real
actions with this intervention version, and verify the resulting RAM digest.
"""
import copy, hashlib, json
from collections import Counter
from pathlib import Path
import gymnasium as gym
import numpy as np
from .navigation_native import NavigationNative as NativeEmulator
from .paths import ROOT
from .game_state import ROM_SHA256, word, signed_byte
from .scene_state import ForestTerrain
from .training.interface import load_catalog
from .training.temporal import FrameHistory
from .training.schema import FEATURE_KEYS

VERSION='navigation-v2'
ROWS=(0x9F81,0xABB1,0xBA40)
WIDTHS=(1792,1792,1280)
WEIGHTS=dict(new_progress=1.,checkpoint_reached=20.,level_completed=100.,
             navigation_damage=-20.,fall=-25.,lives_lost=-40.,elapsed_seconds=-.1,game_over=-40.)

def map_section(r):
    row=word(r,0x50E)
    return ROWS.index(row) if int(r[0x4D])==2 and row in ROWS else None

def supported(r):
    # Mode7 includes the now executed final-marker sequence as well as encounters.
    return map_section(r) is not None and int(r[0x33]) in (7,8) and int(r[0x2F4])==0

def live(r):
    s=map_section(r)
    return supported(r) and int(r[0x60])==0 and word(r,0x56)<WIDTHS[s] and word(r,0x58)<65535

def completed(old,r):
    # Actual bank7 C1C5/C1C7 -> C04F STA$33, after native end marker.
    return (map_section(old)==2 and int(old[0x33])==7 and int(r[0x33])==4
            and int(old[0x204])==1 and int(old[0x98])==22 and int(r[0x60])==0)

class NavigationInterventions:
    REMOVE={2,4,16,17,18,19,20,21}
    def __init__(self):
        self.prg=(ROOT/'roms/prince_valiant.nes').read_bytes()[16:16+131072]
        self.edits=Counter();self.guard=None;self.resources=None
    def before(self,r):
        self.guard=None;self.resources=(int(r[0xF7]),r[0xF8:0xFB].copy())
        if not supported(r):return
        # Protect only while verified enemy routines can run, never the marker.
        hostile=any(int(k) in self.REMOVE|{14,22} for k in r[0xC1:0xCB])
        # Protection is now scoped to CPU $D540 LDA$F3 inside the damage
        # subroutine, by the isolated native backend. Never hold F3 for a frame.
        self.clean(r)
    def clean(self,r):
        if not supported(r):return
        pointer=word(r,0x80)
        if 0x8000<=pointer<0xC000:
            for record in range(25):
                addr=pointer+10*record
                if addr+9>=0xC000:break
                values=self.prg[2*16384+addr-0x8000:2*16384+addr-0x8000+10]
                if not int(values[0]):break
                if int(values[0]) in self.REMOVE:
                    r[0x2CE+record]=128
        for slot in range(10):
            kind=int(r[0xC1+slot])
            if kind in self.REMOVE or kind==22:
                # Keep type23: its native parent-resolution path publishes201.
                r[0xC1+slot]=0;self.edits['removed_'+str(kind)]+=1
            elif kind==14 and int(r[0x99+slot])==1:
                # Exact executed zero-counter branch 8D27..8D34 ->8869.
                r[0xC1+slot]=13;r[0x99+slot]=0;r[0xB7+slot]=24;r[0x204]=1
                self.edits['mandatory_204']+=1
            elif kind==3 and map_section(r)==2 and int(r[0xAD+slot])&31 in (2,3):
                # Let the native phase2 completion publish gate202/203.
                if int(r[0x99+slot])<2:
                    r[0x99+slot]=2;r[0xB7+slot]=1;self.edits['mandatory_gate']+=1
    def after(self,r):
        if self.guard is not None and int(r[0xF3]) in (1,2):r[0xF3]=max(0,self.guard-1)
        self.clean(r)
        old_lives,old_score=self.resources
        # Elimination never supplies score or extra retries. Negative life changes
        # and native pit/death machinery are untouched. End bonus is not a reward.
        r[0xF8:0xFB]=old_score
        if signed_byte(r[0xF7])>signed_byte(old_lives):r[0xF7]=old_lives

class NavigationHistory:
    def __init__(self):self.frontiers=[-1]*3;self.visited=set();self.objectives=set();self.completed=False
    def seed(self,r):
        if live(r):
            s=map_section(r);self.frontiers[s]=max(self.frontiers[s],word(r,0x56));self.visited.add(s)
        for a in (0x201,0x202,0x203,0x204):
            if supported(r) and int(r[a])==1:self.objectives.add(a)
    def observable(self):
        return np.asarray([max(0,x)/1792 for x in self.frontiers]+[int(s in self.visited) for s in range(3)],np.float32)
    def transition(self,a,b):
        events=dict(new_progress=0.,checkpoint_reached=0.,level_completed=0.,navigation_damage=None,
                    fall=int(a[0x60]==0 and b[0x60]==1),lives_lost=max(0,signed_byte(a[0xF7])-signed_byte(b[0xF7])),elapsed_seconds=1/60,
                    game_over=int(b[0x60]==255 and signed_byte(b[0xF7])<0 and a[0x60]!=255))
        if live(a) and live(b) and map_section(a)==map_section(b):
            events['navigation_damage']=max(0,int(a[0xFC])-int(b[0xFC]))/255
        if live(b):
            s=map_section(b);x=word(b,0x56)
            if self.frontiers[s]>=0:events['new_progress']=max(0,x-self.frontiers[s])/16
            if s not in self.visited and s>0:events['checkpoint_reached']=1.
        if completed(a,b) and not self.completed:events['level_completed']=1.;self.completed=True
        self.seed(b)
        return events

class NavigationEncoder:
    """Fixed scales; raw categorical IDs embedded, absent slots explicitly masked."""
    def __init__(self,groups=None,observation_version='legacy_v3'):
        self.groups=set(groups or ('player','camera','objects','projectiles','terrain','triggers','controller','resources','reward_history'))
        self.observation_version=observation_version
        if observation_version not in ('legacy_v3','rea_expanded_v1','rea_expanded_v2'):
            raise ValueError(f'Unsupported observation schema: {observation_version}')
        self.terrain=ForestTerrain(ROOT/'roms/prince_valiant.nes',supported)
        self.rope_geometry=None
        if observation_version in ('rea_expanded_v1','rea_expanded_v2'):
            from .rope_geometry import RopeGeometry
            self.rope_geometry=RopeGeometry(ROOT/'roms/prince_valiant.nes')
        self.resident_prg_bank=(ROOT/'roms/prince_valiant.nes').read_bytes()[16+2*16384:16+3*16384] if observation_version=='rea_expanded_v2' else None
    def __call__(self,r,previous,history):
        s=map_section(r);valid=live(r);x,y=(word(r,0x56),word(r,0x58)) if valid else (0,0)
        continuity=previous is not None and live(previous) and valid and map_section(previous)==s and previous[0xF7]==r[0xF7]
        dx=x-word(previous,0x56) if continuity else 0;dy=y-word(previous,0x58) if continuity else 0
        continuity=continuity and abs(dx)<=16 and abs(dy)<=32
        scalars=[x/1792,y/192,float(valid),dx/8 if continuity else 0,dy/16 if continuity else 0,float(continuity),signed_byte(r[0x66])/16,int(r[0x67])/8]
        if 'player' not in self.groups:scalars=[0.]*len(scalars)
        scalars += [word(r,0x6E)/1792] if 'camera' in self.groups else [0.]
        scalars += [int(r[0xFC])/255,max(0,signed_byte(r[0xF7]))/8,int(r[0x60])==0,int(r[0x60])==1] if 'resources' in self.groups else [0.]*4
        scalars += history.observable().tolist() if 'reward_history' in self.groups else [0.]*6
        cat=[int(r[a]) for a in (0x65,0x5F,0x5E,0x33,0x2F4,0x4D,0x529)]
        cat += [int(r[0xD]),int(r[0xF])] if 'controller' in self.groups else [0,0]
        objects=[];codes=[]
        for i in range(10):
            active=bool(r[0xC1+i]) and 'objects' in self.groups
            ex,ey=word(r,0x28C+2*i),word(r,0x2A0+2*i)
            same=continuity and active and previous[0xC1+i]==r[0xC1+i] and previous[0xE9+i]==r[0xE9+i]
            vx=ex-word(previous,0x28C+2*i) if same else 0;vy=ey-word(previous,0x2A0+2*i) if same else 0
            same=same and abs(vx)<=16 and abs(vy)<=32
            objects.append([(ex-x)/256,(ey-y)/192,(ex-word(r,0x6E))/256,float(active),bool(r[0xCB+i]&4),bool(r[0xCB+i]&16),int(r[0x2BA+2*i])/255,int(r[0x2BB+2*i])/255,vx/8 if same else 0,vy/16 if same else 0,float(same)])
            codes.append([int(r[a+i]) if active else 0 for a in (0xC1,0xCB,0x99,0xD5,0xDF,0xA3,0xAD,0xB7,0xE9)])
            if not active:objects[-1]=[0.]*11
        shots=[];shot_codes=[]
        for i in range(3):
            active=bool(r[0x509+2*i]) and 'projectiles' in self.groups
            shots.append([(word(r,0x502+2*i)-x)/256,(int(r[0x2B4+2*i])-y)/192,float(active),int(r[0x509+2*i]&127)==1] if active else [0.]*4)
            shot_codes.append([int(r[a+2*i]) if active else 0 for a in (0x2B5,0x509,0x508)])
        terrain=self.terrain.observe(r,player_centered=True);mask=terrain['valid'].astype(np.float32)
        if 'terrain' not in self.groups:mask*=0
        triggers=self.terrain.triggers(r);rect=triggers['rectangles'].astype(np.float32);tv=triggers['valid'].astype(np.float32)
        if 'triggers' not in self.groups:tv*=0
        rect[:,0]=(rect[:,0]-x)/256;rect[:,1]=(rect[:,1]-y)/192;rect[:,2:]/=256;rect*=tv[:,None]
        result=dict(scalars=np.asarray(scalars,np.float32),categorical=np.asarray(cat,np.int64),entities=np.asarray(objects,np.float32),entity_codes=np.asarray(codes,np.int64),
            projectiles=np.asarray(shots,np.float32),projectile_codes=np.asarray(shot_codes,np.int64),tiles=np.where(mask,terrain['tile_ids'],256).astype(np.int64),
            classes=np.where(mask,terrain['collision_classes'],256).astype(np.int64),terrain_valid=mask,triggers=np.column_stack((rect,tv)),
            trigger_codes=np.where(tv[:,None],np.clip(triggers['action_arguments_raw'],0,255),0).astype(np.int64))
        if self.observation_version in ('rea_expanded_v1','rea_expanded_v2'):
            # Cell-aligned overlay of observed actor anchors and raw phases.
            # It does not claim that the anchor is a collision footprint or
            # that every visible phase is supporting/collidable.
            dynamic=np.zeros((12,17,11),np.float32)
            dynamic_scope=bool(valid and 'objects' in self.groups and 'terrain' in self.groups)
            if dynamic_scope:dynamic[:,:,0]=1.
            if dynamic_scope:
                ox,oy=map(int,terrain['origin'])
                for i in range(10):
                    kind=int(r[0xC1+i]);phase=int(r[0x99+i])
                    if kind not in (11,12):continue
                    ex,ey=word(r,0x28C+2*i),word(r,0x2A0+2*i)
                    row=(ey-oy)//16;col=(ex-ox)//16
                    if not (0<=row<12 and 0<=col<17):continue
                    if kind==11:
                        dynamic[row,col,1]=1.
                        if 129<=phase<=132:dynamic[row,col,2+phase-129]=1.
                    else:
                        dynamic[row,col,6]=1.
                        if 0<=phase<=3:dynamic[row,col,7+phase]=1.
            rope_grid=np.zeros((12,17,9),np.float32)
            if dynamic_scope:rope_grid[:,:,0]=1.
            if self.rope_geometry is not None and dynamic_scope:
                decoded=self.rope_geometry.decode(r);ox,oy=map(int,terrain['origin'])
                if decoded['scope_valid']:
                    for rope in decoded['ropes']:
                        if not rope['geometry_valid']:continue
                        for point in rope['attachment_points']:
                            row=(int(point['y'])-oy)//16;col=(int(point['world_x'])-ox)//16
                            segment=int(point['segment'])
                            if 0<=row<12 and 0<=col<17 and 0<=segment<8:
                                rope_grid[row,col,1+segment]=1.
            result['dynamic_actors']=dynamic
            result['rope_points']=rope_grid
        if self.observation_version=='rea_expanded_v2':
            # The executed bank-7 resident scanner reads a 10-byte record at
            # $80/$81 + 10*n and consults $02CE+n. The existing Level 1
            # intervention backend uses the selected bank-2 record table;
            # expose the raw bytes and status without assigning meanings to
            # undocumented record fields. Stop at the verified zero terminator.
            resident_codes=np.zeros((25,11),np.int64)
            resident_valid=np.zeros((25,1),np.float32)
            resident_scope=bool(valid and 'objects' in self.groups)
            pointer=word(r,0x80)
            if resident_scope and 0x8000<=pointer<0xC000:
                for n in range(25):
                    address=pointer+10*n
                    if address+9>=0xC000:break
                    start=address-0x8000
                    record=self.resident_prg_bank[start:start+10]
                    if len(record)!=10 or record[0]==0:break
                    resident_valid[n,0]=1.
                    resident_codes[n,0]=int(r[0x2CE+n])
                    resident_codes[n,1:]=np.frombuffer(record,dtype=np.uint8).astype(np.int64)
            result['resident_codes']=resident_codes
            result['resident_valid']=resident_valid
        return result

    def append_rea_structured_features(self,r,observation):
        """Append versioned REA fields after all preserved Level1 features."""
        if self.observation_version not in ('rea_expanded_v1','rea_expanded_v2'):return observation
        phase_scope=bool(supported(r));geometry_scope=bool(phase_scope and live(r))
        slot=int(r[0x2F2])
        rope_selected=(geometry_scope and int(r[0x5F])==23 and slot<10
                       and int(r[0xC1+slot])==6 and 39<=int(r[0xDF+slot])<=46
                       and word(r,0x2A0+2*slot)<256 and 0<=int(r[0x2F3])<8)
        observation['categorical']=np.r_[observation['categorical'],
            int(r[0x14]) if phase_scope else 0,
            int(r[0x63]) if geometry_scope else 0,
            slot if rope_selected else 0,
            int(r[0x2F3]) if rope_selected else 0].astype(np.int64)
        observation['scalars']=np.r_[observation['scalars'],
            int(r[0x68])/256. if geometry_scope else 0.,
            int(r[0x69])/256. if geometry_scope else 0.,
            float(geometry_scope),float(phase_scope),float(rope_selected)].astype(np.float32)
        if self.observation_version=='rea_expanded_v2':
            # Keep historical reward-credit bits (already present in the
            # caller's scalar vector) distinct from current gate RAM.
            gate_scope=bool(supported(r))
            kinds=np.asarray(r[0xC1:0xCB],dtype=np.uint8)
            target_scope=bool(gate_scope and np.any(kinds==3))
            counter_scope=bool(gate_scope and np.any(kinds==14))
            scripted_scope=bool(gate_scope and np.any(kinds==4))
            camera=int(r[0x6E])
            target_x=word(r,0x400)
            target_y=int(r[0x402])
            trigger_x=word(r,0x405)
            spawn_offset=word(r,0x407)
            spawn_world=(camera+spawn_offset)&0xFFFF
            observation['categorical']=np.r_[observation['categorical'],
                int(r[0x6B]) if gate_scope else 0,
                int(r[0x404]) if scripted_scope else 0,
                int(r[0x40A]) if scripted_scope else 0].astype(np.int64)
            observation['scalars']=np.r_[observation['scalars'],
                float(gate_scope),
                *([float(int(r[a])!=0) for a in (0x201,0x202,0x203,0x204)] if gate_scope else [0.]*4),
                (target_x-word(r,0x56))/256. if target_scope else 0.,
                (target_y-word(r,0x58))/192. if target_scope else 0.,
                float(target_scope),
                int(r[0x403])/255. if counter_scope else 0.,float(counter_scope),
                (trigger_x-camera)/256. if scripted_scope else 0.,
                (spawn_world-word(r,0x56))/256. if scripted_scope else 0.,
                (int(r[0x409])-word(r,0x58))/192. if scripted_scope else 0.,
                float(scripted_scope)].astype(np.float32)
        return observation

class NavigationEnv:
    def __init__(self,source='gameplay',horizon=12000,frame_skip=1,offsets=(-10,-5,0),groups=None,weights=None):
        self.encoder=NavigationEncoder(groups);self.catalog=load_catalog();self.allowed=np.ones(len(self.catalog),bool);self.allowed[10]=False
        self.horizon=horizon;self.frame_skip=frame_skip;self.weights=weights or WEIGHTS.copy();self.history=FrameHistory(offsets);self.source=source
        rom=ROOT/'roms/prince_valiant.nes'
        if hashlib.sha256(rom.read_bytes()).hexdigest()!=ROM_SHA256:raise ValueError('ROM mismatch')
        self.native=NativeEmulator(str(rom));self.native.reset();self.interventions=NavigationInterventions();self.tracker=NavigationHistory()
        p=Path(source);p=p if p.is_absolute() else ROOT/p if p.suffix else ROOT/'savestates'/f'{source}.json'
        save=json.loads(p.read_text());self.source_save=save
        base=save
        if save.get('navigation_version'):
            if save['navigation_version']!=VERSION:raise ValueError('Intervention version mismatch')
            b=Path(save['base_source']);b=b if b.is_absolute() else ROOT/b if b.suffix else ROOT/'savestates'/f'{b}.json'
            base=json.loads(b.read_text())
        for mask,count in base['actions']:
            for _ in range(count):
                self.native.frame_advance(mask);self.tracker.seed(self.native.ram_buffer())
        if hashlib.sha256(self.native.ram_buffer().tobytes()).hexdigest()!=base['fingerprint']['ram']:raise ValueError('Authentic source mismatch')
        if save.get('navigation_version'):
            self.interventions.clean(self.native.ram_buffer())
            self.native.damage_guard=True
            for mask,count in save['actions']:
                for _ in range(count):
                    self.interventions.before(self.native.ram_buffer());self.native.frame_advance(mask);self.interventions.after(self.native.ram_buffer());self.tracker.seed(self.native.ram_buffer())
            if hashlib.sha256(self.native.ram_buffer().tobytes()).hexdigest()!=save['fingerprint']['ram']:raise ValueError('Edited replay mismatch')
        self.tracker.seed(self.native.ram_buffer());self.initial_snapshot=self.native.dump_state();self.initial_tracker=copy.deepcopy(self.tracker)
        self.reset()
    def reset(self,*,seed=None,options=None):
        self.native.load_state(self.initial_snapshot);self.tracker=copy.deepcopy(self.initial_tracker);self.interventions=NavigationInterventions()
        self.elapsed=0;self.start_elapsed=0;self.actions=[];self.previous=None;self.done=False;self.sections_this_run=set();self.total_components=Counter();self.falls=0
        self.interventions.clean(self.native.ram_buffer());self.ram=self.native.ram_buffer().copy()
        self.native.damage_guard=True
        self.tracker.seed(self.ram);self.history.reset(self.encoder(self.ram,None,self.tracker))
        return self.history.window(),self.info()
    def info(self):
        return dict(source=self.source,section=map_section(self.ram),player_x=word(self.ram,0x56),player_y=word(self.ram,0x58),health=int(self.ram[0xFC]),elapsed_frames=self.elapsed,policy_elapsed_frames=self.elapsed-self.start_elapsed,reset_prelude_frames=self.start_elapsed,
            frontiers=self.tracker.frontiers.copy(),falls=self.falls,completion=self.tracker.completed,full_level_completion=self.tracker.completed and self.sections_this_run=={0,1,2},
            reward_totals=dict(self.total_components),interventions=dict(self.interventions.edits))
    def step(self,action):
        if self.done:raise RuntimeError('Reset ended episode')
        if not self.allowed[action]:raise ValueError('Masked action')
        return self.advance(self.catalog[int(action)],self.frame_skip)
    def advance(self,mask,count=1):
        total=0.;components=Counter();events=Counter();duration=0;reason=None
        for _ in range(count):
            old=self.ram.copy();self.interventions.before(self.native.ram_buffer());self.native.frame_advance(int(mask));self.interventions.after(self.native.ram_buffer())
            self.interventions.edits['instruction_damage_blocks']+=self.native.damage_blocks;self.interventions.edits['instruction_spawn_filters']+=self.native.spawn_filters
            self.ram=self.native.ram_buffer().copy();self.elapsed+=1;duration+=1
            if live(self.ram):self.sections_this_run.add(map_section(self.ram))
            e=self.tracker.transition(old,self.ram);self.falls+=e['fall']
            for key,value in e.items():
                if value is not None:events[key]+=value;components[key]+=value*self.weights[key]
            self.history.push(self.elapsed,self.encoder(self.ram,old,self.tracker));self.previous=old
            if self.actions and self.actions[-1][0]==mask:self.actions[-1][1]+=1
            else:self.actions.append([int(mask),1])
            if self.tracker.completed:reason='level_completed'
            elif self.ram[0x60]==255 and signed_byte(self.ram[0xF7])<0:reason='game_over'
            elif map_section(self.ram) is None or int(self.ram[0x33]) not in (7,8):reason='unsupported_mode'
            elif self.elapsed>=self.horizon:reason='time_limit'
            if reason:break
        total=sum(components.values());self.total_components.update(components);self.done=reason is not None
        info=self.info();info.update(transition_frames=duration,events=dict(events),reward_components=dict(components),end_reason=reason,bootstrap_valid=reason not in ('game_over','level_completed','unsupported_mode'))
        return self.history.window(),total,reason in ('game_over','level_completed'),reason in ('time_limit','unsupported_mode'),info
    def snapshot(self):
        return self.native.dump_state(),copy.deepcopy({k:getattr(self,k) for k in ('tracker','interventions','history','elapsed','start_elapsed','actions','previous','ram','done','sections_this_run','total_components','falls')})
    def restore(self,state):
        native,fields=state;self.native.load_state(native)
        for k,v in fields.items():setattr(self,k,copy.deepcopy(v))
        return self.history.window(),self.info()
    def close(self):self.native.close()

    def save_start(self,path):
        prior=self.source_save.get('actions',[]) if self.source_save.get('navigation_version') else []
        base=self.source_save.get('base_source',self.source)
        payload=dict(navigation_version=VERSION,base_source=base,actions=prior+self.actions,
                     fingerprint=dict(ram=hashlib.sha256(self.ram.tobytes()).hexdigest()),full_start=False,
                     provenance='Executed modified-environment controls; not an original-game observation copy',state=self.info())
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,indent=2))

def navigation_child(pipe,configuration,offsets):
    import os,traceback,sys
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    cache={};env=None;stashed=None
    pipe.send(dict(ready=True,pid=os.getpid(),torch_imported='torch' in sys.modules))
    try:
        while True:
            command,payload=pipe.recv()
            if command=='close':break
            try:
                if command=='reset':
                    source=payload['source']
                    if source not in cache:cache[source]=NavigationEnv(source,offsets=offsets,groups=configuration.get('feature_groups'),weights=configuration.get('weights'))
                    env=cache[source];env.horizon=payload.get('horizon',12000);env.frame_skip=configuration.get('frame_skip',1)
                    obs,info=env.reset()
                    for m,c in payload.get('prelude',[]):
                        obs,_,t,u,info=env.advance(m,c)
                        if t or u:raise ValueError('Invalid reset variation')
                    env.start_elapsed=env.elapsed;env.horizon+=env.start_elapsed;env.total_components.clear()
                    pipe.send(dict(obs=obs,info=info,mask=env.allowed.copy()))
                elif command=='step':
                    o,r,t,u,i=env.step(int(payload));pipe.send(dict(obs=o,reward=r,terminated=t,truncated=u,info=i))
                elif command=='stash':stashed=(env,env.snapshot());pipe.send(dict(stashed=True))
                elif command=='restore':env,state=stashed;obs,info=env.restore(state);stashed=None;pipe.send(dict(obs=obs,info=info,mask=env.allowed.copy()))
                else:raise ValueError('Unknown navigation command')
            except Exception:pipe.send(dict(error=traceback.format_exc()))
    except (EOFError,BrokenPipeError):pass
    finally:
        for env in cache.values():env.close()
        pipe.close()
