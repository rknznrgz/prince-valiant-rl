"""Vanilla Level 1 single-policy environment and backward reset curriculum.

Demonstration controls run only inside reset reconstruction, never during step.
No navigation environment or intervention object is constructed.
"""
from collections import Counter, deque
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import zlib

import numpy as np

from .paths import ROOT
from .recorder_native import RecorderNative
from .navigation import NavigationEncoder, map_section, supported, live, completed
from .game_state import word, signed_byte, decode_display_score
from .training.temporal import FrameHistory
from .control_history import ControlHistory
from .kind18_events import Kind18Events, KEYS as KIND18_KEYS, COVERAGE as KIND18_COVERAGE

OUT=ROOT/'training/backward_level1_v1'
GATES=(0x201,0x202,0x203,0x204)
NAVIGATION_REWARD_VERSION='level1_progress_milestones_v2'
TIMEOUT_TERMINAL_VERSION='level1_section0_timeout_terminal_v1'
COMBAT_OBSERVER_DIAGNOSTIC_COVERAGE=dict(version='experimental_combat_observer_abi2_v1',complete=False,
    units=dict(accepted_hits='accepted remaining-hit-count reductions; not generic HP loss',
        defeats='verified defeat events; out-of-scope kind18 records are listed separately'),
    by_enemy_kind={
        '14':dict(accepted_hits=dict(available=True,
            scope='Level1 section2, phase1; verified full recorded slot0 encounter; dispatch-attributed commits'),
            defeats=dict(available=True,
            scope='Level1 section2 kind14 phase1; dispatch-attributed lethal branch and defeat writes')),
        '18':dict(accepted_hits=dict(available=False,
            reason='ABI2 observer does not implement kind18 accepted-hit records'),
            defeats=dict(available=True,
            scope='Level1 section0 kind18 phase0; verified flags5/7, packed counter low bits1->0, projectile slot2')),
        'other':dict(accepted_hits=dict(available=False),defeats=dict(available=False))},
    out_of_scope_kind18_defeats='retained as unverified diagnostics; excluded from verified counts and rewards',
    diagnostics_excluded_from_rewards_except_scoped_abi2_kind18_defeat=True,
    generic_enemy_damage_available=False,generic_enemy_kills_available=False)
ABI2_KIND18_DEFEAT_REWARD_COVERAGE=dict(version='experimental_combat_observer_abi2_v1',
    event='ABI2 event_type=2, enemy_kind=18',available_scope='supported Level1 section0; verified flags5/7 dispatches',
    record_constraints=dict(target_slots=[8,9],projectile_slot=2,handler_bank=2,
        commit_pc=0x87F6,lethal_pc=0x87FC,counter_low_bits='1->0'),
    outside_scope='unavailable; no scoped bonus',kind14_event_weight=0.0,
    generic_enemy_damage_available=False,generic_enemy_kills_available=False)
SCORE_GAIN_REWARD_COVERAGE=dict(version='display_score_bcd_f4_f6_v1',
    event='positive displayed-score delta',units='displayed score points',decoder='strict packed BCD RAM $F4-$F6',
    scope='Level1Env native-frame transitions with valid six-digit BCD endpoints',complete=False,
    unverified=['death/retry score semantics','score rollover','non-Level1 mode semantics'],
    observation_contract_preserved=True,generic_enemy_damage_available=False,generic_enemy_kills_available=False)


def section0_timeout_terminal(a,b):
    """Executed final-timeout -> continue-countdown path, verified in section0.

    Natural B-only witness74652 plus native/6502 counterfactuals are in
    training/level1_research_recovery_v1/timeout_boundary. A normal game timer
    does not count UP from0000 to0009; the continue sequence reuses these bytes.
    Do not generalize to unverified sections/modes or treat title as completion.
    """
    return (supported(a) and supported(b) and map_section(a)==map_section(b)==0
            and int(a[0x33])==int(b[0x33])==8 and int(a[0xF7])==int(b[0xF7])==0
            and int(a[0x60])==int(b[0x60])==0
            and not np.any(a[0x286:0x28A]) and b[0x286:0x28A].tolist()==[0,0,0,9]
            and int(a[0x28A])<=1 and int(b[0x28A]) in (59,60))


class TaskHistory:
    def __init__(self):
        self.frontiers=[-1]*3;self.visited=set();self.objectives=set();self.completed=False
        self.kind18=Kind18Events()

    def seed(self,r,*,combat=True):
        if combat:
            if not hasattr(self,'kind18'):self.kind18=Kind18Events()
            self.kind18.seed(r)
        if live(r):
            s=map_section(r);self.frontiers[s]=max(self.frontiers[s],word(r,0x56));self.visited.add(s)
        if supported(r):self.objectives.update(a for a in GATES if int(r[a])==1)

    def observable(self):
        return np.asarray([max(0,x)/1792 for x in self.frontiers]+[int(s in self.visited) for s in range(3)],np.float32)

    def transition(self,a,b):
        # Older serialized histories have no detector; restored RAM is only a
        # baseline. No prehistory is paid as new combat events.
        if not hasattr(self,'kind18'):
            self.kind18=Kind18Events();self.kind18.seed(a)
        valid=supported(a) and supported(b)
        e=dict(new_progress=0.,checkpoint_reached=0.,level_completed=0.,player_damage=None,
               death_started=0.,lives_lost=0.,elapsed_seconds=1/60,game_over=0.,objective_cleared=0.,
               enemy_kills=None,enemy_damage=None,score_gain=None)
        e.update(self.kind18.transition(a,b))
        if valid:
            e['death_started']=float(int(a[0x60])==0 and int(b[0x60])==1)
            e['lives_lost']=max(0,signed_byte(a[0xF7])-signed_byte(b[0xF7]))
            # Fatal health changes can be pit deaths. Charge death/life costs,
            # never label an ambiguous fatal HP write as an enemy damage event.
            if live(a) and live(b) and a[0xF7]==b[0xF7] and map_section(a)==map_section(b):
                e['player_damage']=max(0,int(a[0xFC])-int(b[0xFC]))/255
            if live(b):
                s=map_section(b)
                if self.frontiers[s]>=0:e['new_progress']=max(0,word(b,0x56)-self.frontiers[s])/16
                if s>0 and s not in self.visited:e['checkpoint_reached']=1.
            for gate in GATES:
                if a[gate]==0 and b[gate]==1 and gate not in self.objectives:e['objective_cleared']+=1
        if completed(a,b) and not self.completed:
            e['level_completed']=1.;self.completed=True
        if int(b[0x60])==255 and signed_byte(b[0xF7])<0 and not (int(a[0x60])==255 and signed_byte(a[0xF7])<0):
            e['game_over']=1.
        if getattr(self,'section0_timeout_terminal_enabled',False) and section0_timeout_terminal(a,b):e['game_over']=1.
        self.seed(b,combat=False)
        return e


class HumanRoute:
    def __init__(self,directory=None,config=None):
        directory=directory if directory is not None else (config or {}).get('demonstration_directory',OUT)
        self.directory=Path(directory)
        if not self.directory.is_absolute():self.directory=ROOT/self.directory
        self.meta=json.loads((self.directory/'route.json').read_text())
        if not self.meta['verified'] or self.meta['interventions']:raise ValueError('Unverified/intervened route')
        path=self.directory/'route.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.meta['route_sha256']:raise ValueError('Route digest changed')
        with np.load(path,allow_pickle=False) as d:
            self.ram=d['ram'];self.buttons=d['buttons'];self.valid=d['valid_starts']
        self.start=self.meta['level_start'];self.end=self.meta['completion_frame']
        self.valid=self.valid[(self.valid>=self.start)&(self.valid<self.end)]
        if config and config.get('latest_action_dependent_reset') is not None:
            # Dispatcher22 still permits movement; it is not proof of a passive
            # cutscene. This upper bound has a replay/NOOP counterfactual witness.
            self.valid=self.valid[self.valid<=config['latest_action_dependent_reset']]
        else:
            # Preserve the original v1 reset interface exactly.
            self.valid=self.valid[self.ram[self.valid,0x98]!=22]
        self.state_frames=np.array(sorted(map(int,self.meta['states'])))
        self.cache={}
        # Prefix bookkeeping uses ONLY observed past states; no future frontiers.
        self.prefix_frontiers=np.full((len(self.ram),3),-1,np.int32)
        self.prefix_gates=np.zeros((len(self.ram),4),bool)
        h=TaskHistory()
        for i,r in enumerate(self.ram):
            if i>=self.start:h.seed(r)
            self.prefix_frontiers[i]=h.frontiers
            self.prefix_gates[i]=[a in h.objectives for a in GATES]

    def tracker(self,frame):
        h=TaskHistory();h.frontiers=self.prefix_frontiers[frame].tolist()
        h.visited={s for s,x in enumerate(h.frontiers) if x>=0}
        h.objectives={a for a,yes in zip(GATES,self.prefix_gates[frame]) if yes}
        h.kind18.seed(self.ram[frame])
        return h

    def restore(self,native,frame,*,discard_observer_events=False):
        if frame not in self.valid:raise ValueError('Invalid or already completing curriculum start')
        if discard_observer_events and not all(hasattr(native,name) for name in ('read_events','overflow')):
            raise TypeError('Observer-event discard requires an experimental observer owner')
        base=int(self.state_frames[self.state_frames<=frame][-1])
        if base not in self.cache:
            if len(self.cache)>=64:self.cache.pop(next(iter(self.cache)))
            self.cache[base]=zlib.decompress((self.directory/'states'/self.meta['states'][str(base)]).read_bytes())
        native.load_state(self.cache[base])
        np.testing.assert_array_equal(native.ram_buffer(),self.ram[base])
        for i in range(base,frame):
            native.frame_advance(int(self.buttons[i]))
            np.testing.assert_array_equal(native.ram_buffer(),self.ram[i+1])
            if discard_observer_events:
                native.read_events()  # Route reconstruction is baseline only, never an episode event.
                if native.overflow():raise RuntimeError('Combat observer overflow during route restoration')
        return frame-base


class Level1Env:
    def __init__(self,config,route=None):
        if config.get('section0_timeout_terminal',False) and config.get('terminal_version')!=TIMEOUT_TERMINAL_VERSION:
            raise ValueError('Timeout terminal requires its explicit version')
        if config.get('final_encounter_potential',False):
            from .level1_potential import VERSION
            if config.get('potential_version',config.get('reward_version'))!=VERSION or 'combat_potential' not in config['weights']:
                raise ValueError('Potential reward requires its explicit version and coefficient')
        self.config=config;self.route=route or HumanRoute(config=config)
        self._observer_failure_dir=Path(config['_observer_failure_dir']) if config.get('_observer_failure_dir') else None
        self._observer_failure_sequence=0
        self.combat_observer_enabled=config.get('combat_observer_diagnostics',False)
        self.verified_enemy_kills=config.get('verified_enemy_kills',False)
        if not isinstance(self.verified_enemy_kills,bool):
            raise ValueError('verified_enemy_kills must be a boolean')
        kill_weight=config['weights'].get('enemy_kills',0.)
        if isinstance(kill_weight,bool) or not isinstance(kill_weight,(int,float)) or not np.isfinite(kill_weight) or kill_weight<0:
            raise ValueError('enemy_kills weight must be a finite nonnegative number')
        if kill_weight and not self.verified_enemy_kills:
            raise ValueError('Nonzero enemy_kills requires verified_enemy_kills=True')
        if self.verified_enemy_kills and not self.combat_observer_enabled:
            raise ValueError('verified_enemy_kills requires combat_observer_diagnostics=True')
        if self.verified_enemy_kills and config['weights'].get('abi2_kind18_defeat',0):
            raise ValueError('Use enemy_kills or abi2_kind18_defeat, not both; avoids duplicate defeat bonuses')
        if not isinstance(self.combat_observer_enabled,bool):
            raise ValueError('combat_observer_diagnostics must be a boolean')
        self.abi2_kind18_defeat_weight=config['weights'].get('abi2_kind18_defeat',0.0)
        if isinstance(self.abi2_kind18_defeat_weight,bool) or not isinstance(self.abi2_kind18_defeat_weight,(int,float)) or not np.isfinite(self.abi2_kind18_defeat_weight):
            raise ValueError('abi2_kind18_defeat weight must be a finite number')
        if self.abi2_kind18_defeat_weight and not self.combat_observer_enabled:
            raise ValueError('Nonzero abi2_kind18_defeat weight requires combat_observer_diagnostics=True')
        self.score_gain_weight=config.get('weights',{}).get('score_gain',0.0)
        if (isinstance(self.score_gain_weight,bool) or not isinstance(self.score_gain_weight,(int,float))
                or not np.isfinite(self.score_gain_weight) or self.score_gain_weight<0):
            raise ValueError('score_gain weight must be a finite nonnegative number')
        self.score_decrease_policy=config.get('score_decrease_policy','strict')
        if self.score_decrease_policy not in ('strict','quarantine_valid_bcd_unexplained_decrease'):
            raise ValueError('score_decrease_policy must be strict or quarantine_valid_bcd_unexplained_decrease')
        if self.combat_observer_enabled:
            from .kind18_observer import Kind18ObserverNative
            try:self.native=Kind18ObserverNative(ROOT/'roms/prince_valiant.nes')
            except Exception as exc:
                raise RuntimeError('Combat observer diagnostics failed to initialize (ABI/provenance check)') from exc
        else:self.native=RecorderNative(ROOT/'roms/prince_valiant.nes')
        self._combat_diagnostic_failure=None
        self._reset_combat_diagnostic_totals()
        self._reset_combat_reward_scope_totals()
        self.route_guidance=None
        self.route_waypoints=None
        if config.get('route_waypoint_version'):
            from .route_waypoints import RouteWaypoints
            if 'route_waypoint' not in config['weights']:raise ValueError('Missing route-goal coefficient')
            self.route_waypoints=RouteWaypoints(config)
        if config.get('route_guidance_version'):
            from .route_guidance import RouteGuidance
            if 'route_corridor' not in config['weights']:raise ValueError('Missing corridor coefficient')
            self.route_guidance=RouteGuidance(config)
        self.encoder=NavigationEncoder(config['feature_groups'],config.get('observation_version','legacy_v3'))
        self.catalog=tuple(config['action_masks']);self.allowed=np.ones(len(self.catalog),bool)
        self.history=FrameHistory(config['history_offsets']);self.done=True
        self.controls=None
        self.discounted_totals=Counter();self.discounted_totals_valid=False

    @staticmethod
    def _empty_combat_diagnostic_counts():
        return dict(accepted_hits=0,defeats=0,by_enemy_kind={
            '14':dict(accepted_hits=0,defeats=0),
            '18':dict(accepted_hits=None,defeats=0)},
            unverified=dict(defeats=0,by_enemy_kind={'18':dict(defeats=0)}))

    def _reset_combat_diagnostic_totals(self):
        self._combat_diagnostic_totals=self._empty_combat_diagnostic_counts()
        self._last_combat_diagnostic_step=None
        self._last_combat_reward_coverage=None

    @staticmethod
    def _empty_combat_reward_scope_totals():
        return dict(supported_native_frames=0,unsupported_native_frames=0)

    def _reset_combat_reward_scope_totals(self):
        self._combat_reward_scope_totals=self._empty_combat_reward_scope_totals()

    def _reset_score_gain_tracking(self):
        self._score_gain_baseline=None
        self._score_gain_points=0
        self._score_gain_native_frames=0
        self._score_gain_last_decision=dict(native_frames=0,valid_native_frames=0,
            quarantined_native_frames=0,score_points=None,valid_score_points=0,
            score_reward_applied=0.,zero_applied_by_policy=False,measured_zero_native_frames=0)
        self._score_gain_failure=None
        self._score_gain_failure_details=None
        self._score_gain_quarantined=False
        self._score_gain_quarantine_reason=None
        self._score_gain_quarantine_details=None
        self._score_gain_quarantined_native_frames=0
        self._score_gain_quarantine_started=False

    @staticmethod
    def _score_gain_endpoint_diagnostic(ram):
        """Bounded evidence only; never copy the complete live RAM image."""
        return dict(score_bcd_f4_f6=[int(x) for x in ram[0xF4:0xF7]],
            state_60=int(ram[0x60]),lives_f7_raw=int(ram[0xF7]),
            lives_f7_signed=signed_byte(ram[0xF7]),health_fc=int(ram[0xFC]),
            retry_continue_02ff=int(ram[0x02FF]))

    def _fail_score_gain(self,message,exc=None,details=None):
        self._score_gain_failure=str(message)
        self._score_gain_failure_details=copy.deepcopy(details)
        suffix=(f'; diagnostic={json.dumps(details,sort_keys=True,separators=(",",":"))}'
                if details is not None else '')
        error=RuntimeError(f'Score reward diagnostics failed: {message}{suffix}')
        if exc is not None:raise error from exc
        raise error

    def _seed_score_gain_baseline(self,ram):
        if not self.score_gain_weight:return
        try:self._score_gain_baseline=decode_display_score(ram)
        except Exception as exc:self._fail_score_gain('could not establish a valid packed-BCD score baseline',exc)

    def _score_gain_transition(self,before,after,*,episode_native_frame=None):
        if not self.score_gain_weight:return None
        if self._score_gain_failure is not None:
            self._fail_score_gain('previous score decoder failure invalidated this episode; reset before continuing')
        capture_frame_index=getattr(self,'_score_clear_native_frame_index',None)
        if getattr(self,'_score_clear_capture',None) is not None and capture_frame_index is not None:
            capture_frame_index=max(0,int(capture_frame_index)-1)
        worker_pid=getattr(self,'_score_gain_worker_pid',None)
        context=dict(episode_native_frame=episode_native_frame,
            episode_elapsed_before=int(getattr(self,'elapsed',0)),
            game_elapsed_before=int(getattr(self,'game_elapsed',0)),
            reset_route_frame=getattr(self,'initial_frame',None),
            worker_pid=worker_pid,
            frame_identity=dict(worker_id=getattr(self,'_score_gain_worker_id',worker_pid),
                worker_pid=worker_pid,native_frame_index=capture_frame_index,
                episode_id=getattr(self,'_score_gain_episode_id',None),
                episode_native_frame=episode_native_frame),
            tracker_baseline=self._score_gain_baseline,
            before=self._score_gain_endpoint_diagnostic(before),
            after=self._score_gain_endpoint_diagnostic(after),
            raw_before_ram=bytes(np.asarray(before,dtype=np.uint8)).hex(),
            raw_after_ram=bytes(np.asarray(after,dtype=np.uint8)).hex())
        try:
            old_score=decode_display_score(before);new_score=decode_display_score(after)
        except Exception as exc:
            context['decoded_before']=None
            context['decoded_after']=None
            self._fail_score_gain('invalid packed-BCD score during native transition',exc,context)
        context['decoded_before']=old_score
        context['decoded_after']=new_score
        if self._score_gain_quarantined:
            self._score_gain_baseline=new_score  # endpoint cursor only; the aggregate remains unavailable
            self._score_gain_quarantined_native_frames+=1
            return None
        if self._score_gain_baseline is None:self._score_gain_baseline=old_score
        if old_score!=self._score_gain_baseline:
            context['tracker_baseline']=self._score_gain_baseline
            self._fail_score_gain(f'score baseline mismatch: expected {self._score_gain_baseline}, observed {old_score}',details=context)
        if new_score<old_score:
            reason=f'unexplained displayed-score decrease {old_score}->{new_score}; rollover/reset is unverified'
            if self.score_decrease_policy=='strict':self._fail_score_gain(reason,details=context)
            self._score_gain_quarantined=True
            self._score_gain_baseline=new_score
            self._score_gain_quarantine_reason=reason
            self._score_gain_quarantine_details=copy.deepcopy(context)
            self._score_gain_quarantine_started=True
            self._score_gain_quarantined_native_frames+=1
            return None
        delta=new_score-old_score
        self._score_gain_baseline=new_score
        self._score_gain_points+=delta
        self._score_gain_native_frames+=1
        return delta

    def _fail_combat_diagnostics(self,message,exc=None):
        self._combat_diagnostic_failure=str(message)
        error=RuntimeError(f'Combat observer diagnostics failed: {message}')
        if exc is not None:raise error from exc
        raise error

    def _drain_combat_observer(self):
        if not self.combat_observer_enabled:return []
        try:
            events=self.native.read_events()
            overflow=self.native.overflow()
        except Exception as exc:self._fail_combat_diagnostics('observer read/status call failed',exc)
        if overflow:self._fail_combat_diagnostics('ABI2 event buffer overflow')
        if not isinstance(events,(list,tuple)):
            self._fail_combat_diagnostics('observer event data is missing or malformed')
        return events

    @staticmethod
    def _kind18_defeat_scope_available(before,after):
        return (supported(before) and supported(after) and map_section(before)==0
            and map_section(after)==0)

    def _validate_scoped_kind18_defeat(self,event,before,after):
        try:
            slot=self._abi2_integer(event,'target_slot')
            projectile_slot=self._abi2_integer(event,'projectile_slot')
            handler_bank=self._abi2_integer(event,'handler_bank')
            commit_pc=self._abi2_integer(event,'commit_pc')
            lethal_pc=self._abi2_integer(event,'lethal_pc')
            counter_before=self._abi2_integer(event,'counter_before')
            counter_after=self._abi2_integer(event,'counter_after')
            if not self._kind18_defeat_scope_available(before,after):
                return False
            valid=(slot in (8,9) and projectile_slot==2
                and handler_bank==2 and commit_pc==0x87F6 and lethal_pc==0x87FC
                and counter_before-counter_after==1
                and (counter_before&0x1F)==1 and (counter_after&0x1F)==0
                and int(before[0xC1+slot])==18 and int(after[0xC1+slot]) in (7,8))
        except Exception as exc:self._fail_combat_diagnostics('malformed scoped kind18 defeat',exc)
        if not valid:self._fail_combat_diagnostics('contradictory kind18 defeat record inside verified section0 coverage')
        return True

    def _abi2_integer(self,event,key):
        value=event[key]
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)):
            raise ValueError(f'ABI2 field {key} is not an integer')
        return int(value)

    def _preserve_abi2_failure(self,event,before,after,exc):
        """Persist bounded evidence in the worker before a rejected event escapes."""
        if self._observer_failure_dir is None:return
        def byte(ram,address):return int(ram[address]) if 0<=address<len(ram) else None
        raw={}
        if isinstance(event,dict):
            for index,(key,value) in enumerate(event.items()):
                if index>=16:break
                name=str(key)[:64]
                if isinstance(value,(str,int,float,bool)) or value is None:
                    rendered=value if not isinstance(value,str) else value[:256]
                else:rendered=repr(value)[:256]
                raw[name]=dict(type=type(value).__name__,value=rendered)
        else:raw={'record':dict(type=type(event).__name__,value=repr(event)[:256])}
        slot=event.get('target_slot') if isinstance(event,dict) else None
        slot=int(slot) if isinstance(slot,(int,np.integer)) and not isinstance(slot,bool) and 0<=int(slot)<10 else None
        identity=getattr(self.native,'abi_identity',None)
        context=dict(worker_pid=os.getpid(),observer_abi=identity or 2,
            event_native_frame=event.get('native_frame') if isinstance(event,dict) else None,
            environment_elapsed_frames=int(getattr(self,'elapsed',0)),
            target_slot=slot,
            before=dict(mode=byte(before,0x33),map=byte(before,0x4D),section=map_section(before),
                target_kind=byte(before,0xC1+slot) if slot is not None else None,
                target_counter=byte(before,0xB7+slot) if slot is not None else None,
                projectile_bytes={hex(a):byte(before,a) for a in (0x501,0x509,0x50B,0x50D)}),
            after=dict(mode=byte(after,0x33),map=byte(after,0x4D),section=map_section(after),
                target_kind=byte(after,0xC1+slot) if slot is not None else None,
                target_counter=byte(after,0xB7+slot) if slot is not None else None,
                projectile_bytes={hex(a):byte(after,a) for a in (0x501,0x509,0x50B,0x50D)}),
            raw_fields=raw,exception=dict(type=type(exc).__name__,message=str(exc)[:1000]))
        directory=self._observer_failure_dir
        directory.mkdir(parents=True,exist_ok=True)
        self._observer_failure_sequence+=1
        target=directory/f"abi2_failure_worker{os.getpid()}_{self._observer_failure_sequence:04d}.json"
        temporary=target.with_suffix('.json.tmp')
        with temporary.open('w',encoding='utf-8') as stream:
            json.dump(context,stream,indent=2,sort_keys=True,default=str)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,target)
        return target

    def _count_combat_observer_events(self,events,counts,records,seen,before,after):
        scoped_kind18_defeats=0
        for event in events:
            try:
                event_type=self._abi2_integer(event,'event_type')
                kind=self._abi2_integer(event,'enemy_kind')
                signature=tuple(self._abi2_integer(event,k) for k in ('native_frame','event_type','target_slot',
                    'projectile_slot','enemy_kind','handler_bank','commit_pc','lethal_pc','counter_before','counter_after'))
                native_frame,target_slot,projectile_slot=signature[0],signature[2],signature[3]
                handler_bank,commit_pc,lethal_pc,counter_before,counter_after=signature[5:]
                if (event_type not in (1,2) or kind not in (14,18) or native_frame<0
                        or not 0<=target_slot<10 or not 0<=projectile_slot<4
                        or not 0<=handler_bank<=0xFF
                        or not 0<=commit_pc<=0xFFFF or not 0<=lethal_pc<=0xFFFF
                        or not 0<=counter_before<=0xFF or not 0<=counter_after<=0xFF
                        or (event_type==1 and kind!=14)):
                    raise ValueError('ABI2 record fields are outside the observed event schema')
            except Exception as exc:
                try:artifact=self._preserve_abi2_failure(event,before,after,exc)
                except Exception as preserve_exc:exc.add_note(f'ABI2 diagnostic persistence failed: {preserve_exc!r}')
                else:exc.add_note(f'ABI2 diagnostic artifact: {artifact}')
                self._fail_combat_diagnostics('malformed ABI2 event record',exc)
            valid_counter_transition=(counter_before>0 and counter_after==counter_before-1)
            if event_type==2 and kind==18:
                # Kind18 stores flags in the high three bits. Its defeat path
                # commits low-five-bit 1->0, preserves those flags, then takes
                # the zero/lethal branch. The full byte need not be 1->0.
                valid_counter_transition=((counter_before&0x1F)==1
                    and (counter_after&0x1F)==0
                    and (counter_before&0xE0)==(counter_after&0xE0))
            elif event_type==2 and kind==14:
                valid_counter_transition=(counter_before,counter_after)==(1,0)
            if not valid_counter_transition:
                exc=ValueError('ABI2 event counter transition contradicts its producer semantics')
                try:artifact=self._preserve_abi2_failure(event,before,after,exc)
                except Exception as preserve_exc:exc.add_note(f'ABI2 diagnostic persistence failed: {preserve_exc!r}')
                else:exc.add_note(f'ABI2 diagnostic artifact: {artifact}')
                self._fail_combat_diagnostics('contradictory ABI2 event semantics',exc)
            if signature in seen:
                exc=ValueError('duplicate ABI2 event record')
                try:artifact=self._preserve_abi2_failure(event,before,after,exc)
                except Exception as preserve_exc:exc.add_note(f'ABI2 diagnostic persistence failed: {preserve_exc!r}')
                else:exc.add_note(f'ABI2 diagnostic artifact: {artifact}')
                self._fail_combat_diagnostics('duplicate ABI2 event record',exc)
            seen.add(signature)
            if event_type==1 and kind==14:
                counts['accepted_hits']+=1;counts['by_enemy_kind']['14']['accepted_hits']+=1
            elif event_type==2 and kind in (14,18):
                if kind==18:
                    try:in_scope=self._validate_scoped_kind18_defeat(event,before,after)
                    except Exception as exc:
                        try:artifact=self._preserve_abi2_failure(event,before,after,exc)
                        except Exception as preserve_exc:exc.add_note(f'ABI2 diagnostic persistence failed: {preserve_exc!r}')
                        else:exc.add_note(f'ABI2 diagnostic artifact: {artifact}')
                        raise
                    if in_scope:
                        scoped_kind18_defeats+=1
                        counts['defeats']+=1;counts['by_enemy_kind']['18']['defeats']+=1
                    else:
                        counts['unverified']['defeats']+=1
                        counts['unverified']['by_enemy_kind']['18']['defeats']+=1
                else:
                    counts['defeats']+=1;counts['by_enemy_kind']['14']['defeats']+=1
            else:
                exc=ValueError(f'unsupported event type/kind pair {event_type}/{kind}')
                try:artifact=self._preserve_abi2_failure(event,before,after,exc)
                except Exception as preserve_exc:exc.add_note(f'ABI2 diagnostic persistence failed: {preserve_exc!r}')
                else:exc.add_note(f'ABI2 diagnostic artifact: {artifact}')
                self._fail_combat_diagnostics(str(exc),exc)
            records.append(dict(event))
        return scoped_kind18_defeats

    def encode(self,r,previous,h,last_action,action_valid,controls=None):
        o=self.encoder(r,previous,h)
        o['categorical']=np.r_[o['categorical'],int(last_action)].astype(np.int64)
        o['scalars']=np.r_[o['scalars'],[int(a in h.objectives) for a in GATES],float(action_valid)].astype(np.float32)
        if self.config.get('hud_deadline_features',False):
            digits=[int(x) for x in r[0x286:0x28A]]
            valid=supported(r) and all(0<=x<=9 for x in digits) and digits[2]<6
            seconds=(60*(digits[0]*10+digits[1])+10*digits[2]+digits[3]) if valid else 0
            o['scalars']=np.r_[o['scalars'],seconds/600,int(r[0x28A])/255 if valid else 0,float(valid)].astype(np.float32)
            o['categorical']=np.r_[o['categorical'],int(r[0x98])].astype(np.int64)
        if self.config.get('control_edge_features',False):
            controls=controls if controls is not None else self.controls
            if controls is None:raise ValueError('Exact past controller history required')
            values=controls.observable() if 'controller' in self.encoder.groups else np.zeros(12,np.float32)
            o['scalars']=np.r_[o['scalars'],values].astype(np.float32)
        if self.config.get('verified_counter_feature',False):
            valid=map_section(r)==2 and int(r[0xC1])==14 and int(r[0x99])==1 and 1<=int(r[0xB7])<=24
            o['scalars']=np.r_[o['scalars'],int(r[0xB7])/24 if valid else 0.,float(valid)].astype(np.float32)
        if self.config.get('explicit_map_identity',False):
            section=map_section(r)
            o['categorical']=np.r_[o['categorical'],255 if section is None else section].astype(np.int64)
            o['scalars']=np.r_[o['scalars'],float(section is not None)].astype(np.float32)
        if self.config.get('vanilla_defense_codes',False):
            # Authentic RAM only. This environment never constructs training
            # intervention backends; do not apply this schema to edited scenes.
            o['categorical']=np.r_[o['categorical'],int(r[0xF3]),int(r[0x284])].astype(np.int64)
        return self.encoder.append_rea_structured_features(r,o)

    def reset(self,frame=None,*,horizon=None,variation=0,training_segment=False,episodic_life=False):
        frame=self.route.start if frame is None else int(frame)
        self._combat_diagnostic_failure=None
        self._reset_combat_diagnostic_totals()
        self._reset_combat_reward_scope_totals()
        self._reset_score_gain_tracking()
        self.episodic_life=bool(episodic_life)
        self.life_index=0;self.game_elapsed=0;self.last_end_reason=None
        self.training_segment=bool(training_segment and frame!=self.route.start)
        self.initial_frame=frame
        try:
            self.replay_frames=self.route.restore(self.native,frame,
                discard_observer_events=self.combat_observer_enabled)
            if self.combat_observer_enabled:
                # Clear pending observer context and its local frame counter after replay.
                self.native.load_state(self.native.dump_state())
        except Exception as exc:
            if self.combat_observer_enabled:
                self._fail_combat_diagnostics('route restoration failed',exc)
            raise
        # Only a completed trusted route restore starts a new score episode.
        self._reset_score_gain_tracking()
        self.tracker=self.route.tracker(frame);self.elapsed=0;self.done=False
        self.tracker.section0_timeout_terminal_enabled=bool(self.config.get('section0_timeout_terminal',False))
        self.horizon=int(horizon or min(self.config['full_horizon'],max(self.config['minimum_episode_frames'],
            (self.route.end-frame)*self.config['horizon_suffix_multiplier']+self.config['horizon_extra_frames'])))
        self.totals=Counter();self.raw_totals=Counter();self.discounted_totals=Counter();self.discounted_totals_valid=True
        self.last_action=int(self.route.buttons[frame-1]) if frame>self.route.start else 0
        earliest=max(self.route.start,frame+min(self.config['history_offsets']))
        self.controls=(ControlHistory.from_recording(self.route.buttons,earliest,self.route.start,
                       self.config.get('control_age_cap_frames',120)) if self.config.get('control_edge_features',False) else None)
        for index,f in enumerate(range(earliest,frame+1)):
            if index and self.controls is not None:self.controls.push(int(self.route.buttons[f-1]))
            r=self.route.ram[f];previous=self.route.ram[f-1] if f>self.route.start else None
            encoded=self.encode(r,previous,self.route.tracker(f),int(self.route.buttons[f-1]) if previous is not None else 0,previous is not None)
            if index==0:self.history.reset(encoded)
            else:self.history.push(index,encoded)
        self.ram=self.native.ram_buffer().copy()
        # Evaluation-only waiting variations, before policy control, no rewards.
        for _ in range(int(variation)):
            old=self.ram;self.native.frame_advance(0);self.ram=self.native.ram_buffer().copy()
            if self.combat_observer_enabled:self._drain_combat_observer()  # variation is pre-episode baseline
            if self.controls is not None:self.controls.push(0)
            if not live(self.ram) or map_section(self.ram)!=0 or word(self.ram,0x56)!=word(self.route.ram[self.route.start],0x56):
                raise ValueError('Full-start variation moved the player or left valid initial scene')
            self.tracker.seed(self.ram);self.history.push(self.history.current+1,self.encode(self.ram,old,self.tracker,0,True))
            self.last_action=0
        if self.combat_observer_enabled and variation:
            self.native.load_state(self.native.dump_state())
        self.variation=int(variation)
        # Restored score and any evaluation-only waiting variation establish a
        # baseline; neither is a newly earned score event.
        self._seed_score_gain_baseline(self.ram)
        return self.history.window(),self.info()

    def info(self):
        diagnostics_enabled=self.combat_observer_enabled
        if not diagnostics_enabled:
            combat_diagnostics=dict(enabled=False,status='disabled',observer_abi=None,
                availability=dict(accepted_hits=False,defeats=False),counts=None,episode_totals=None,
                decision_native_frames=0,records=None,failure=None)
        elif self._combat_diagnostic_failure is not None:
            combat_diagnostics=dict(enabled=True,status='failed',observer_abi=2,
                availability=dict(accepted_hits=False,defeats=False),counts=None,episode_totals=None,
                decision_native_frames=0,records=None,failure=self._combat_diagnostic_failure)
        else:
            combat_diagnostics=dict(enabled=True,status='available',observer_abi=2,
                availability=dict(accepted_hits=True,defeats=True,
                    by_enemy_kind={'14':dict(accepted_hits=True,defeats=True),
                                   '18':dict(accepted_hits=False,defeats=True)}),
                counts=(copy.deepcopy(self._last_combat_diagnostic_step['counts'])
                    if self._last_combat_diagnostic_step is not None else None),
                episode_totals=copy.deepcopy(self._combat_diagnostic_totals),
                decision_native_frames=(self._last_combat_diagnostic_step['native_frames']
                    if self._last_combat_diagnostic_step is not None else 0),
                records=(copy.deepcopy(self._last_combat_diagnostic_step['records'])
                    if self._last_combat_diagnostic_step is not None else None),failure=None)
        combat_diagnostics['coverage']=copy.deepcopy(COMBAT_OBSERVER_DIAGNOSTIC_COVERAGE)
        current_scope=(hasattr(self,'ram') and self._kind18_defeat_scope_available(self.ram,self.ram))
        coverage=self._last_combat_reward_coverage
        if coverage is None:
            reward_scope_status='failed' if self._combat_diagnostic_failure else ('available' if current_scope else 'unavailable')
            coverage=dict(status=reward_scope_status,supported_native_frames=0,unavailable_native_frames=0,
                event_count=0)
        else:coverage=copy.deepcopy(coverage)
        coverage.update(weight=self.abi2_kind18_defeat_weight,
            observer_required=bool(self.abi2_kind18_defeat_weight),
            observed=self.combat_observer_enabled and self._combat_diagnostic_failure is None,
            observer_status=('failed' if self._combat_diagnostic_failure else
                'available' if self.combat_observer_enabled else 'disabled'),
            scope=copy.deepcopy(ABI2_KIND18_DEFEAT_REWARD_COVERAGE))
        coverage.update(episode_supported_native_frames=self._combat_reward_scope_totals['supported_native_frames'],
            episode_unsupported_native_frames=self._combat_reward_scope_totals['unsupported_native_frames'])
        score_failed=self._score_gain_failure is not None
        quarantined=bool(getattr(self,'_score_gain_quarantined',False))
        score_available=bool(self.score_gain_weight and not score_failed and not quarantined)
        score_status='disabled' if not self.score_gain_weight else ('failed' if score_failed else
            'quarantined' if quarantined else 'available')
        score_decision=self._score_gain_last_decision
        score_coverage=dict(status=score_status,enabled=bool(self.score_gain_weight),available=score_available,
            policy=getattr(self,'score_decrease_policy','strict'),
            quarantined=quarantined,quarantined_episodes=int(quarantined),
            episode_quarantined_native_frames=int(getattr(self,'_score_gain_quarantined_native_frames',0)),
            quarantine_started_this_decision=bool(getattr(self,'_score_gain_quarantine_started',False)),
            quarantine_reason=getattr(self,'_score_gain_quarantine_reason',None),
            quarantine_details=copy.deepcopy(getattr(self,'_score_gain_quarantine_details',None)),
            weight=self.score_gain_weight,units='displayed score points',decoder='strict packed BCD RAM $F4-$F6',
            current_score_points=(self._score_gain_baseline if score_available else None),
            episode_positive_score_points=(self._score_gain_points if score_available else None),
            # A failed episode can still report how many transitions were
            # decoded before failure; its aggregate score remains unavailable.
            episode_decoded_native_frames=(self._score_gain_native_frames if self.score_gain_weight else 0),
            last_decision_native_frames=score_decision['native_frames'],
            last_decision_valid_native_frames=score_decision.get('valid_native_frames',score_decision['native_frames']),
            last_decision_quarantined_native_frames=score_decision.get('quarantined_native_frames',0),
            last_decision_valid_score_points=int(score_decision.get('valid_score_points',0)),
            last_decision_score_points=(score_decision['score_points'] if score_decision.get('valid_native_frames',score_decision['native_frames'])==score_decision['native_frames'] else None),
            last_decision_score_reward_applied=float(score_decision.get('score_reward_applied',0.)),
            last_decision_zero_applied_by_policy=bool(score_decision.get('zero_applied_by_policy',False)),
            last_decision_zero_reward_native_frames=int(score_decision.get('quarantined_native_frames',0)),
            last_decision_measured_zero_native_frames=int(score_decision.get('measured_zero_native_frames',0)),
            observed_zero=(score_decision['score_points']==0 if score_available and score_decision['native_frames'] else None),
            coverage=copy.deepcopy(SCORE_GAIN_REWARD_COVERAGE),
            failure=self._score_gain_failure,
            failure_details=copy.deepcopy(getattr(self,'_score_gain_failure_details',None)))
        return dict(section=map_section(self.ram),player_x=word(self.ram,0x56),player_y=word(self.ram,0x58),
            ram_sha256=hashlib.sha256(self.ram.tobytes()).hexdigest(),
            health=int(self.ram[0xFC]),lives=signed_byte(self.ram[0xF7]),elapsed_frames=self.elapsed,episode_horizon_frames=self.horizon,
            objective_flags_raw={hex(a):int(self.ram[a]) for a in GATES},
            timer_digits=self.ram[0x286:0x28A].tolist(),timer_subcounter_raw=int(self.ram[0x28A]),player_dispatch_raw=int(self.ram[0x98]),
            player_state_raw=int(self.ram[0x5F]),life_phase_raw=int(self.ram[0x60]),
            slot0_debug_raw=dict(kind=int(self.ram[0xC1]),phase=int(self.ram[0x99]),counter=int(self.ram[0x2BA]),
                x=word(self.ram,0x28C),y=word(self.ram,0x2A0)),
            frontiers=self.tracker.frontiers.copy(),completion=self.tracker.completed,
            full_start=self.initial_frame==self.route.start and self.life_index==0,reset_frame=self.initial_frame,
            game_origin_frame=self.initial_frame,full_start_game=self.initial_frame==self.route.start,
            training_segment=self.training_segment,
            episodic_life=self.episodic_life,life_index=self.life_index,game_elapsed_frames=self.game_elapsed,
            reset_replay_frames=self.replay_frames,variation=self.variation,
            reward_totals=dict(self.totals),event_totals=dict(self.raw_totals),interventions=[],
            combat_event_coverage=copy.deepcopy(KIND18_COVERAGE),
            combat_observer_diagnostics=combat_diagnostics,
            combat_reward_coverage=dict(abi2_kind18_defeat=coverage,enemy_kills=dict(
                enabled=self.verified_enemy_kills,complete=False,
                status=('available' if self.verified_enemy_kills and not self._combat_diagnostic_failure else
                        'failed' if self.verified_enemy_kills else 'disabled'),
                units='verified ABI2 defeat events; partial enemy coverage',
                supported_kinds=[14,18],uncovered='Other kinds and defeat sources outside ABI2 verified paths',
                weight=self.config['weights'].get('enemy_kills',0.),
                scope=copy.deepcopy(COMBAT_OBSERVER_DIAGNOSTIC_COVERAGE['by_enemy_kind']))),
            score_gain_coverage=score_coverage,
            discounted_reward_totals=dict(self.discounted_totals),discounted_return_valid=self.discounted_totals_valid,
            discounted_return_value=sum(self.discounted_totals.values()) if self.discounted_totals_valid else None,
            discounted_base_return_value=sum(v for k,v in self.discounted_totals.items() if k not in ('combat_potential','route_corridor','route_waypoint')) if self.discounted_totals_valid else None)

    def step(self,action):
        if not 0<=int(action)<len(self.catalog):raise ValueError('Action index outside catalog')
        return self.step_mask(self.catalog[int(action)])

    def step_mask(self,mask):
        """Advance one native-frame action mask through the normal task/history path."""
        if self.done:raise RuntimeError('Reset ended episode')
        if self.abi2_kind18_defeat_weight and self._combat_diagnostic_failure is not None:
            self._fail_combat_diagnostics('previous observer failure invalidated this episode; reset before continuing')
        if self.score_gain_weight and self._score_gain_failure is not None:
            self._fail_score_gain('previous score decoder failure invalidated this episode; reset before continuing')
        mask=int(mask)
        if not 0<=mask<=255:raise ValueError('Controller mask must be a byte')
        reward=0.;components=Counter();discounted_components=Counter();raw=Counter();availability={};duration=0;reason=None
        combat_frames=0
        score_decision_frames=0;score_decision_points=0;score_quarantined_frames=0;score_measured_zero_frames=0
        self._score_gain_quarantine_started=False
        diagnostic_counts=self._empty_combat_diagnostic_counts() if self.combat_observer_enabled else None
        diagnostic_records=[];diagnostic_seen=set();diagnostic_frames=0
        scoped_kind18_defeats=0;scoped_native_frames=0;unavailable_native_frames=0
        for _ in range(self.config['frame_skip']):
            previous_maxima=self.tracker.frontiers.copy() if self.route_waypoints is not None else None
            old=self.ram;self.native.frame_advance(mask);self.ram=self.native.ram_buffer().copy()
            capture=getattr(self,'_score_clear_capture',None)
            captured_transition=None
            if capture is not None:
                captured_transition=capture.before_validation(pre_ram=old,post_ram=self.ram,mask=mask,
                    elapsed_before=self.elapsed+duration,game_elapsed_before=self.game_elapsed+duration,
                    reset_frame=self.initial_frame,baseline=self._score_gain_baseline,
                    reset_provenance=getattr(self,'_score_clear_reset_provenance',None),
                    native_frame_index=int(getattr(self,'_score_clear_native_frame_index',0)))
                self._score_clear_native_frame_index=getattr(self,'_score_clear_native_frame_index',0)+1
            frame_scope=self._kind18_defeat_scope_available(old,self.ram)
            if frame_scope:
                scoped_native_frames+=1
                self._combat_reward_scope_totals['supported_native_frames']+=1
            else:
                unavailable_native_frames+=1
                self._combat_reward_scope_totals['unsupported_native_frames']+=1
            if self.combat_observer_enabled:
                defeats_before=diagnostic_counts['defeats']
                frame_events=self._drain_combat_observer()  # exactly one destructive read per native frame
                diagnostic_frames+=1
                frame_kind18_defeats=self._count_combat_observer_events(
                    frame_events,diagnostic_counts,diagnostic_records,diagnostic_seen,old,self.ram)
                scoped_kind18_defeats+=frame_kind18_defeats
            else:frame_kind18_defeats=0
            try:
                score_gain_points=self._score_gain_transition(old,self.ram,
                    episode_native_frame=self.elapsed+duration+1)
            except Exception as exc:
                if capture is not None:
                    captured_transition['terminal_marker']='not_classified_before_score_validation_failure'
                    captured_transition['task_events']='unavailable_score_validation_precedes_task_transition'
                    try:
                        capture.persist(failing_transition=captured_transition,error=repr(exc))
                    except Exception as persist_exc:
                        exc.add_note(f'score-clear ring persistence failed: {persist_exc!r}')
                raise
            e=self.tracker.transition(old,self.ram)
            if self.verified_enemy_kills:
                e['enemy_kills']=float(diagnostic_counts['defeats']-defeats_before)
            if score_gain_points is not None:
                e['score_gain']=float(score_gain_points)
                score_decision_frames+=1;score_decision_points+=score_gain_points
                score_measured_zero_frames+=int(score_gain_points==0)
            elif self.score_gain_weight and self._score_gain_quarantined:
                score_quarantined_frames+=1
            combat_frames+=int(e[KIND18_KEYS[0]] is not None)
            if self.route_waypoints is not None:e['route_waypoint']=self.route_waypoints.transition(old,self.ram,previous_maxima)
            if self.route_guidance is not None:
                terminal=self.tracker.completed or bool(e['game_over']) or (int(self.ram[0x60])==255 and signed_byte(self.ram[0xF7])<0) or (self.episodic_life and e['lives_lost']>0)
                e['route_corridor']=self.route_guidance.difference(old,self.ram,self.config['gamma'],bootstrap_valid=not terminal and supported(self.ram))
            if self.config.get('final_encounter_potential',False):
                from .level1_potential import potential_difference
                terminal=self.tracker.completed or bool(e['game_over']) or (int(self.ram[0x60])==255 and signed_byte(self.ram[0xF7])<0) or (self.episodic_life and e['lives_lost']>0)
                e['combat_potential']=potential_difference(old,self.ram,self.config['gamma'],bootstrap_valid=not terminal and supported(self.ram))
            for k,value in e.items():
                if k in KIND18_KEYS:availability[k]=availability.get(k,False) or value is not None
                else:availability[k]=value is not None
                if value is not None:
                    if k in KIND18_KEYS:
                        raw[k]+=value
                        continue  # Diagnostics only, regardless of config weights.
                    weighted=value*self.config['weights'][k]
                    raw[k]+=value;components[k]+=weighted
                    discounted_components[k]+=self.config['gamma']**duration*weighted
                    self.discounted_totals[k]+=self.config['gamma']**self.elapsed*weighted
                    reward+=self.config['gamma']**duration*weighted
            if frame_kind18_defeats:
                count=float(frame_kind18_defeats)
                weighted=count*self.abi2_kind18_defeat_weight
                raw['abi2_kind18_defeat']+=count
                if weighted:
                    components['abi2_kind18_defeat']+=weighted
                    discounted_components['abi2_kind18_defeat']+=self.config['gamma']**duration*weighted
                    self.discounted_totals['abi2_kind18_defeat']+=self.config['gamma']**self.elapsed*weighted
                    reward+=self.config['gamma']**duration*weighted
            duration+=1;self.elapsed+=1;self.game_elapsed+=1;self.last_action=mask
            if self.controls is not None:self.controls.push(mask)
            self.history.push(self.history.current+1,self.encode(self.ram,old,self.tracker,mask,True))
            if self.tracker.completed:reason='level_completed'
            elif e['game_over'] or (int(self.ram[0x60])==255 and signed_byte(self.ram[0xF7])<0):reason='game_over'
            elif not supported(self.ram):reason='unsupported_mode'
            elif self.episodic_life and e['lives_lost']>0:reason='training_life_loss'
            elif self.training_segment and self.config.get('truncate_suffix_on_life_loss',False) and e['lives_lost']>0:
                # Collector reset boundary, not a game terminal. The actual
                # next native state/history remains valid for bootstrapping.
                reason='curriculum_life_loss'
            elif self.elapsed>=self.horizon:reason='time_limit'
            if capture is not None:
                capture.after_transition(captured_transition,task_events=e,terminal_marker=reason)
            if reason:break
        self.totals.update(components);self.raw_totals.update(raw);self.done=reason is not None;self.last_end_reason=reason
        self._score_gain_last_decision=dict(native_frames=score_decision_frames,
            valid_native_frames=score_decision_frames,
            quarantined_native_frames=score_quarantined_frames,
            score_points=(score_decision_points if self.score_gain_weight and not score_quarantined_frames else None),
            valid_score_points=score_decision_points,
            score_reward_applied=score_decision_points*self.score_gain_weight,
            zero_applied_by_policy=bool(score_quarantined_frames),
            measured_zero_native_frames=score_measured_zero_frames)
        self._score_gain_last_decision['native_frames']=duration if self.score_gain_weight else 0
        if score_quarantined_frames:availability['score_gain']=False
        elif self.score_gain_weight and score_decision_frames:availability['score_gain']=True
        if self.combat_observer_enabled:
            legacy_unverified=self._combat_diagnostic_totals.setdefault(
                'unverified',dict(defeats=0,by_enemy_kind={'18':dict(defeats=0)}))
            legacy_unverified.setdefault('defeats',0)
            legacy_unverified.setdefault('by_enemy_kind',{}).setdefault('18',dict(defeats=0)).setdefault('defeats',0)
            for name in ('accepted_hits','defeats'):
                self._combat_diagnostic_totals[name]+=diagnostic_counts[name]
            for kind in ('14','18'):
                for name in ('accepted_hits','defeats'):
                    value=diagnostic_counts['by_enemy_kind'][kind][name]
                    if value is not None:self._combat_diagnostic_totals['by_enemy_kind'][kind][name]+=value
            self._combat_diagnostic_totals['unverified']['defeats']+=diagnostic_counts['unverified']['defeats']
            self._combat_diagnostic_totals['unverified']['by_enemy_kind']['18']['defeats']+=(
                diagnostic_counts['unverified']['by_enemy_kind']['18']['defeats'])
            self._last_combat_diagnostic_step=dict(counts=diagnostic_counts,
                native_frames=diagnostic_frames,records=diagnostic_records)
        coverage_status=('available' if scoped_native_frames and not unavailable_native_frames
            else 'partial' if scoped_native_frames else 'unavailable')
        if self._combat_diagnostic_failure is not None:coverage_status='failed'
        self._last_combat_reward_coverage=dict(status=coverage_status,
            supported_native_frames=scoped_native_frames,unavailable_native_frames=unavailable_native_frames,
            event_count=scoped_kind18_defeats)
        info=self.info();info.update(transition_frames=duration,reward_components=dict(components),events=dict(raw),
            discounted_reward_components=dict(discounted_components),
            availability=availability,end_reason=reason,game_terminal=reason in ('level_completed','game_over'),
            training_task_terminal=reason=='training_life_loss',
            bootstrap_valid=reason not in ('level_completed','game_over','unsupported_mode','training_life_loss'))
        info['combat_event_coverage'].update(available_native_frames=combat_frames,evaluated_native_frames=duration)
        return self.history.window(),reward,reason in ('game_over','level_completed','training_life_loss'),reason in ('time_limit','unsupported_mode','curriculum_life_loss'),info

    def continue_life(self):
        """Start a value-learning episode without resetting or advancing the game.

        Keep actual history, controls and visited progress across the life loss.
        Evaluation never enables these virtual task boundaries.
        """
        if not self.episodic_life or not self.done or self.last_end_reason!='training_life_loss':
            raise RuntimeError('Only a verified training life-loss boundary can continue')
        self.done=False;self.last_end_reason=None;self.life_index+=1;self.elapsed=0
        self.totals=Counter();self.raw_totals=Counter();self.discounted_totals=Counter();self.discounted_totals_valid=True
        self._reset_combat_diagnostic_totals()
        self._reset_combat_reward_scope_totals()
        self._score_gain_points=0;self._score_gain_native_frames=0
        self._score_gain_last_decision=dict(native_frames=0,valid_native_frames=0,
            quarantined_native_frames=0,score_points=None,valid_score_points=0,
            score_reward_applied=0.,zero_applied_by_policy=False,measured_zero_native_frames=0)
        return self.history.window(),self.info()

    def snapshot(self):
        fields=('tracker','history','elapsed','done','horizon','totals','raw_totals','discounted_totals','discounted_totals_valid','last_action','ram',
                'initial_frame','replay_frames','variation','training_segment','controls',
                'episodic_life','life_index','game_elapsed','last_end_reason','_combat_diagnostic_totals',
                '_combat_reward_scope_totals','_score_gain_baseline','_score_gain_points',
                '_score_gain_native_frames','_score_gain_last_decision','_score_gain_failure',
                '_score_gain_failure_details','_score_gain_quarantined','_score_gain_quarantine_reason',
                '_score_gain_quarantine_details','_score_gain_quarantined_native_frames',
                '_score_gain_quarantine_started')
        return dict(native=self.native.dump_state(),fields=copy.deepcopy({k:getattr(self,k) for k in fields}))

    def reset_recovery(self,identifier,*,horizon=None):
        """Trusted local policy-replay state, including exact past observations."""
        import pickle
        manifest_path=ROOT/self.config['policy_recovery_manifest']
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=self.config['policy_recovery_sha256']:
            raise ValueError('Recovery manifest changed')
        manifest=json.loads(manifest_path.read_text())
        if manifest['interventions'] or manifest['route_sha256']!=self.route.meta['route_sha256']:
            raise ValueError('Recovery provenance mismatch')
        entry=next(x for x in manifest['checkpoints'] if x['id']==identifier)
        blob=(ROOT/entry['path']).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=entry['sha256']:raise ValueError('Recovery checkpoint changed')
        self.restore_snapshot(pickle.loads(blob))
        # Keep native health/lives, controller history, and progress high-water
        # marks. Only the new episode's diagnostics and horizon restart.
        self.elapsed=0;self.game_elapsed=0;self.done=False;self.last_end_reason=None
        self.training_segment=True;self.episodic_life=False
        self.horizon=int(horizon or self.config['forward_training_horizon'])
        self.totals=Counter();self.raw_totals=Counter();self.discounted_totals=Counter();self.discounted_totals_valid=True
        self._reset_combat_diagnostic_totals()
        self._reset_combat_reward_scope_totals()
        self._reset_score_gain_tracking()
        self._seed_score_gain_baseline(self.ram)
        return self.history.window(),self.info()

    def restore_snapshot(self,state):
        self.native.load_state(state['native'])
        self._reset_combat_diagnostic_totals()
        self._reset_score_gain_tracking()
        self._combat_diagnostic_failure=None
        # Older checkpoints did not retain discounted episode prefixes. Keep
        # the continuation exact, but report that diagnostic total unavailable.
        self.discounted_totals=Counter();self.discounted_totals_valid=False
        self.training_segment=False
        self.controls=None
        self.episodic_life=False;self.life_index=0;self.game_elapsed=state['fields'].get('elapsed',0);self.last_end_reason=None
        for k,v in state['fields'].items():setattr(self,k,copy.deepcopy(v))
        self._combat_reward_scope_totals=copy.deepcopy(state['fields'].get(
            '_combat_reward_scope_totals',self._empty_combat_reward_scope_totals()))
        # Legacy snapshots predate quarantine state and rebase score at restored
        # RAM. New snapshots preserve quarantine exactly, including its cause.
        self._score_gain_quarantined=bool(state['fields'].get('_score_gain_quarantined',False))
        self._score_gain_quarantine_reason=copy.deepcopy(state['fields'].get('_score_gain_quarantine_reason'))
        self._score_gain_quarantine_details=copy.deepcopy(state['fields'].get('_score_gain_quarantine_details'))
        self._score_gain_quarantined_native_frames=int(state['fields'].get('_score_gain_quarantined_native_frames',0))
        self._score_gain_quarantine_started=bool(state['fields'].get('_score_gain_quarantine_started',False))
        np.testing.assert_array_equal(self.ram,self.native.ram_buffer())
        if self.score_gain_weight:
            if self._score_gain_failure is not None:
                self._fail_score_gain('cannot resume a snapshot marked with score decoder failure')
            try:current_score=decode_display_score(self.ram)
            except Exception as exc:self._fail_score_gain('restored snapshot has invalid packed-BCD score',exc)
            saved_baseline=state['fields'].get('_score_gain_baseline')
            if saved_baseline is None:
                # Legacy or weight-zero snapshots do not contain score history;
                # current restored RAM becomes a fresh baseline with no payback.
                self._score_gain_baseline=current_score
                self._score_gain_points=0;self._score_gain_native_frames=0
                self._score_gain_last_decision=dict(native_frames=0,valid_native_frames=0,
                    quarantined_native_frames=0,score_points=None,valid_score_points=0,
                    score_reward_applied=0.,zero_applied_by_policy=False,measured_zero_native_frames=0)
            elif isinstance(saved_baseline,bool) or not isinstance(saved_baseline,(int,np.integer)):
                self._fail_score_gain('snapshot score baseline is malformed')
            elif int(saved_baseline)!=current_score:
                self._fail_score_gain(f'snapshot score baseline {saved_baseline} does not match restored score {current_score}')
        return self.history.window(),self.info()

    def close(self):self.native.close()


class BackwardCurriculum:
    def __init__(self,config,valid,start,end):
        self.config=config;self.valid=np.asarray(valid);self.start=start;self.end=end
        candidates=self.valid[self.valid<=end-config['initial_remaining_frames']]
        if not len(candidates):raise ValueError('No valid initial curriculum frontier')
        self.frontier=int(candidates[-1]);self.learned=[];self.outcomes=[];self.advancements=[]
        self.step=config['backward_step_frames'];self.stage=0;self.refinements=[];self.stage_start_frames=0
        self.initial_fallback_candidates=self._validated_initial_fallback_candidates()
        self.initial_fallback_cursor=0;self.initial_fallback_exhausted=False
        self.initial_failure_frames=0

    def _validated_initial_fallback_candidates(self):
        configured=self.config.get('initial_failure_fallback_starts',[])
        candidates=[]
        for value in configured:
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)):
                raise ValueError('Initial fallback starts must be integer route frames')
            frame=int(value)
            upper=min(self.end-1,int(self.config.get('latest_action_dependent_reset',self.end-1)))
            if frame<=self.frontier or frame>upper:
                raise ValueError('Initial fallback start is outside the eligible suffix bounds')
            if frame not in self.valid:
                raise ValueError('Initial fallback start is not a verified valid route reset')
            if frame not in candidates:candidates.append(frame)
        return candidates

    def _initial_failure_fallback(self,frames,rate):
        if self.learned or self.frontier==self.start:return False
        minimum_episodes=max(self.config.get('advancement_min_episodes',0),
            self.config.get('refinement_min_episodes',0))
        minimum_frames=self.config.get('initial_fallback_min_frames',self.config.get('refinement_min_frames',0))
        if (len(self.outcomes)<minimum_episodes or rate>=self.config.get('refinement_success_floor',0.)
                or self.initial_failure_frames<minimum_frames):return False
        if self.initial_fallback_cursor>=len(self.initial_fallback_candidates):
            if not self.initial_fallback_exhausted:
                self.initial_fallback_exhausted=True
                self.advancements.append(dict(kind='initial_suffix_fallback_exhausted',frames=frames,
                    old=self.frontier,new=self.frontier,success_rate=rate,episodes=len(self.outcomes),
                    failure_frames=self.initial_failure_frames,candidates=self.initial_fallback_candidates.copy()))
                return True
            return False
        old=self.frontier;new=self.initial_fallback_candidates[self.initial_fallback_cursor]
        self.initial_fallback_cursor+=1
        episode_count=len(self.outcomes)
        self.frontier=new;self.stage+=1;self.stage_start_frames=frames;self.initial_failure_frames=0
        self.advancements.append(dict(kind='initial_verified_suffix_fallback',frames=frames,old=old,new=new,
            success_rate=rate,episodes=episode_count,failure_frames=minimum_frames,
            suffix_index=self.initial_fallback_cursor-1,remaining_candidates=len(self.initial_fallback_candidates)-self.initial_fallback_cursor))
        self.outcomes=[]
        return True

    def sample(self,rng,role=None):
        full=float(self.config.get('full_start_fraction',0.))
        coverage=float(self.config.get('coverage_fraction',0.))
        if not 0<=full<=1 or not 0<=coverage<=1-full:raise ValueError('Invalid reset mixture')
        if role not in (None,'full_start','coverage','frontier'):raise ValueError('Unknown collector role')
        mixed=bool(full+coverage or role is not None)
        if full+coverage or role in ('full_start','coverage'):
            draw=rng.random() if role is None else None
            if role=='full_start' or (role is None and draw<full):
                return dict(frame=self.start,horizon=self.config['full_horizon'],stage=self.stage,
                            frontier=self.frontier,rehearsal=True,kind='full_start',training_segment=False)
            if role=='coverage' or (role is None and draw<full+coverage):
                anchors=np.asarray(self.config['coverage_reset_frames'],dtype=int)
                if not len(anchors) or not np.isin(anchors,self.valid).all():raise ValueError('Unverified coverage reset frame')
                anchor=int(rng.choice(anchors))
                candidates=self.valid[(self.valid>=anchor)&(self.valid<=anchor+self.config['frontier_jitter_frames'])]
                return dict(frame=int(rng.choice(candidates)),stage=self.stage,frontier=self.frontier,
                            rehearsal=True,kind='coverage',training_segment=anchor!=self.start)
        rehearsal=bool(self.learned and rng.random()<self.config['rehearsal_fraction'])
        anchor=int(rng.choice(self.learned)) if rehearsal else self.frontier
        jitter=self.config['frontier_jitter_frames']
        if not rehearsal and self.config.get('adaptive_refinement',False):
            later=[x for x in self.learned if x>anchor]
            if later:jitter=min(jitter,min(later)-anchor-1)
        candidates=self.valid[(self.valid>=anchor)&(self.valid<=anchor+max(0,jitter))]
        result=dict(frame=int(rng.choice(candidates)),stage=self.stage,frontier=self.frontier,rehearsal=rehearsal)
        if mixed:result.update(kind='rehearsal' if rehearsal else 'frontier',training_segment=True)
        return result

    def observe(self,case,success,frames,native_frames=0):
        if case['rehearsal'] or case['stage']!=self.stage:return False
        self.outcomes.append(bool(success));self.outcomes=self.outcomes[-self.config['advancement_window']:]
        rate=float(np.mean(self.outcomes))
        if not success and not self.learned:self.initial_failure_frames+=int(native_frames)
        if self.config.get('initial_failure_fallback_starts') and self._initial_failure_fallback(frames,rate):return True
        if (self.config.get('adaptive_refinement',False) and len(self.outcomes)>=self.config.get('refinement_min_episodes',40)
                and rate<self.config.get('refinement_success_floor',.05)
                and frames-self.stage_start_frames>=self.config.get('refinement_min_frames',50000)):
            later=[x for x in self.learned if x>self.frontier]
            if later:
                upper=min(later);gap=upper-self.frontier;fine=max(self.config['minimum_backward_step_frames'],gap//2)
                if fine<gap:
                    eligible=self.valid[(self.valid>self.frontier)&(self.valid<=upper-fine)]
                    if len(eligible):
                        old=self.frontier;self.frontier=int(eligible[-1]);self.stage+=1
                        self.refinements.append((old,upper,fine))
                        self.advancements.append(dict(kind='refine_easier',frames=frames,old=old,new=self.frontier,
                            success_rate=rate,episodes=len(self.outcomes),fine_step=fine))
                        self.outcomes=[];self.stage_start_frames=frames
                        return True
        if len(self.outcomes)<self.config['advancement_min_episodes'] or rate<self.config['advancement_threshold'] or self.frontier==self.start:return False
        old=self.frontier;step=self.step
        for low,high,fine in self.config.get('fine_step_regions',[])+self.refinements:
            if low<=old<=high:step=min(step,fine)
        target=max(self.start,old-step)
        new=int(self.valid[self.valid<=target][-1])
        self.learned.append(old);self.frontier=new;self.stage+=1
        self.advancements.append(dict(kind='advance_earlier',frames=frames,old=old,new=new,success_rate=rate,episodes=len(self.outcomes)))
        self.outcomes=[];self.stage_start_frames=frames
        return True

    def state(self):
        return copy.deepcopy({k:getattr(self,k) for k in ('frontier','learned','outcomes','advancements','step','stage','refinements','stage_start_frames',
            'initial_fallback_candidates','initial_fallback_cursor','initial_fallback_exhausted','initial_failure_frames')})

    def load(self,state):
        for k,v in state.items():setattr(self,k,copy.deepcopy(v))
        # Legacy checkpoints have no initial fallback state. Reconstruct from
        # their saved frontier and do not retroactively apply a new config.
        if 'initial_fallback_candidates' not in state:
            self.initial_fallback_candidates=[];self.initial_fallback_cursor=0
            self.initial_fallback_exhausted=False;self.initial_failure_frames=0
        if (self.frontier not in self.valid or any(x not in self.valid for x in self.learned)
                or self.initial_fallback_cursor<0 or self.initial_fallback_cursor>len(self.initial_fallback_candidates)):
            raise ValueError('Saved curriculum state is incompatible with the verified route')
        if 'stage_start_frames' not in state and self.advancements:self.stage_start_frames=self.advancements[-1]['frames']


def level1_child(pipe,configuration,offsets):
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    env=None;stashed=None
    try:
        env=Level1Env(configuration)
        env._score_gain_worker_pid=os.getpid()
        pipe.send(dict(ready=True,pid=os.getpid(),torch_imported='torch' in sys.modules))
        while True:
            command,payload=pipe.recv()
            if command=='close':break
            try:
                if command=='reset':
                    if payload.get('recovery_checkpoint'):
                        o,i=env.reset_recovery(payload['recovery_checkpoint'],horizon=payload.get('horizon'))
                    else:
                        o,i=env.reset(payload.get('frame'),horizon=payload.get('horizon'),variation=payload.get('variation',0),
                                     training_segment=payload.get('training_segment',False),episodic_life=payload.get('episodic_life',False))
                    result=dict(obs=o,info=i,mask=env.allowed.copy())
                elif command=='continue_life':
                    o,i=env.continue_life();result=dict(obs=o,info=i,mask=env.allowed.copy())
                elif command=='step':
                    o,r,t,u,i=env.step(int(payload));result=dict(obs=o,reward=r,terminated=t,truncated=u,info=i)
                elif command=='stash':stashed=env.snapshot();result=dict(stashed=True)
                elif command=='restore':
                    o,i=env.restore_snapshot(stashed);stashed=None;result=dict(obs=o,info=i,mask=env.allowed.copy())
                elif command=='snapshot':result=env.snapshot()
                elif command=='load_snapshot':
                    o,i=env.restore_snapshot(payload);result=dict(obs=o,info=i,mask=env.allowed.copy())
                elif command=='capture_score_clear':
                    from .score_clear_capture import ScoreClearRing
                    env._score_clear_capture=ScoreClearRing(payload['worker_id'],payload['output_dir'],
                        capacity=payload.get('capacity',33),provenance=payload.get('provenance'))
                    env._score_clear_capture.begin_episode(payload.get('episode_id',f'worker_{payload["worker_id"]}'))
                    env._score_clear_native_frame_index=int(payload.get('native_frame_index_base',0))
                    env._score_clear_reset_provenance=dict(payload.get('reset_provenance',{}))
                    result=dict(capture_ready=True,worker_id=int(payload['worker_id']))
                elif command=='capture_episode':
                    capture=getattr(env,'_score_clear_capture',None)
                    if capture is not None:
                        capture.begin_episode(payload['episode_id'])
                        env._score_clear_native_frame_index=int(payload.get('native_frame_index_base',
                            getattr(env,'_score_clear_native_frame_index',0)))
                        if 'reset_provenance' in payload:
                            env._score_clear_reset_provenance=dict(payload['reset_provenance'])
                    result=dict(capture_episode=payload['episode_id'])
                elif command=='dump_score_clear':
                    capture=getattr(env,'_score_clear_capture',None)
                    result=dict(capture_path=(capture.persist(error=payload) if capture is not None else None),
                                transitions=(len(capture.transitions) if capture is not None else 0))
                elif command=='screen':result=env.native.copy_screen_rgb()
                else:raise ValueError('Unknown level1 command')
                pipe.send(result)
            except Exception:pipe.send(dict(error=traceback.format_exc()))
    except (EOFError,BrokenPipeError):pass
    except Exception:
        try:pipe.send(dict(error=traceback.format_exc()))
        except (EOFError,BrokenPipeError):pass
    finally:
        if env:env.close()
        pipe.close()
