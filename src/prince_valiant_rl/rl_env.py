"""Minimal Gymnasium RAM environment, scoped to tested forest checkpoints."""
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from nes_py._native import NativeEmulator

from prince_valiant_rl.game_state import ROM_SHA256, extract_game_state, signed_byte
from prince_valiant_rl.scene_state import ForestTerrain

from .paths import ROOT
# Discrete action indices; masks are held for every repeated frame.
ACTIONS = (0, 128, 64, 32, 16, 2, 1, 129, 130, 65)
ACTION_NAMES = ('NOOP', 'RIGHT', 'LEFT', 'DOWN', 'UP', 'B', 'A', 'RIGHT+A', 'RIGHT+B', 'LEFT+A')


class PrinceValiantRamEnv(gym.Env):
    """Coordinates/steps are pixels, counters are raw integer units.

    Observation is a nested Dict. Optional semantics use -1 plus validity
    masks in the observation, and None in info['state']. Default reset is the
    established initial gameplay save; no pixel interpretation is performed.
    The native PPU still runs, but screen access happens only in render().
    """
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}

    def __init__(self, rom=ROOT / 'roms/prince_valiant.nes',
                 checkpoint=ROOT / 'savestates/gameplay.json', frame_skip=1,
                 max_episode_frames=6000, render_mode=None, action_masks=None):
        super().__init__()
        self.rom, self.checkpoint = Path(rom), Path(checkpoint)
        if hashlib.sha256(self.rom.read_bytes()).hexdigest() != ROM_SHA256:
            raise ValueError('Unsupported ROM hash')
        if not isinstance(frame_skip, int) or frame_skip < 1:
            raise ValueError('frame_skip must be a positive integer')
        if not isinstance(max_episode_frames, int) or max_episode_frames < 1:
            raise ValueError('max_episode_frames must be a positive integer')
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('render_mode must be None or rgb_array')
        self.frame_skip, self.max_episode_frames = frame_skip, max_episode_frames
        self.render_mode = render_mode
        self.terrain_decoder = ForestTerrain(self.rom)
        self.action_masks = tuple(ACTIONS if action_masks is None else action_masks)
        if not self.action_masks or any(not isinstance(a, int) or not 0 <= a <= 255 for a in self.action_masks):
            raise ValueError('action_masks must contain valid held-button bytes')
        self.action_space = spaces.Discrete(len(self.action_masks))
        integer = lambda shape: spaces.Box(-65536, 65535, shape=shape, dtype=np.int32)
        self.observation_space = spaces.Dict({
            'player': spaces.Dict({'position': integer((2,)), 'movement': integer((5,)),
                                   'velocity_derived': integer((2,)),
                                   'velocity_valid': spaces.MultiBinary(2),
                                   'retries_remaining': integer((1,)), 'health': integer((1,)),
                                   'valid': spaces.MultiBinary(3)}),
            'camera': integer((2,)),
            'entities': integer((10, 5)),
            'entity_masks': spaces.MultiBinary((10, 2)),
            'actor_details': integer((10, 8)),
            'actor_hidden': spaces.MultiBinary(10),
            'actor_sprite_bounds': integer((10, 4)),
            'actor_sprite_bounds_valid': spaces.MultiBinary(10),
            'player_projectiles': integer((3, 7)),
            'projectile_masks': spaces.MultiBinary((3, 2)),
            'hud': integer((8,)),
            'hud_valid': spaces.MultiBinary(1),
            'damage_protection_raw': integer((2,)),
            'scene_identity_raw': integer((4,)),
            'raw_ram': spaces.Box(0, 255, shape=(2048,), dtype=np.uint8),
            'gameplay_dispatch_valid': spaces.MultiBinary(1),
            'terrain': spaces.Dict({'origin': integer((2,)),
                                   'tile_ids': integer((12, 17)),
                                   'collision_classes': integer((12, 17)),
                                   'valid': spaces.MultiBinary((12, 17))}),
            'nearby_terrain': spaces.Dict({'origin': integer((2,)),
                                          'player_offset': integer((2,)),
                                          'map_shape': integer((2,)),
                                          'tile_ids': integer((12, 17)),
                                          'collision_classes': integer((12, 17)),
                                          'valid': spaces.MultiBinary((12, 17))}),
            'triggers': spaces.Dict({'rectangles': integer((16, 4)),
                                    'action_arguments_raw': integer((16, 5)),
                                    'valid': spaces.MultiBinary(16)}),
            'terminal': spaces.MultiBinary(2),
            'unknown': spaces.Dict({'lives_level_completion': integer((3,)),
                                    'valid': spaces.MultiBinary(3)}),
        })
        self.native = None
        self._done = True
        # Optional scene interventions run at native-frame frequency, before
        # observation extraction. Default gameplay has no hooks.
        self.before_native_frame = None
        self.after_native_frame = None

    def _observe(self, previous=None):
        ram = self.native.ram_buffer().copy()
        state = extract_game_state(ram, previous, terrain_decoder=self.terrain_decoder)
        p, c = state['player'], state['camera']
        obs = dict(
            player=dict(position=np.array([p['world_x'], p['y']], np.int32),
                        velocity_derived=np.array([v if v is not None else 0 for v in
                                                   (p['velocity_x_derived'], p['velocity_y_derived'])], np.int32),
                        velocity_valid=np.array([p['velocity_x_derived'] is not None,
                                                 p['velocity_y_derived'] is not None], np.int8),
                        movement=np.array([p['facing_left'], p['movement_state'], p['state_frame'],
                                           p['vertical_step'], p['horizontal_step_magnitude']], np.int32),
                        retries_remaining=np.array([p['retries_remaining'] if p['retries_remaining'] is not None else -1], np.int32),
                        health=np.array([p['health'] if p['health'] is not None else -1],np.int32),
                        valid=np.array([state['validity']['player_position_valid'], p['retries_remaining'] is not None,
                                        p['health'] is not None], np.int8)),
            camera=np.array([c['world_x'], c['scroll_x_copy']], np.int32),
            entities=np.array([[e['world_x'],e['y'],e['kind_code'],e['flags_raw'],e['screen_anchor_x']]
                               for e in state['entities']], np.int32),
            entity_masks=np.array([[e['occupied'], e['render_eligible']] for e in state['entities']], np.int8),
            actor_details=np.array([[e['phase_raw'], e['direction_raw'], e['animation_raw'],
                                     *e['arguments_raw'], e['motion_or_timer_raw'], int(e['hidden'])]
                                    for e in state['entities']], np.int32),
            actor_hidden=np.array([e['hidden'] for e in state['entities']], np.int8),
            actor_sprite_bounds=np.array([e['sprite_bounds_screen'] if e['sprite_bounds_screen'] is not None
                                           else [-1]*4 for e in state['entities']], np.int32),
            actor_sprite_bounds_valid=np.array([e['sprite_bounds_screen'] is not None
                                                 for e in state['entities']], np.int8),
            player_projectiles=np.array([[e['world_x'], e['y'], e['direction_raw'], e['state_raw'],
                                          e['animation_raw'], e['screen_anchor_x'], e['slot']]
                                         for e in state['player_projectiles']], np.int32),
            projectile_masks=np.array([[e['active'], e['flying']] for e in state['player_projectiles']], np.int8),
            hud=np.array([*state['hud']['score_digits'], *state['hud']['timer_digits'],
                          state['hud']['timer_subcounter_raw']], np.int32),
            hud_valid=np.array([state['hud']['valid']], np.int8),
            damage_protection_raw=np.array(list(state['protection'].values()), np.int32),
            scene_identity_raw=np.array(state['scene_identity_raw'], np.int32),
            raw_ram=ram.copy(),
            gameplay_dispatch_valid=np.array([state['validity']['gameplay_dispatch_matches']], np.int8),
            terrain={k: state['terrain'][k] for k in ('origin', 'tile_ids', 'collision_classes', 'valid')},
            nearby_terrain={k: state['nearby_terrain'][k] for k in
                            ('origin', 'player_offset', 'map_shape', 'tile_ids', 'collision_classes', 'valid')},
            triggers=state['triggers'],
            terminal=np.array([state['terminal']['life_sequence_active'],state['terminal']['game_over']], np.int8),
            unknown=dict(lives_level_completion=np.full(3,-1,np.int32), valid=np.zeros(3,np.int8)),
        )
        return obs, state, ram

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        payload = json.loads(self.checkpoint.read_text(encoding='utf-8'))
        if payload.get('version') != 1 or payload.get('rom_sha256') != ROM_SHA256:
            raise ValueError('Checkpoint format or ROM does not match')
        # Fresh native instance avoids the known reset/cross-instance snapshot issue.
        candidate = NativeEmulator(str(self.rom))
        try:
            candidate.reset()
            replay_observer = (options or {}).get('replay_observer')
            if replay_observer is not None:
                replay_observer(candidate.ram_buffer().copy())
            for mask, frames in payload['actions']:
                if not isinstance(mask,int) or not 0 <= mask <= 255 or not isinstance(frames,int) or frames < 1:
                    raise ValueError('Invalid checkpoint actions')
                for _ in range(frames):
                    candidate.frame_advance(mask)
                    if replay_observer is not None:
                        replay_observer(candidate.ram_buffer().copy())
            digest = hashlib.sha256(candidate.ram_buffer().tobytes()).hexdigest()
            if digest != payload['fingerprint']['ram']:
                raise ValueError('Checkpoint RAM hash differs')
            state = extract_game_state(candidate.ram_buffer().copy())
            if not state['validity']['player_position_valid'] or state['terminal']['life_sequence_active']:
                raise ValueError('Checkpoint must be a live, tested forest gameplay position')
        except Exception:
            candidate.close()
            raise
        if self.native is not None:
            self.native.close()
        self.native = candidate
        self.elapsed_frames, self._done = 0, False
        obs, state, self._ram = self._observe()
        self.frontier = state['player']['world_x']
        return obs, dict(state=state, elapsed_frames=0, checkpoint=str(self.checkpoint),
                         checkpoint_replay_frames=sum(frames for _,frames in payload['actions']),
                         reward_semantics='new world-X frontier minus 10 per retry lost; training choice')

    def step(self, action):
        if self._done:
            raise RuntimeError('Call reset before stepping a finished environment')
        if not self.action_space.contains(action):
            raise ValueError('Invalid discrete action')
        reward = 0.0
        terminated = truncated = False
        for _ in range(self.frame_skip):
            previous = self._ram
            if self.before_native_frame is not None:
                self.before_native_frame(self.native.ram_buffer())
            self.native.frame_advance(self.action_masks[int(action)])
            if self.after_native_frame is not None:
                self.after_native_frame(self.native.ram_buffer())
            self.elapsed_frames += 1
            obs, state, self._ram = self._observe(previous)
            p = state['player']
            if state['validity']['player_position_valid'] and not state['terminal']['life_sequence_active']:
                frontier = max(self.frontier, p['world_x'])
                reward += frontier - self.frontier
                self.frontier = frontier
            retries_lost = signed_byte(previous[0xF7]) - signed_byte(self._ram[0xF7])
            if retries_lost > 0:
                reward -= 10.0 * retries_lost
            terminated = state['terminal']['game_over']
            # Unsupported gameplay modes are an observation-scope limit,
            # not evidence that the game has ended or the level completed.
            scope_exit = not state['validity']['gameplay_dispatch_matches']
            truncated = not terminated and (self.elapsed_frames >= self.max_episode_frames or scope_exit)
            if terminated or truncated:
                break
        self._done = terminated or truncated
        reason = 'game_over' if terminated else ('scope_exit' if scope_exit else 'time_limit' if truncated else None)
        return obs, float(reward), bool(terminated), bool(truncated), dict(
            state=state, elapsed_frames=self.elapsed_frames, end_reason=reason)

    def render(self):
        if self.native is None:
            return None
        return self.native.copy_screen_rgb()

    def close(self):
        if self.native is not None:
            self.native.close()
            self.native = None
        self._done = True
