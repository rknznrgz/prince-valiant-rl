"""Conservative transition events. None means detector unavailable, not zero.

Map identities are ROM row-table identities observed in the three forest maps.
No object identity, collision rectangle, universal HP, or completion is inferred.
"""
from dataclasses import dataclass, field
import numpy as np
from ..game_state import signed_byte, forest_dispatch_valid

MAP_ROWS = (0x9F81, 0xABB1, 0xBA40)
MAP_WIDTHS = (112,112,80)  # decoded row strides, in sixteen-pixel cells
FPS = 60.0  # emulator-frame convention; embedded PAL text does not establish PAL timing

def word(r, a):
    return int(r[a]) + 256 * int(r[a + 1])

def section(r):
    if not forest_dispatch_valid(r) or int(r[0x4D]) != 2:
        return None
    row = word(r, 0x50E)
    return MAP_ROWS.index(row) if row in MAP_ROWS else None

def live_position(r):
    s=section(r)
    # The third row table switches one native frame BEFORE X resets1784->24.
    # Do not seed its frontier with an out-of-map coordinate from the old scene.
    return s is not None and int(r[0x60]) == 0 and 0 <= word(r,0x56) < MAP_WIDTHS[s]*16 and word(r,0x58) != 65535

@dataclass
class EventTracker:
    frontiers: list = field(default_factory=lambda: [-1, -1, -1])
    visited: set = field(default_factory=set)
    cleared: set = field(default_factory=set)

    def seed(self, r):
        """Consume save prehistory without paying rewards; retains all visited routes."""
        s = section(r)
        if live_position(r):
            self.frontiers[s] = max(self.frontiers[s], word(r, 0x56))
            self.visited.add(s)
        if s == 2:
            self.cleared.update(a for a in (0x202, 0x203) if int(r[a]) == 1)

    def observable(self):
        return np.asarray(self.frontiers + [int(i in self.visited) for i in range(3)] +
                          [int(a in self.cleared) for a in (0x202, 0x203)], np.float32)

    def transition(self, old, new, frames=1):
        a, b = section(old), section(new)
        supported = a is not None and b is not None
        e = dict(new_progress=None, player_damage=None, lives_lost=None,
                 elapsed_seconds=frames / FPS, checkpoint_reached=None,
                 level_completed=None, game_over=None, enemy_kills=None,
                 enemy_damage=None, escape_success=None, encounter_cleared=None, death_started=None)
        if supported:
            lost = max(0, signed_byte(old[0xF7]) - signed_byte(new[0xF7]))
            e['lives_lost'] = lost
            e['death_started'] = int(int(old[0x60]) == 0 and int(new[0x60]) == 1)
            # Positive restoration is never rewarded; same-life losses are normalized HP.
            hp_lost = max(0, int(old[0xFC]) - int(new[0xFC]))
            e['player_damage'] = hp_lost / 255 if not lost and int(old[0x60]) == 0 else 0.0
            # Fatal health-to-zero writes also occur on falling: cause is ambiguous.
            # Charge verified life loss later, rather than inventing damage attribution.
            if hp_lost and int(old[0x60]) == 0 and int(new[0x60]) != 0:
                e['player_damage'] = None
            e['game_over'] = int(int(new[0x60]) == 255 and signed_byte(new[0xF7]) < 0
                                 and not (int(old[0x60]) == 255 and signed_byte(old[0xF7]) < 0))
            e['new_progress'] = 0.0
            e['checkpoint_reached'] = 0
            if live_position(new):
                x = word(new, 0x56)
                # A map's initial spawn coordinate is a baseline, not earned distance.
                if self.frontiers[b] >= 0:
                    e['new_progress'] = max(0, x - self.frontiers[b]) / 16
                if b not in self.visited and b in (1, 2):
                    e['checkpoint_reached'] = 1
                self.frontiers[b] = max(self.frontiers[b], x)
                self.visited.add(b)
            if a == b == 2:
                e['encounter_cleared'] = 0
                for addr in (0x202, 0x203):
                    if int(old[addr]) == 0 and int(new[addr]) == 1 and addr not in self.cleared:
                        e['encounter_cleared'] += 1
                        self.cleared.add(addr)
                # Executed bank2 $8D23 DEC $B7,X: type14, phase1 only.
                slots = [i for i in range(10) if int(old[0xC1+i]) == int(new[0xC1+i]) == 14
                         and int(old[0x99+i]) == int(new[0x99+i]) == 1]
                if slots:
                    e['enemy_damage'] = sum(1 for i in slots if 0 <= int(new[0xB7+i]) < int(old[0xB7+i]) <= 24
                                            and int(old[0xB7+i]) - int(new[0xB7+i]) == 1)
        return e

DEFAULT_WEIGHTS = {
    'progress': dict(new_progress=1, checkpoint_reached=20, level_completed=0,
                     player_damage=-20, lives_lost=-150, elapsed_seconds=-0.1),
    'evade': dict(escape_success=0, player_damage=-20, lives_lost=-150, elapsed_seconds=-0.1),
    'fight': dict(enemy_kills=0, enemy_damage=1, encounter_cleared=20,
                  player_damage=-20, lives_lost=-150, elapsed_seconds=-0.1),
    'manager': dict(new_progress=1, checkpoint_reached=20, level_completed=0,
                    player_damage=-20, lives_lost=-150, elapsed_seconds=-0.1, game_over=-150),
}

def reward_components(events, weights):
    components = {key: dict(available=events[key] is not None, unweighted=events[key],
                           weight=float(value), weighted=float(value * events[key]) if events[key] is not None else None)
                  for key, value in weights.items()}
    return sum(v['weighted'] for v in components.values() if v['weighted'] is not None), components

def bootstrap_allowed(terminated, truncated, reason):
    return not terminated and reason != 'scope_exit'

def duration_gae(rewards, values, next_values, durations, bootstrap, continuation, gamma=.999, lam=.95):
    """SMDP GAE: rewards already contain the base-step discounted option sum.

    Bootstrap is independent of recursion across collector/episode boundaries.
    lambda is per base decision, so both gamma and lambda use actual duration.
    """
    out = np.zeros(len(rewards), np.float32)
    tail = 0.0
    for i in range(len(out)-1, -1, -1):
        discount = gamma ** int(durations[i])
        delta = rewards[i] + discount * bootstrap[i] * next_values[i] - values[i]
        tail = delta + discount * lam ** int(durations[i]) * continuation[i] * tail
        out[i] = tail
    return out, out + np.asarray(values)
