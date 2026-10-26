"""Conservative frame-end signature from ordinary_enemy_kind18_20261021.

Counts are scoped diagnostics, never generic HP loss or universal kill coverage.
The caller supplies contiguous native frames of the verified ROM in Level1.
No CPU trace, collision box, or exact projectile-to-target attribution is available.
"""
import numpy as np
from .game_state import word, signed_byte

KEYS = ('kind18_accepted_hits', 'kind18_kills')
COVERAGE = dict(version='level1_kind18_projectile2_signature_v1',
    units=dict(kind18_accepted_hits='accepted lethal hit signatures', kind18_kills='defeated enemy signatures'),
    scope='Level1 forest section0; kind18 counter1 -> kind7 timer24; projectile2 only',
    attribution='frame-end corroboration only; exact handler target unavailable',
    runtime_evidence='frame-end RAM only; no execution trace or collision attribution',
    complete=False,
    uncovered=['other types', 'other sections', 'counter variants', 'surviving hits',
               'other projectile slots', 'simultaneous impacts', 'other kill sources',
               'unobserved same-frame slot reuse'])


def scene(r):
    return (int(r[0x33]) == 8 and int(r[0x4D]) == 2 and int(r[0x2F4]) == 0
            and word(r, 0x50E) == 0x9F81 and int(r[0x60]) == 0
            and signed_byte(r[0xF7]) >= 0)


class Kind18Events:
    def __init__(self):
        self.previous = None
        self.consumed = set()

    def seed(self, ram):
        """Establish a baseline on reset/prehistory; emit nothing."""
        self.previous = ram.copy()
        self.consumed.clear()

    def transition(self, old, new):
        continuous = self.previous is not None and np.array_equal(old, self.previous)
        self.previous = new.copy()
        valid = (continuous and scene(old) and scene(new)
                 and old[0xF7] == new[0xF7])
        if not valid:
            # A discontinuity is a new baseline, not evidence of an enemy edge.
            self.consumed.clear()
            return dict.fromkeys(KEYS)
        result = dict.fromkeys(KEYS, 0)
        kinds_old, kinds_new = old[0xC1:0xCB], new[0xC1:0xCB]
        for slot in range(10):
            if int(kinds_old[slot]) == 0 and int(kinds_new[slot]) == 18:
                self.consumed.discard(slot)
        candidates = [s for s in range(10) if int(kinds_old[s]) == 18 and int(kinds_new[s]) == 7]
        # Only the witnessed projectile2 signature, with no competing flying
        # projectile or other kind-changing target. This is corroboration, not
        # a reconstruction of the internal collision test or selected slot.
        projectile = (int(old[0x50D]) == 1 and int(new[0x50D]) == 2
                      and int(new[0x50C]) == 21
                      and all(int(r[a]) == 0 for r in (old, new) for a in (0x509, 0x50B)))
        # Consume even an uncorroborated defeat edge: later observations cannot
        # retroactively establish the projectile attribution for this lifetime.
        eligible = [s for s in candidates if s not in self.consumed]
        self.consumed.update(candidates)
        if len(candidates) != 1:
            return result
        slot = candidates[0]
        stable_others = all(kinds_old[s] == kinds_new[s] for s in range(10) if s != slot)
        signature = (int(old[0xB7+slot]) == 1 and int(new[0xB7+slot]) == 24
                     and int(old[0x99+slot]) == int(new[0x99+slot]) == 0
                     and int(old[0xCB+slot]) == int(new[0xCB+slot]) == 7
                     and word(old, 0x2A0+2*slot) == word(new, 0x2A0+2*slot)
                     and abs(word(new, 0x28C+2*slot)-word(old, 0x28C+2*slot)) <= 1)
        if slot in eligible and stable_others and projectile and signature:
            result = dict.fromkeys(KEYS, 1)
        return result
