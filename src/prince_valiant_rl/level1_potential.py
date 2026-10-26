"""Optional, versioned final-encounter shaping; disabled in the baseline task.

Only slot0 raw kind14/phase1 has an executed remaining-arrow-hit interpretation.
This is a bounded state potential, not a generic enemy damage/kill detector.
All inputs already exist in the structured observation. No historical tracker
or demonstration position is used.
"""
from .navigation import map_section, supported


VERSION='level1_type14_potential_v1'


def final_encounter_potential(ram):
    if not supported(ram) or map_section(ram)!=2:return 0.
    if int(ram[0x204])==1:return 1.
    if int(ram[0xC1])==14 and int(ram[0x99])==1:
        remaining=int(ram[0xB7])
        if 1<=remaining<=24:return (24-remaining)/24
    # Zero here defines the shaping function outside its verified subgoal.
    # It does NOT assert absent/dead/zero-HP semantics for other objects.
    return 0.


def potential_difference(before,after,gamma,*,bootstrap_valid=True):
    """Native-frame F=gamma*Phi(next)-Phi(current).

    True terminals and unreliable unsupported boundaries use zero endpoint
    potential. A valid time-limit continuation keeps its actual potential.
    With correct continuation values, discounted sums telescope; restoring an
    enemy or losing subgoal progress reclaims the earlier shaping credit.
    """
    following=final_encounter_potential(after) if bootstrap_valid else 0.
    return gamma*following-final_encounter_potential(before)
