"""Level2 score-delta reward v4, preserving the v3 baseline as history.

Directions control the aim, not rewarded navigation progress. Score gains are
the game's own combat-progress proxy, never an inferred count of enemy kills.
"""
from .level2_runtime import checked_ram,transition as baseline_transition,WEIGHTS as BASE_WEIGHTS

REWARD_VERSION='level2_score_delta_v4'
SCORE_PER_POINT=.01
WEIGHTS={**BASE_WEIGHTS,'score':SCORE_PER_POINT}

class ScoreDiscontinuity(ValueError):
    """A reset/rollover must not be turned into a fabricated positive award."""

def decode_score(ram):
    row=checked_ram(ram)
    if row.ndim!=1:raise ValueError('Single RAM row required')
    score=0
    for byte in row[0xF4:0xF7]:
        high,low=int(byte)>>4,int(byte)&15
        if high>9 or low>9:raise ValueError('Invalid packed-BCD Level2 score')
        score=score*100+high*10+low
    return score

def transition(pre,post,*,level2_scope):
    result=baseline_transition(pre,post,level2_scope=level2_scope)
    before=decode_score(pre);after=decode_score(post)
    gain=after-before
    if gain<0:
        raise ScoreDiscontinuity('Score decreased: reset/rollover outside verified continuous Level2 transition scope')
    result['components']={**result['components'],'score':SCORE_PER_POINT*gain}
    result['reward']=sum(result['components'].values())
    result.update(reward_version=REWARD_VERSION,score_before=before,score_after=after,score_gain=gain,
                  score_available=True,score_is_kill_count=False)
    return result
