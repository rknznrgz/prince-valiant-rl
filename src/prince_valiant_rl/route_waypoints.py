"""Training-defined demonstrated coordinate goals, not native checkpoint events.

Each goal can pay only at its first forward crossing, using the existing
observable per-section maximum X. No new hidden reward ledger is introduced.
"""
import hashlib,json
from pathlib import Path
from .game_state import word,signed_byte
from .navigation import map_section,live,supported,WIDTHS
from .paths import ROOT

VERSION='level1_demonstrated_crossing_goals_v1'


def demonstrated_goals(route,spacing=256):
    if spacing<64:raise ValueError('Avoid excessively dense redundant goals')
    goals=[];maximum=[-1]*3;targets=[spacing]*3
    for frame in range(route.start+1,route.end):
        a,b=route.ram[frame-1],route.ram[frame]
        if not live(b):continue
        section=map_section(b);x=word(b,0x56);old=maximum[section];maximum[section]=max(old,x)
        if (x<=old or x<targets[section] or x>=WIDTHS[section]-32 or not live(a) or map_section(a)!=section or a[0xF7]!=b[0xF7]):continue
        if (int(b[0x5F])!=0 or signed_byte(b[0x66])!=0 or word(a,0x58)!=word(b,0x58) or not 0<word(b,0x56)-word(a,0x56)<=4):continue
        goals.append(dict(section=section,x=x,y=word(b,0x58),frame=frame,required_player_state_raw=0))
        targets[section]=x+spacing
    return goals


class RouteWaypoints:
    def __init__(self,c):
        if c.get('route_waypoint_version')!=VERSION:raise ValueError('Explicit route-goal version required')
        path=Path(c['route_waypoint_data']);path=path if path.is_absolute() else ROOT/path
        if hashlib.sha256(path.read_bytes()).hexdigest()!=c['route_waypoint_sha256']:raise ValueError('Route-goal digest mismatch')
        payload=json.loads(path.read_text());self.goals=payload['goals'];self.tolerance=4
        for g in self.goals:
            if g['section'] not in (0,1,2) or not 0<=g['x']<WIDTHS[g['section']] or not 0<=g['y']<256:raise ValueError('Invalid route goal')

    def transition(self,a,b,previous_maxima):
        if not supported(a) or not supported(b):return None
        if not live(a) or not live(b) or int(b[0xFC])==0:return 0.
        section=map_section(b)
        if map_section(a)!=section or a[0xF7]!=b[0xF7]:return 0.
        x,y=word(b,0x56),word(b,0x58);dx=x-word(a,0x56)
        if not 0<dx<=4 or abs(y-word(a,0x58))>4 or int(b[0x5F])!=0 or signed_byte(b[0x66])!=0:return 0.
        previous=max(previous_maxima[section],word(a,0x56))
        return float(sum(g['section']==section and previous<g['x']<=x and abs(y-g['y'])<=self.tolerance for g in self.goals))
