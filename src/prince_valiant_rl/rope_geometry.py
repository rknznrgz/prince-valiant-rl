"""ROM-derived rope attachment positions, diagnostic schema v1.

Verified bank7 CPU EA16..EA9D uses table EA9E..EB1D. These are
attachment-segment positions at the sampled object phase, NOT sprite bounds or
universal collision hitboxes. Object animation can update after player position;
the selected point is therefore not always the current player's coordinates.
The optional v4 observation wires these points into a separate terrain-aligned
grid; legacy v3 observations and checkpoints remain unchanged.
"""
import hashlib
from pathlib import Path
from .game_state import ROM_SHA256,word
from .navigation import supported,live

VERSION='forest_rope_attachment_geometry_v1'
TABLE_FILE_OFFSET=0x1EAAE

class RopeGeometry:
    def __init__(self,rom):
        data=Path(rom).read_bytes()
        if hashlib.sha256(data).hexdigest()!=ROM_SHA256:raise ValueError('Unverified rope ROM')
        self.table=data[TABLE_FILE_OFFSET:TABLE_FILE_OFFSET+128]

    def decode(self,ram):
        scope=bool(supported(ram));ropes=[]
        slot_raw=int(ram[0x2F2]);segment_raw=int(ram[0x2F3]);state_raw=int(ram[0x5F])
        for slot in range(10):
            if not scope or int(ram[0xC1+slot])!=6:continue
            animation=int(ram[0xDF+slot]);anchor_x=word(ram,0x28C+2*slot);anchor_y=word(ram,0x2A0+2*slot)
            geometry_valid=39<=animation<=46 and anchor_y<256
            points=None
            if geometry_valid:
                points=[]
                for depth in range(8):
                    index=2*((animation-39)*8+depth);dx=self.table[index];dx=dx if dx<128 else dx-256
                    x=(anchor_x+(dx if ram[0xD5+slot] else -dx))&65535
                    y=(anchor_y-48+self.table[index+1])&255
                    points.append(dict(segment=depth,world_x=x,y=y,screen_x=x-word(ram,0x6E)))
            ropes.append(dict(slot=slot,kind_raw=6,resident_id_raw=int(ram[0xE9+slot]),
                animation_raw=animation,direction_raw=int(ram[0xD5+slot]),flags_raw=int(ram[0xCB+slot]),
                anchor_world_x=anchor_x,anchor_y=anchor_y,geometry_valid=geometry_valid,
                attachment_points=points,endpoint=points[7] if points is not None else None))
        selected=next((r for r in ropes if r['slot']==slot_raw and r['geometry_valid']),None)
        attached=bool(scope and live(ram) and state_raw==23 and selected is not None and 0<=segment_raw<8)
        return dict(schema=VERSION,scope_valid=scope,ropes=ropes,attachment_valid=attached,
            attached_slot=slot_raw if attached else None,attached_segment=segment_raw if attached else None,
            selected_attachment_point=selected['attachment_points'][segment_raw] if attached else None,
            availability=dict(attachment_geometry=scope,exact_collision_hitboxes=False),
            limitations='Snapshot-phase attachment points may lead player coordinates by one native update. Catch rectangles and other object hitboxes are not supplied.')
