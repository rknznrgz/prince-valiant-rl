"""ROM-backed forest terrain. Uses collision data, never screen pixels."""
import hashlib
from pathlib import Path
import numpy as np
from prince_valiant_rl.game_state import ROM_SHA256, word, forest_dispatch_valid


class ForestTerrain:
    def __init__(self, rom, dispatch_predicate=None):
        self.dispatch_predicate = dispatch_predicate or forest_dispatch_valid
        data = Path(rom).read_bytes()
        if hashlib.sha256(data).hexdigest() != ROM_SHA256:
            raise ValueError('Unsupported ROM')
        self.prg = data[16:16 + 131072]
        self._forest_bytes=np.frombuffer(self.prg,dtype=np.uint8)[2*0x4000:3*0x4000]
        self._collision_luts={}

    def read(self, address, bank):
        if not 0x8000 <= address <= 0xFFFF:
            raise ValueError('Not a PRG address')
        mapped = 7 if address >= 0xC000 else bank
        return self.prg[mapped * 0x4000 + (address & 0x3FFF)]

    def observe(self, ram, *, player_centered=False):
        """12x17 raw cells, camera-aligned or centered on RAM $56/$58.

        Rows increase downward; columns increase toward world +X. Invalid
        cells use -1 and a zero mask, distinct from valid raw tile code zero.
        No dynamic actor is painted into the static ROM map.
        """
        x, y = word(ram, 0x56), word(ram, 0x58)
        origin_x = (x // 16 - 8) * 16 if player_centered else (word(ram, 0x6E) // 16) * 16
        origin_y = (y // 16 - 6) * 16 if player_centered else 0
        tiles = np.full((12, 17), -1, np.int32)
        classes = tiles.copy()
        valid = np.zeros((12, 17), np.int8)
        map_shape = np.full(2, -1, np.int32)
        # Only bank 2's forest collision lookup has been execution-validated.
        supported = self.dispatch_predicate(ram) and ram[0x4D] == 2
        if player_centered:
            supported = supported and x != 0xFFFF and y != 0xFFFF and int(ram[0x60]) != 255
        if supported:
            rows = [word(ram, 0x50E + 2*r) for r in range(12)]
            widths = [b-a for a,b in zip(rows, rows[1:])]
            # Both widths have actual bank-2 CPU-read/return evidence. Require
            # the complete row table to match; do not guess a width per row.
            supported = (len(set(widths)) == 1 and widths[0] in (80, 112)
                         and 0x8000 <= rows[0] and rows[-1] + widths[0] <= 0xC000)
        if supported:
            width = widths[0]
            thresholds = word(ram, 0x7C)
            count = int(ram[0x2C])
            if 0x8000 <= thresholds <= 0xFFFF - count and 1 <= count <= 16:
                map_shape[:] = [12, width]
                limits = tuple(self.read(thresholds + i, 2) for i in range(count))
                if limits not in self._collision_luts:
                    lut=np.full(256,255,np.int32);codes=np.arange(256)
                    # Reverse assignment retains the FIRST qualifying threshold,
                    # including duplicate/unsorted tables. Tile zero stays255.
                    for i in range(count-1,-1,-1):lut[(codes!=0)&(codes<=limits[i])]=i*2
                    self._collision_luts[limits]=lut
                world_rows=origin_y//16+np.arange(12);world_cols=origin_x//16+np.arange(17)
                pointers=np.asarray(rows)[np.clip(world_rows,0,11)]
                addresses=pointers[:,None]+world_cols[None,:]
                available=(world_rows[:,None]>=0)&(world_rows[:,None]<12)&(world_cols[None,:]>=0)&(world_cols[None,:]<width)&(addresses>=0x8000)&(addresses<0xC000)
                codes=self._forest_bytes[addresses[available]-0x8000]
                tiles[available]=codes;classes[available]=self._collision_luts[limits][codes];valid[available]=1
        return dict(origin=np.array([origin_x, origin_y], np.int32), tile_ids=tiles,
                    collision_classes=classes, valid=valid,
                    player_offset=np.array([x - origin_x, y - origin_y], np.int32),
                    map_shape=map_shape,
                    tile_size=16, dynamic_floor_changes_in_actor_state=True)

    def sprite_bounds(self, ram, frame, x, y, flags):
        """Union of hardware sprite components; not a collision hitbox.

        Restrict to normal Y orientation. Hidden objects and untested formats
        explicitly have unavailable bounds. Multiple OAM entries form a frame.
        """
        if ram[0x4D] != 2 or not self.dispatch_predicate(ram) or flags & 0x90:
            return None
        table = word(ram, 0x8D)
        try:
            pointer = self.read(table + frame*2, 2) | self.read(table + frame*2 + 1, 2) << 8
            components = []
            for i in range(64):
                offset = pointer + 4*i
                dy = self.read(offset, 2)
                if dy == 128:
                    break
                dx = self.read(offset + 3, 2)
                dx = dx - 256 if dx >= 128 else dx
                dy = dy - 256 if dy >= 128 else dy
                sx = x - 8 - dx if flags & 64 else x + dx
                components.append((sx, y + dy))
            else:
                return None
            if not components:
                return None
            return [min(a for a,b in components), min(b for a,b in components),
                    max(a for a,b in components)+8, max(b for a,b in components)+8]
        except (ValueError, IndexError):
            return None

    def triggers(self, ram):
        """Sparse trigger records, distinct from solid terrain tiles.

        Rectangles are decoded by CB3C. Action arguments remain raw: an
        overlap does not by itself imply a checkpoint, exit, or reward.
        """
        rectangles = np.full((16, 4), -1, np.int32)
        arguments = np.full((16, 5), -1, np.int32)
        valid = np.zeros(16, np.int8)
        if self.dispatch_predicate(ram) and ram[0x4D] == 2:
            pointer = word(ram, 0x7E)
            try:
                for slot in range(16):
                    data = [self.read(pointer + 11*slot + i, 2) for i in range(11)]
                    x = data[0] | data[1] << 8
                    if x == 0:
                        break
                    rectangles[slot] = [x, data[3], data[2], data[5]]
                    arguments[slot] = data[6:11]
                    valid[slot] = 1
            except (ValueError, IndexError):
                valid[:] = 0
        return dict(rectangles=rectangles, action_arguments_raw=arguments, valid=valid)
