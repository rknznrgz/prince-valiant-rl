"""RAM-only state for the verified forest scene of this exact ROM.

No screen reads. Unknown semantics are None, not guessed values. Coordinates
are integer pixels; stored movement steps are not instantaneous velocities.
"""
import numpy as np

ROM_SHA256 = '96ad0a41a34fccc98bb7255890b131f77d4131cc69594c77e3ff33cd3a8fe6a5'


def word(ram, address):
    return int(ram[address]) | int(ram[address + 1]) << 8


def signed_byte(value):
    return int(value) - 256 if int(value) >= 128 else int(value)


def decode_display_score(ram):
    """Decode the six-digit displayed score from strict packed BCD at $F4-$F6.

    This is deliberately separate from ``extract_game_state``'s legacy
    ``hud.score_digits`` observation feature, which reads $F8-$FA and is part
    of existing policy/checkpoint inputs. Invalid nibbles are not coerced.
    """
    ram = np.asarray(ram)
    if ram.shape != (0x800,) or ram.dtype != np.uint8:
        raise ValueError('Expected a uint8 array of 2048 RAM bytes')
    score = 0
    for byte in ram[0xF4:0xF7]:
        high, low = int(byte) >> 4, int(byte) & 0x0F
        if high > 9 or low > 9:
            raise ValueError(f'Invalid packed BCD display score byte: 0x{int(byte):02X}')
        score = score * 100 + high * 10 + low
    return score


def forest_dispatch_valid(ram):
    """Normal forest or the execution-tested bank-2 encounters."""
    ordinary = int(ram[0x33]) == 8
    encounter_codes22_23 = (int(ram[0x33]) == 7 and int(ram[0x4D]) == 2 and
                       any(int(v) in (22, 23) for v in ram[0xC1:0xCB]))
    encounter_code14_phase1 = (int(ram[0x33]) == 7 and int(ram[0x4D]) == 2 and
                        any(int(ram[0xC1+s]) == 14 and int(ram[0x99+s]) == 1
                            for s in range(10)))
    return (ordinary or encounter_codes22_23 or encounter_code14_phase1) and int(ram[0x2F4]) == 0


def extract_game_state(ram, previous_ram=None, elapsed_frames=1, terrain_decoder=None):
    """Decode a copied 2KB RAM frame. Differences require contiguous frames.

    A caller must verify ROM identity and restrict interpretation to the tested
    forest gameplay checkpoints. A dispatch value of 8 is necessary here, but
    is not proof that a later level has the same structures.
    """
    ram = np.asarray(ram)
    if ram.shape != (0x800,) or ram.dtype != np.uint8:
        raise ValueError('Expected a uint8 array of 2048 RAM bytes')
    if elapsed_frames < 1:
        raise ValueError('elapsed_frames must be positive')
    x, y = word(ram, 0x56), word(ram, 0x58)
    camera = word(ram, 0x6E)
    phase = int(ram[0x60])
    retries = signed_byte(ram[0xF7])
    mode_valid = forest_dispatch_valid(ram)
    position_valid = mode_valid and x != 0xFFFF and y != 0xFFFF and phase != 255
    dx = dy = None
    if previous_ram is not None:
        previous_ram = np.asarray(previous_ram)
        if previous_ram.shape != (0x800,) or previous_ram.dtype != np.uint8:
            raise ValueError('Expected previous uint8 RAM frame')
        # Death/reinitialization and state discontinuities are not velocities.
        if position_valid and forest_dispatch_valid(previous_ram) and int(previous_ram[0x60]) == 0 and phase == 0:
            px, py = word(previous_ram, 0x56), word(previous_ram, 0x58)
            if px != 0xFFFF and py != 0xFFFF and signed_byte(previous_ram[0xF7]) == retries:
                dx, dy = (x - px) / elapsed_frames, (y - py) / elapsed_frames
    entities = []
    for slot in range(10):
        ex, ey = word(ram, 0x28C + 2 * slot), word(ram, 0x2A0 + 2 * slot)
        code, flags = int(ram[0xC1 + slot]), int(ram[0xCB + slot])
        entities.append(dict(slot=slot, world_x=ex, y=ey, kind_code=code,
                             flags_raw=flags, occupied=code != 0,
                             render_eligible=code != 0 and bool(flags & 4),
                             hidden=bool(flags & 16),
                             phase_raw=int(ram[0x99 + slot]),
                             direction_raw=int(ram[0xD5 + slot]),
                             animation_raw=int(ram[0xDF + slot]),
                             arguments_raw=[int(ram[a + slot]) for a in (0xA3, 0xAD, 0xB7)],
                             motion_or_timer_raw=word(ram, 0x2BA + 2 * slot),
                             screen_anchor_x=ex - camera,
                             coordinate_semantics='object anchor; not necessarily final sprite position'))
    projectiles = [dict(slot=s, world_x=word(ram, 0x502 + 2*s),
                        y=int(ram[0x2B4 + 2*s]), direction_raw=int(ram[0x2B5 + 2*s]),
                        state_raw=int(ram[0x509 + 2*s]), animation_raw=int(ram[0x508 + 2*s]),
                        active=bool(ram[0x509 + 2*s]),
                        flying=(int(ram[0x509 + 2*s]) & 127) == 1,
                        screen_anchor_x=word(ram, 0x502 + 2*s)-camera)
                   for s in range(3)]
    for entity in entities:
        flags = entity['flags_raw'] | (64 if entity['direction_raw'] & 1 else 0)
        entity['sprite_bounds_screen'] = (terrain_decoder.sprite_bounds(
            ram, entity['animation_raw'], entity['screen_anchor_x'], entity['y'], flags)
            if terrain_decoder is not None and entity['render_eligible'] else None)
        bounds = entity['sprite_bounds_screen']
        entity['sprite_intersects_screen'] = (bounds[0] < 256 and bounds[2] > 0 and
                                             bounds[1] < 240 and bounds[3] > 0) if bounds else None
    return dict(
        validity=dict(rom_sha256=ROM_SHA256, tested_scope='forest gameplay only',
                      gameplay_dispatch_matches=mode_valid, player_position_valid=position_valid),
        player=dict(world_x=x, y=y, facing_left=bool(ram[0x65]),
                    movement_state=int(ram[0x5F]), state_frame=int(ram[0x5E]),
                    life_phase_raw=phase, vertical_step=signed_byte(ram[0x66]),
                    horizontal_step_magnitude=int(ram[0x67]),
                    velocity_x_derived=dx, velocity_y_derived=dy,
                    subpixel_x=None, subpixel_y=None,
                    health=int(ram[0xFC]) if mode_valid else None,
                    retries_remaining=retries if retries >= 0 else None,
                    lives=None, screen_anchor_x=x-camera),
        camera=dict(world_x=camera, scroll_x_copy=word(ram, 0x18)),
        entities=entities, player_projectiles=projectiles,
        hud=dict(score_digits=[int(ram[a]) for a in range(0xF8, 0xFB)],
                 timer_digits=[int(ram[a]) for a in range(0x286, 0x28A)],
                 timer_subcounter_raw=int(ram[0x28A]), valid=mode_valid),
        protection=dict(damage_gate_raw=int(ram[0xF3]), flags_raw=int(ram[0x284])),
        scene_identity_raw=[int(ram[a]) for a in (0x33, 0x2F4, 0x4D, 0x529)],
        terrain=terrain_decoder.observe(ram) if terrain_decoder is not None else None,
        nearby_terrain=terrain_decoder.observe(ram, player_centered=True) if terrain_decoder is not None else None,
        triggers=terrain_decoder.triggers(ram) if terrain_decoder is not None else None,
        level=None, level_complete=None,
        game_dispatch_raw=int(ram[0x33]), player_dispatch_raw=int(ram[0x2F4]),
        terminal=dict(life_sequence_active=phase == 1,
                      game_over=bool(mode_valid and phase == 255 and retries < 0),
                      success=None),
    )
