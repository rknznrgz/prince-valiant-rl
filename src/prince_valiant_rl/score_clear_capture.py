"""Bounded, opt-in RAM transition ring for score-clear diagnosis."""
from collections import deque
import json
import os
from pathlib import Path
import tempfile

from .game_state import decode_display_score


class ScoreClearRing:
    def __init__(self, worker_id, output_dir, *, capacity=33, provenance=None):
        if capacity < 33:
            raise ValueError('Score-clear capture capacity must retain 32 predecessors plus the current transition')
        self.worker_id = int(worker_id)
        self.output_dir = Path(output_dir)
        self.capacity = int(capacity)
        self.provenance = dict(provenance or {})
        self.transitions = deque(maxlen=self.capacity)
        self.episode_id = None
        self.failing_transition = None
        self.failure_error = None

    @staticmethod
    def _score(ram):
        try:
            return decode_display_score(ram)
        except Exception:
            return None

    @staticmethod
    def _markers(ram):
        return dict(state=int(ram[0x5F]), life_phase=int(ram[0x60]),
                    lives_raw=int(ram[0xF7]), lives=int(ram[0xF7])-256 if int(ram[0xF7]) >= 128 else int(ram[0xF7]),
                    health=int(ram[0xFC]), retry_continue=int(ram[0x02FF]))

    def begin_episode(self, episode_id):
        self.episode_id = str(episode_id)

    def before_validation(self, *, pre_ram, post_ram, mask, elapsed_before,
                          game_elapsed_before, reset_frame, baseline, reset_provenance,
                          native_frame_index):
        pre = bytes(pre_ram)
        post = bytes(post_ram)
        item = dict(worker_id=self.worker_id, episode_id=self.episode_id,
                    episode_native_frame=int(elapsed_before)+1,
                    native_frame_index=int(native_frame_index), executed_mask=int(mask),
                    raw_pre_ram=pre.hex(), raw_post_ram=post.hex(),
                    score_pre_bcd=pre[0xF4:0xF7].hex(), score_post_bcd=post[0xF4:0xF7].hex(),
                    decoded_score_pre=self._score(pre_ram), decoded_score_post=self._score(post_ram),
                    score_baseline=baseline, reset_frame=reset_frame,
                    reset_restore_provenance=reset_provenance,
                    state_pre=self._markers(pre_ram), state_post=self._markers(post_ram),
                    terminal_marker=None, task_events=None)
        self.transitions.append(item)
        return item

    def after_transition(self, item, *, task_events, terminal_marker):
        item['task_events'] = dict(task_events)
        item['terminal_marker'] = terminal_marker

    def document(self, *, failing_transition=None, error=None):
        if failing_transition is not None:self.failing_transition = failing_transition
        if error is not None:self.failure_error = str(error)
        return dict(schema='level1_score_clear_native_ring_v1', worker_id=self.worker_id,
                    ring_capacity=self.capacity, provenance=self.provenance,
                    transition_count=len(self.transitions), transitions=list(self.transitions),
                    failing_transition=self.failing_transition, error=self.failure_error)

    def persist(self, *, failing_transition=None, error=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / f'worker_{self.worker_id}.json'
        payload = json.dumps(self.document(failing_transition=failing_transition, error=error),
                             separators=(',', ':'), allow_nan=False).encode('utf-8')
        fd, temp_name = tempfile.mkstemp(prefix=target.name+'.', suffix='.pending', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, target)
        except BaseException:
            try: os.unlink(temp_name)
            except OSError: pass
            raise
        return str(target)
