"""Owner-local reusable checkpoint states for bounded training experiments."""
import copy
import numpy as np
from .interface import TrainingEnv

class CachedStart:
    """Replay and verify once; restore only into the owning native emulator.

    Emulator snapshot, derived motion context and complete reward prehistory
    are restored together. Snapshots are neither serialized nor shared.
    """
    def __init__(self, checkpoint, task='manager', horizon=1200, weights=None):
        self.env=TrainingEnv(checkpoint,task=task,horizon=horizon,weights=weights)
        self.obs,self.info=self.env.reset()
        self.snapshot=self.env.core.native.dump_state()
        self.state=copy.deepcopy({key:getattr(self.env,key) for key in ('tracker','_previous','_last','ram','decisions')})
        self.frontier=self.env.core.frontier
        self.restore_count=0

    def reset(self):
        e=self.env;e.core.native.load_state(self.snapshot)
        for key,value in self.state.items():setattr(e,key,copy.deepcopy(value))
        e.core._ram=e.ram.copy();e.core.frontier=self.frontier;e.core.elapsed_frames=0;e.core._done=False
        assert np.array_equal(e.core.native.ram_buffer(),e.ram),'Owner-local snapshot RAM mismatch'
        self.restore_count+=1
        return e,copy.deepcopy(self.obs),copy.deepcopy(self.info)

    def close(self):self.env.close()
