"""Pure CPU owner-local scene/source reuse; no neural-network imports."""
from .scenes import ControlledScene

class SequentialScenePool:
    def __init__(self,worker):self.worker=worker;self.sources={};self.scenes={}
    def get(self,name):
        if name not in self.scenes:
            env=ControlledScene(name)
            if env.config['worker']!=self.worker:raise ValueError('Worker owner mismatch')
            env.cache=self.sources;self.scenes[name]=env
        return self.scenes[name]
    def close(self):
        for pool in self.sources.values():
            pool.env.core.before_native_frame=None;pool.env.core.after_native_frame=None;pool.close()
        self.sources.clear()
