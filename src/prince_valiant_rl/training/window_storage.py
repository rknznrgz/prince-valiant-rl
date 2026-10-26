"""Lossless CPU storage for categorical bytes; neural inputs retain their values."""
import numpy as np

BYTE_FIELDS=('categorical','entity_codes','projectile_codes','trigger_codes','tiles','classes')

def compact_window(window):
    result=dict(window)
    for key in BYTE_FIELDS:
        if key not in result:continue
        value=np.asarray(result[key])
        if np.issubdtype(value.dtype,np.integer) and value.dtype.itemsize>1 and value.size and value.min()>=0 and value.max()<=255:
            result[key]=value.astype(np.uint8)
    return result

def materialize_observation_batch(rollout,indices,window_keys):
    """Stack only the requested rollout rows, preserving index order and dtype."""
    return {key:np.stack([rollout[int(i)]['obs'][key] for i in indices])
            for key in window_keys}
