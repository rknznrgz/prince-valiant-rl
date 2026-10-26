"""Opt-in observation v3: executed controller input latches, not scene labels."""
import numpy as np
import gymnasium as gym
from .interface import FeatureEncoder

class InputFeatureEncoder(FeatureEncoder):
    def __call__(self,r,previous,history,state=None):
        out=super().__call__(r,previous,history,state=state)
        # Executed bank7 C6B4-C6BD: held $0D, new presses $0F.
        # The final scalar marks availability, including old demonstration rows.
        out['scalars']=np.concatenate([out['scalars'],np.ones(1,np.float32)])
        out['categorical']=np.concatenate([out['categorical'],np.asarray([r[0x0D],r[0x0F]],np.int64)])
        return out

def configure_encoder(env,enabled):
    if isinstance(env.encoder,InputFeatureEncoder)==bool(enabled):return
    env.encoder=InputFeatureEncoder() if enabled else FeatureEncoder()
    from .events import EventTracker
    sample=env.encoder(np.zeros(2048,np.uint8),None,EventTracker())
    env.observation_space=gym.spaces.Dict({k:gym.spaces.Box(-4096,4096,v.shape,v.dtype) if v.dtype==np.float32
        else gym.spaces.Box(0,256,v.shape,v.dtype) for k,v in sample.items()})

def upgrade_model(baseline):
    """Explicit compatible transfer; new input columns start at zero.

    No new semantic value is fabricated for demonstrations: absent controller
    fields receive availability0 and their embeddings are gated to zero.
    """
    from .temporal_models import TemporalActorCritic
    c=dict(baseline.config,input_latches=True,feature_schema_version=3,observation_version='structured_input_latches_v3')
    shapes=dict(baseline.shapes);shapes['scalars']=(37,);shapes['categorical']=(11,)
    model=TemporalActorCritic(shapes,baseline.actions,c).to(next(baseline.parameters()).device)
    old=baseline.state_dict();new=model.state_dict()
    original_categorical_width=9*c['embedding_dim']
    additional_categorical_width=2*c['embedding_dim']
    for key,target in new.items():
        if key.endswith('mlp.0.weight'):
            target.zero_();target[:,:36]=old[key][:,:36]
            categorical_end=37+original_categorical_width
            target[:,37:categorical_end]=old[key][:,36:36+original_categorical_width]
            target[:,categorical_end+additional_categorical_width:]=old[key][:,36+original_categorical_width:]
        else:target.copy_(old[key])
    model.load_state_dict(new)
    return model

def augment_unavailable_demonstrations(data):
    """Unknown old controller fields remain explicitly unavailable, not measured0."""
    data=dict(data)
    data['scalars']=np.concatenate([data['scalars'],np.zeros((len(data['scalars']),1),np.float32)],-1)
    data['categorical']=np.concatenate([data['categorical'],np.zeros((len(data['categorical']),2),data['categorical'].dtype)],-1)
    return data

def attach_recorded_demonstrations(data,path):
    import hashlib
    from ..paths import ROOT
    records=np.load(path)
    if str(records['dataset_sha256'])!=hashlib.sha256((ROOT/'training/temporal_v1/dataset.npz').read_bytes()).hexdigest():raise ValueError('Controller-data base version mismatch')
    if len(records['codes'])!=len(data['scalars']):raise ValueError('Controller field alignment mismatch')
    out=dict(data);out['scalars']=np.concatenate([data['scalars'],records['available'][:,None]],-1)
    out['categorical']=np.concatenate([data['categorical'],records['codes'].astype(data['categorical'].dtype)],-1)
    return out
