"""Image-only supervised representation and opt-in RAM/vision temporal fusion."""
from collections import deque
import hashlib

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .temporal_models import TemporalActorCritic, WindowPath
from ..backward_level1 import Level1Env
from ..game_state import extract_game_state, word
from ..navigation import supported, live

PREPROCESS='gray_area_128x96_playfield_v1'
PIXEL_OFFSETS=(-6,-3,0)
VISUAL_MODES=('ram','random','frozen','finetune')


def preprocess(rgb):
    """Keep the entire 256x192 playfield, remove HUD, halve with area sampling."""
    rgb=np.asarray(rgb)
    if rgb.shape!=(240,256,3) or rgb.dtype!=np.uint8:raise ValueError('Expected native RGB screen')
    return np.asarray(Image.fromarray(rgb[:192]).convert('L').resize((128,96),Image.Resampling.BOX)).copy()


def image_stack(frames,current):
    blank=np.zeros((96,128),np.uint8)
    return np.stack([frames.get(current+off,blank) for off in PIXEL_OFFSETS])


class VisualLevel1Env(Level1Env):
    """Exact native pixels at the existing RAM timestamps; no reset-history leaks.

    Route history is reconstructed from authentic past actions before a reset.
    Arbitrary snapshot restore without pixel history is rejected, never padded
    with stale frames. The standard Level1Env remains unchanged.
    """
    def __init__(self,config,route=None):
        super().__init__(config,route)
        if config['frame_skip']!=1:raise ValueError('Visual pilot requires one native frame per action')
        self.pixels=deque(maxlen=1-min(config['history_offsets'])-min(PIXEL_OFFSETS))

    def reset(self,frame=None,**kwargs):
        frame=self.route.start if frame is None else int(frame)
        variation=kwargs.get('variation',0)
        if variation:raise ValueError('Use explicit native perturbation steps for local visual starts')
        obs,info=super().reset(frame,**kwargs)
        self.pixels.clear()
        earliest=max(self.route.start,frame+min(self.config['history_offsets'])+min(PIXEL_OFFSETS))
        self.route.restore(self.native,earliest)
        for f in range(earliest,frame+1):
            np.testing.assert_array_equal(self.native.ram_buffer(),self.route.ram[f])
            self.pixels.append((f-frame,preprocess(self.native.copy_screen_rgb())))
            if f<frame:self.native.frame_advance(int(self.route.buttons[f]))
        np.testing.assert_array_equal(self.ram,self.native.ram_buffer())
        return self.with_pixels(obs),info

    def with_pixels(self,obs):
        frames=dict(self.pixels);current=self.pixels[-1][0]
        # RAM history_valid is authoritative for unavailable timestamps.
        obs=dict(obs)
        obs['screen']=np.stack([image_stack(frames,current+off) if valid else np.zeros((3,96,128),np.uint8)
            for off,valid in zip(self.config['history_offsets'],obs['history_valid'])])
        return obs

    def step(self,action):
        obs,reward,t,u,info=super().step(action)
        self.pixels.append((self.pixels[-1][0]+1,preprocess(self.native.copy_screen_rgb())))
        return self.with_pixels(obs),reward,t,u,info

    def snapshot(self):
        state=super().snapshot();state['visual_pixels']=list(self.pixels)
        state['visual_preprocess']=PREPROCESS
        return state

    def restore_snapshot(self,state):
        if state.get('visual_preprocess')!=PREPROCESS or 'visual_pixels' not in state:
            raise ValueError('Visual snapshot needs matching image history; RAM-only snapshot is incompatible')
        obs,info=super().restore_snapshot(state)
        self.pixels.clear();self.pixels.extend(state['visual_pixels'])
        np.testing.assert_array_equal(self.pixels[-1][1],preprocess(self.native.copy_screen_rgb()))
        return self.with_pixels(obs),info


class VisualEncoder(nn.Module):
    """Three ordered grayscale frames only; no RAM inputs or batch statistics."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,5,stride=2,padding=2),nn.ReLU(),
            nn.Conv2d(16,24,3,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(24,32,3,stride=2,padding=1),nn.ReLU(),
            nn.Flatten(),nn.Linear(32*12*16,64),nn.LayerNorm(64),nn.ReLU())
    def forward(self,x):
        if x.shape[-3:]!=(3,96,128):raise ValueError('Visual input/preprocessing mismatch')
        return self.net(x.float()/255.)


class VisualPredictor(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=VisualEncoder()
        self.player=nn.Linear(64,2);self.objects=nn.Linear(64,2*12*16)
        self.terrain=nn.Linear(64,12*16)
    def forward(self,images):
        z=self.encoder(images)
        return dict(player=self.player(z).sigmoid(),objects=self.objects(z).reshape(-1,2,12,16),
            terrain=self.terrain(z).reshape(-1,12,16))


class VisualAuxiliary:
    """Optional supervised replay ablation; frozen heads never enter action path.

    The caller supplies TRAIN-split images/labels and uses the existing policy
    optimizer. Only shared visual encoders receive these auxiliary gradients.
    This is disabled for the declared main comparison.
    """
    def __init__(self,model,predictor_state):
        if model.config.get('visual_mode') not in ('random','finetune'):
            raise ValueError('Auxiliary visual learning requires a trainable visual mode')
        self.predictors=[]
        device=next(model.parameters()).device
        for name in ('actor','critic'):
            predictor=VisualPredictor().to(device);predictor.load_state_dict(predictor_state)
            predictor.requires_grad_(False)
            predictor.encoder=getattr(model,name).visual
            self.predictors.append(predictor)
    def loss(self,images,target):
        return sum(prediction_loss(p(images),target)[0] for p in self.predictors)/len(self.predictors)


def balanced_bce(logits,target,mask):
    loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    pos=mask*target;neg=mask*(1-target)
    # Each present class has equal aggregate influence; empty targets stay valid.
    present=(pos.sum()>0).float()+(neg.sum()>0).float()
    return ((loss*pos).sum()/pos.sum().clamp_min(1)+(loss*neg).sum()/neg.sum().clamp_min(1))/present.clamp_min(1)


def prediction_loss(pred,target):
    mask=target['player_valid'][:,None]
    player=(F.smooth_l1_loss(pred['player'],target['player'],reduction='none')*mask).sum()/(2*mask.sum()).clamp_min(1)
    actor=balanced_bce(pred['objects'][:,0],target['objects'][:,0],target['objects_valid'][:,0])
    projectile=balanced_bce(pred['objects'][:,1],target['objects'][:,1],target['objects_valid'][:,1])
    terrain=balanced_bce(pred['terrain'],target['terrain'],target['terrain_valid'])
    return player*4+actor+projectile+terrain,dict(player=player,actor=actor,projectile=projectile,terrain=terrain)


def labels(ram,decoder,screen=None,visibility_version='display_v2'):
    """Visible anchors/ROM nonempty cells only; not collision or hidden state.

    Slot-free spatial labels use rendered sprite-bound centers, not unstable
    slot identities. Unknown/hidden sprite extents mask their whole target map.
    Terrain target is static tile occupancy, not dynamic support or hitboxes.
    """
    state=extract_game_state(ram,terrain_decoder=decoder)
    valid=supported(ram);camera=word(ram,0x6e)
    x=word(ram,0x56)-camera;y=word(ram,0x58)
    player_valid=live(ram) and 0<=x<256 and 0<=y<192 and int(ram[0xF3])==0
    objects=np.zeros((2,12,16),np.float32);object_valid=np.full_like(objects,float(valid))
    def point(channel,px,py):
        if 0<=px<256 and 0<=py<192:
            cx=int(px)//16;cy=int(py)//16
            objects[channel,max(0,cy-1):min(12,cy+2),max(0,cx-1):min(16,cx+2)]=1
    for e in state['entities']:
        if not e['occupied'] or not e['render_eligible'] or e['hidden']:continue
        bounds=e['sprite_bounds_screen']
        if bounds is None:
            if -32<e['screen_anchor_x']<288:object_valid[0]=0
            continue
        if e['sprite_intersects_screen']:
            point(0,(max(0,bounds[0])+min(256,bounds[2]))/2,(max(0,bounds[1])+min(192,bounds[3]))/2)
    for e in state['player_projectiles']:
        if e['flying']:point(1,e['screen_anchor_x'],e['y'])
    terrain=decoder.observe(ram);target=np.zeros((12,16),np.float32);tv=np.zeros_like(target)
    # Screen-centered 16px cells sample the camera-aligned ROM cell at center.
    columns=((camera+8+16*np.arange(16)-terrain['origin'][0])//16).astype(int)
    ok=(columns>=0)&(columns<17)
    target[:,ok]=(terrain['tile_ids'][:,columns[ok]]!=0)
    tv[:,ok]=terrain['valid'][:,columns[ok]]
    if visibility_version not in ('uniform_v1','display_v2'):raise ValueError('Unknown label visibility version')
    display=preprocess(screen) if screen is not None else None
    invisible=(display is not None and (np.ptp(display)<2 or
        (visibility_version=='display_v2' and float((display>8).mean())<.02)))
    if invisible:
        # Mode/RAM can become live before the first visible gameplay display.
        player_valid=False;object_valid[:]=0;tv[:]=0
    return dict(player=np.array([x/256,y/192],np.float32).clip(0,1),player_valid=np.float32(player_valid),
        objects=objects,objects_valid=object_valid,terrain=target,terrain_valid=tv)


class VisualWindowPath(WindowPath):
    def __init__(self,shapes,outputs,c):
        super().__init__(shapes,outputs,c)
        self.visual=VisualEncoder();self.visual_normalize=nn.LayerNorm(64,elementwise_affine=False)
        old=self.gru;self.gru=nn.GRU(old.input_size+64,c['gru_hidden'],c['gru_layers'],batch_first=True)
        self.frozen=c['visual_mode']=='frozen'
        if self.frozen:
            self.visual.requires_grad_(False);self.visual.eval()
    def train(self,mode=True):
        super().train(mode)
        if self.frozen:self.visual.eval()
        return self
    def frame_features(self,structured,terrain,flat):
        ram=super().frame_features(structured,terrain,flat)
        if self.frozen:
            with torch.no_grad():z=self.visual(flat['screen'])
        else:z=self.visual(flat['screen'])
        return torch.cat([ram,self.visual_normalize(z)],-1)


class VisualActorCritic(TemporalActorCritic):
    def __init__(self,shapes,actions,config):
        if config.get('visual_mode') not in VISUAL_MODES[1:]:raise ValueError('Unknown visual policy mode')
        if tuple(shapes.get('screen',()))!=(3,96,128) or config.get('visual_preprocess')!=PREPROCESS:
            raise ValueError('Incompatible visual observation contract')
        super().__init__(shapes,actions,config)
        self.actor=VisualWindowPath(shapes,actions,config);self.critic=VisualWindowPath(shapes,1,config)
        self.architecture='ram_visual_window_gru_v1'

    def initialize_from_ram(self,base,encoder_state=None):
        """Explicit transfer: same RAM/GRU/head tensors, zero initial visual gate.

        Additional input columns start at zero for equal initial action/value
        functions. They can learn immediately; visual fine-tuning begins as
        fusion becomes nonzero. No incompatible state_dict is silently loaded.
        """
        if self.actions!=base.actions or {k:v for k,v in self.shapes.items() if k!='screen'}!=base.shapes:
            raise ValueError('RAM initialization schema mismatch')
        for path in ('actor','critic'):
            source=getattr(base,path);dest=getattr(self,path);state=dest.state_dict()
            for key,value in source.state_dict().items():
                if key=='gru.weight_ih_l0':
                    state[key].zero_();state[key][:,:value.shape[1]].copy_(value)
                else:
                    if state[key].shape!=value.shape:raise ValueError(f'Incompatible transfer {key}')
                    state[key].copy_(value)
            dest.load_state_dict(state)
            if encoder_state is not None:dest.visual.load_state_dict(encoder_state,strict=True)


def optimizer_groups(model,config):
    groups=[]
    for name,lr in [('actor',config['actor_learning_rate']),('critic',config['critic_learning_rate'])]:
        path=getattr(model,name)
        visual=list(path.visual.parameters()) if hasattr(path,'visual') else []
        ids={id(p) for p in visual}
        groups.append(dict(params=[p for p in path.parameters() if p.requires_grad and id(p) not in ids],lr=lr))
        if any(p.requires_grad for p in visual):
            groups.append(dict(params=[p for p in visual if p.requires_grad],lr=config['visual_learning_rate']))
    return groups


def parameter_digest(module):
    h=hashlib.sha256()
    for key,value in module.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
