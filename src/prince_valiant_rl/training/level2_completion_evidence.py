"""Conservative Level2 completion evidence: flag, visible card, then Level3 title."""
import numpy as np

COMPLETE_BOX=(70,124,235,145)
NEXT_TITLE_BOX=(35,43,230,70)


def region(image,box):
    image=np.asarray(image)
    if image.shape!=(240,256,3) or image.dtype!=np.uint8:raise ValueError('Expected native RGB240x256')
    left,top,right,bottom=box
    return image[top:bottom,left:right]


class CompletionEvidence:
    def __init__(self,complete_template,next_template):
        self.complete=region(complete_template,COMPLETE_BOX).copy();self.next=region(next_template,NEXT_TITLE_BOX).copy()
        if len(np.unique(self.complete.reshape(-1,3),axis=0))<2 or len(np.unique(self.next.reshape(-1,3),axis=0))<2:raise ValueError('Blank terminal templates rejected')
        self.entered=False;self.flag=None;self.card=None;self.next_level=None

    def observe(self,frame,ram,image):
        if int(ram[0xF3])==0:self.entered=True
        if self.entered and self.flag is None and int(ram[0xF3])==127:self.flag=int(frame)
        if self.flag is not None and frame>=self.flag and np.array_equal(region(image,COMPLETE_BOX),self.complete):self.card=self.card if self.card is not None else int(frame)
        if self.card is not None and frame>self.card and np.array_equal(region(image,NEXT_TITLE_BOX),self.next):self.next_level=self.next_level if self.next_level is not None else int(frame)
        return self.confirmed

    @property
    def confirmed(self):return self.flag is not None and self.card is not None and self.next_level is not None


def bonus_event(pre,post,slot,score_gain):
    if not (pre[0xC1+slot]==8 and post[0xC1+slot]==9 and pre[0x3F6+slot] and score_gain==100):return None
    phase=int(pre[0x99+slot])&3
    return dict(slot=slot,type='E' if phase==2 else 'T' if phase==1 else 'unclassified',phase=phase,
        energy_gain=max(0,int(post[0xFC])-int(pre[0xFC])),resource_before=pre[0x286:0x289].tolist(),resource_after=post[0x286:0x289].tolist())
