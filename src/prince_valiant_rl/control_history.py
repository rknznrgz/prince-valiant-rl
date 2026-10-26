"""Past applied-controller edge ages, in native frames, with bounded knowledge."""
import numpy as np

BUTTON_BITS=(1,2,16,32,64,128)  # A,B,UP,DOWN,LEFT,RIGHT; actual NES bit masks.


class ControlHistory:
    def __init__(self,cap=120,previous_mask=0):
        if int(cap)!=cap or cap<1:raise ValueError('Positive native-frame age cap required')
        self.cap=int(cap);self.previous_mask=int(previous_mask)
        self.ages=np.zeros(len(BUTTON_BITS),np.int32)
        self.known=np.zeros(len(BUTTON_BITS),bool)

    def push(self,mask):
        mask=int(mask)
        if not 0<=mask<=255:raise ValueError('Invalid applied controller mask')
        for i,bit in enumerate(BUTTON_BITS):
            if (self.previous_mask^mask)&bit:self.ages[i]=0;self.known[i]=True
            else:
                self.ages[i]=min(self.cap,self.ages[i]+1)
                if self.ages[i]==self.cap:self.known[i]=True
        self.previous_mask=mask

    def observable(self):
        return np.r_[np.where(self.known,self.ages/self.cap,0.),self.known].astype(np.float32)

    @classmethod
    def from_recording(cls,buttons,frame,start,cap=120):
        if not 0<=start<=frame<=len(buttons):raise ValueError('Invalid past-input interval')
        earliest=max(start,frame-cap)
        # A supplied prefix still tells us the immediately preceding applied
        # mask at the playable boundary. Keep ages unknown, but do not invent
        # a new press when a button was already held across that boundary.
        result=cls(cap,int(buttons[earliest-1]) if earliest>0 else 0)
        for mask in buttons[earliest:frame]:result.push(int(mask))
        return result
