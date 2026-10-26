"""Exact scoped text-region comparison allowing small native narrative scroll."""
import numpy as np


def match_reference_region(observed,reference,box,*,max_vertical_shift=0):
    if observed.shape!=reference.shape!=(240,256,3):raise ValueError('Mismatched native screenshots')
    if observed.shape!=(240,256,3) or reference.shape!=(240,256,3):raise ValueError('Expected native RGB240x256')
    if type(max_vertical_shift) is not int or not 0<=max_vertical_shift<=4:raise ValueError('Bounded native title scroll only')
    left,top,right,bottom=box;target=observed[top:bottom,left:right]
    if len(np.unique(target.reshape(-1,3),axis=0))<2:return None
    for shift in range(-max_vertical_shift,max_vertical_shift+1):
        if top+shift<0 or bottom+shift>240:continue
        if np.array_equal(target,reference[top+shift:bottom+shift,left:right]):return shift
    return None
