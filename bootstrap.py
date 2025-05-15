# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 09:18:03 2025

Bootstrapping functions

@author: Karl.Mayer
"""

import numpy as np


def resample_hists(hists):
    """ hists (dict): outcome dictionaries for each circuit label
        returns re_hists (dict): same format, resampled
    """
    
    re_hists = {}
    
    for name in hists:
        outcomes = hists[name]
        re_out = resample_outcomes(outcomes)
        re_hists[name] = re_out
    
    return re_hists


def resample_outcomes(outcomes):
    
    # read in number of shots
    shots = sum(outcomes.values())
    out_list = list(outcomes.keys())
    
    # define probability distribution to resample from
    p = [outcomes[b_str]/shots for b_str in out_list]
    
    # resample
    r = list(np.random.choice(out_list, size=shots, p=p))
    re_out = {b_str:r.count(b_str) for b_str in out_list}
    
    return re_out
