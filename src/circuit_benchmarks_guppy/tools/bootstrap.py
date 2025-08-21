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


def full_bootstrap(hists, seq_len_index=0, num_resamples=500):
    """ non-parametric resampling from circuits
        parametric resampling from hists
    """
    
    # read in seq_len and input states
    seq_len = list(set([sett[seq_len_index] for sett in hists]))
    
    boot_hists = []
    for i in range(num_resamples):
        
        # first do non-parametric resampling
        hists_resamp = {}
        for L in seq_len:
            # make list of exp names to resample from
            circ_list = []
            for sett in hists:
                if sett[seq_len_index] == L:
                    circ_list.append(sett)
            # resample from circ_list
            seq_reps = len(circ_list)
            resamp_circs = np.random.choice(seq_reps, size=seq_reps)
            for rep, rep2 in enumerate(resamp_circs):
                circ_resamp = circ_list[rep2]
                exp_out_resamp = circ_resamp[2]
                sett_resamp = (L, rep, exp_out_resamp)
                outcomes = hists[circ_resamp]
                hists_resamp[sett_resamp] = outcomes
        
        # do parametric resample
        boot_hists.append(resample_hists(hists_resamp))
    
    return boot_hists

