# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:33:45 2024

@author: Karl.Mayer

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Data processing functions

def marginalize_hists(qubits, hists, mar_exp_out=False):
    """ return list of hists of same length as number of qubit pairs """
    
    if type(qubits[0]) == int:
        qubits = [qubits]
    
    # number of qubits needed
    n = max([max(q_pair) for q_pair in qubits]) + 1
    
    mar_hists = []
    for j, q_pair in enumerate(qubits):
        q0, q1 = q_pair
        hists_q = {}
        for name in hists:
            if mar_exp_out:
                exp_out = name[-1][j]
            else:
                exp_out = name[-1]
            out = hists[name]
            mar_out = {}
            for b_str in out:
                counts = out[b_str]
                # marginalize bitstring
                mar_b_str = b_str[q0] + b_str[q1]
                if mar_b_str in mar_out:
                    mar_out[mar_b_str] += counts
                elif mar_b_str not in mar_out:
                    mar_out[mar_b_str] = counts
            # append marginalized outcomes to hists
            name_q = list(name)
            name_q[-1] = exp_out
            hists_q[tuple(name_q)] = mar_out
        mar_hists.append(hists_q)
    
    return mar_hists


def merge_outcomes(out1, out2):
    """ combine outcomes from different circuit executions (useful for RC)
        out1, out2 (dict)
    """
    
    outcomes = out1
    for b_str in out2:
        counts = out2[b_str]
        if b_str in outcomes:
            outcomes[b_str] += counts
        elif b_str not in outcomes:
            outcomes[b_str] = counts
    
    return outcomes


# Analysis functions

def success_probability(exp_out, outcomes):
    
    shots = sum(list(outcomes.values()))
    
    if exp_out in outcomes:
        p = outcomes[exp_out]/shots
    else:
        p = 0.0
    
    return p


def get_success_probs(hists):
    
    
    # read in list of sequence lengths
    seq_len = list(set([sett[-3] for sett in list(hists.keys())]))
    seq_len.sort()
    
    success_probs = {L:[] for L in seq_len}
    
    for setting in hists:
        L = setting[-3]
        exp_out = setting[-1]
        outcomes = hists[setting]
        p = success_probability(exp_out, outcomes)
        success_probs[L].append(p)
    
    
    return success_probs


def get_avg_success_probs(hists):
    
    success_probs = get_success_probs(hists)
    
    avg_success_probs = {}
    for L in success_probs:
        avg_success_probs[L] = float(np.mean(success_probs[L]))
    
    return avg_success_probs


def estimate_fidelity(avg_success_probs, rescale_fidelity=False):
    """ estimate average fidelity from average success probs
        
        rescale_fidelity: if true, convert Clifford fidelity into gate fidelity
    """
    
    # define fit function
    def fit_func(L, a, f):
        return a*f**L + 1/4
    
    
    x = [L for L in avg_success_probs]
    x.sort()
    
    y = [avg_success_probs[L] for L in x]
    
    # perform best fit
    popt, pcov = curve_fit(fit_func, x, y, p0=[0.7, 0.9], bounds=([0,0], [0.75,1]))
    block_avg_fidelity = 1 - 3*(1-popt[1])/4
    
    if rescale_fidelity == True:
        # 1.5 gates per TQ Clifford
        avg_fidelity = 1 - 2*(1 - block_avg_fidelity)/3
    else:
        avg_fidelity = block_avg_fidelity
        
    
    
    return float(avg_fidelity)


# Plotting functions

def plot_TQ_decays(seq_len, avg_success_probs, avg_success_stds=None, **kwargs):
    """ 
        seq_len: list of circuit depths
        avg_succ_probs: list of dicts of success probs
                        one dict for each qubit pair
        avg_succ_stds: list of dicts of error bars (if not None)
    """
    
    num_q_pairs = len(avg_success_probs)
    ylim = kwargs.get('ylim', None)
    title = kwargs.get('title', None)
    labels = kwargs.get('labels', [None for _ in range(num_q_pairs)])
    
    # define fit function
    def fit_func(L, a, f):
        return a*f**L+1/4
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r', 'c', 'm', 'y', 'b', 'g', 'r']
    xfit = np.linspace(seq_len[0], seq_len[-1], 100)
    
    for j, avg_succ_probs in enumerate(avg_success_probs):
        co = colors[j]
        y = [avg_succ_probs[L] for L in seq_len]
        
        if avg_success_stds:
            yerr = [avg_success_stds[j][L] for L in seq_len]
        else:
            yerr = None
        
        # perform best fit
        popt, pcov = curve_fit(fit_func, seq_len, y, p0=[0.7, 0.9], bounds=([0,0], [0.75,1]))
        yfit = fit_func(xfit, *popt)
        plt.errorbar(seq_len, y, yerr=yerr, fmt='o', color=co, label=labels[j])
        plt.plot(xfit, yfit, '-', color=co)
    
    plt.title(title)
    plt.ylabel('Success Probability')
    plt.xlabel('Sequence Length')
    plt.xticks(ticks=seq_len, labels=seq_len)
    plt.ylim(ylim)
    if labels[0]:
        plt.legend()
    plt.show()
    
    
    
    
    
    