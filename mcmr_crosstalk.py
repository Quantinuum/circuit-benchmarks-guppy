# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:17:43 2025

Mid-circuit measurement and reset (MCMR) crosstalk benchmarking

@author: Victor Colusi
"""

import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from math import ceil

from guppylang import guppy
from guppylang.std.builtins import array, barrier, comptime, result, panic 
from guppylang.std.quantum import measure_array, measure, reset, qubit, x
from guppylang.std.qsystem import measure_and_reset
from hugr.package import FuncDefnPointer
# from guppylang.std.qsystem.functional import measure, reset


from experiment import Experiment
import bootstrap as bs
# from qtm_platform.ops import order_in_zones, sleep ##V:  This needs to be updated once we get the documentation for this functionality ###



class MCMR_Crosstalk_Experiment(Experiment):
    
    def __init__(self, focus_qubits, probe_qubits, seq_lengths, seq_reps, **kwargs):
        super().__init__(**kwargs)
        self.focus_qubits = focus_qubits
        self.probe_qubits = probe_qubits 
        self.protocol = 'MCMR Crosstalk'
        self.n_qubits = self.total_qubits() 
        self.parameters = {'n_qubits':self.n_qubits,
                           'seq_lengths':seq_lengths,
                           'seq_reps':seq_reps}
        
        self.which_qubits = np.sort( np.concatenate( (self.probe_qubits, self.focus_qubits), axis = 0) )
        self.seq_lengths = seq_lengths
        self.seq_reps = seq_reps

        self.options['measure'] = kwargs.get('measure', True) # repeated measurement by default
        self.options['reset'] = kwargs.get('reset', False)

        self.setting_labels = ('seq_len', 'seq_rep', 'surv_state')
        

    def total_qubits(self): 
        ''' Given a specified focus and probe qubit, this function returns the number of
        qubits needed for the experiment.  
        '''
        return max( max(self.focus_qubits) , max(self.probe_qubits)) + 1


    def add_settings(self):
        
        for seq_len in self.seq_lengths:
            for rep in range(self.seq_reps):
                surv_state = '' 
                for _ in range(self.n_qubits):
                    surv_state += str(np.random.choice(['0'])) # VEC:  Perhaps change to array with probe qubit names...?
                
                sett = (seq_len, rep, surv_state)
                self.add_setting(sett)
        
    
    def make_circuit(self, setting: tuple) -> FuncDefnPointer:

        seq_len = setting[0] # number of MCMR's to perform on each target qubit
        
        n_qubits = self.n_qubits
        focus_qubits = self.focus_qubits 

        mode_bool = [self.options['measure'], self.options['reset']] # store 3-element boolean array dicating mcmr operation to perform 
        mode_bool.append(mode_bool[0] & mode_bool[1]) # append on boolean for toggling measure + reset



        @guppy  # guppy main program.  
        def main() -> None:
    
            q = array(qubit() for _ in range(comptime(n_qubits))) # initialize the qubit register

            # VEC:  order_in_zones needed here. (!!!!!!!!!!!!!)

            for i in range(comptime(n_qubits)): # put all qubits in the |1> state
                x(q[i])
                
            barrier(q) # add a barrier
   
            for i in range(comptime(seq_len)):
                if comptime(mode_bool[2]):
                    for j in comptime(focus_qubits):
                        measure_and_reset(q[j])
                elif comptime(mode_bool[1]):
                    for j in comptime(focus_qubits):
                        reset(q[j])
                elif comptime(mode_bool[0]):
                    panic("Repeated measurement of a qubit is not available currently in guppy")
                    # for j in py(focus_qubits):
                    #     measure_dirty(q[j]) # repeated measurement has not yet been added to guppy
                barrier(q)
                

            for i in range(comptime(n_qubits)): # return all qubits to the |0> state.  
                x(q[i])
    
            
            # measure all 
            b_str = measure_array(q)
    
            # report measurement outcomes
            for b in b_str:
                result("c", b)
    
        # return the compiled program (HUGR)
        return main.compile()


    def analyze_results(self, error_bars=True, plot=True, display=True, **kwargs):
        
        
        self.marginal_results = marginalize_hists(self.n_qubits, self.probe_qubits, self.results)
        self.success_probs = []
        self.avg_success_probs = []
        for j, hists in enumerate(self.marginal_results):
            succ_probs_j = get_success_probs(hists)
            avg_succ_probs_j = get_avg_success_probs(succ_probs_j)
            self.success_probs.append(succ_probs_j)
            self.avg_success_probs.append(avg_succ_probs_j)
            
        # estimate fidelity
        self.fid_avg = [estimate_fidelity(avg_succ_probs) for avg_succ_probs in self.avg_success_probs] 
        self.mean_fid_avg = float(np.mean(self.fid_avg)) # averaged over all the qubits
        
        # compute error bars
        if error_bars == True:
            self.error_data = [compute_error_bars(hists) for hists in self.marginal_results]
            self.fid_avg_std = [data['avg_fid_std'] for data in self.error_data]
            self.mean_fid_avg_std = float(np.sqrt(sum([s**2 for s in self.fid_avg_std]))/len(self.fid_avg_std))
            
            
        if plot == True:
            self.plot_results(error_bars=error_bars, **kwargs)
            
        if display == True:
            self.display_results(error_bars=error_bars, **kwargs)
            
            
    def plot_results(self, error_bars=True, **kwargs):
        measure = self.options['measure']
        reset = self.options['reset']
        
        ylim = kwargs.get('ylim', None)
        
        title = kwargs.get('title', f'{self.protocol} Decays \n Measure = {measure}, Reset = {reset}')
        
        if measure:
            def fit_func(L,gamma):
                return 1 - L * gamma # VEC:  This needs to be connected with analytic result
        if reset:
            def fit_func(L,gamma):
                return 1 - L * gamma # VEC:  This needs to be connected with analytic result

        colors_base = ['b', 'g', 'r', 'c', 'm', 'y']
        colors = ceil( self.n_qubits / len(colors_base) ) * colors_base
        print( len(colors) )        
        
        x = self.seq_lengths
        xfit = np.linspace(x[0], x[-1], 100)
        
        for j, avg_succ_probs in enumerate(self.avg_success_probs):
            
            ind_probe = self.probe_qubits[j]
            co = colors[j]
        
            y = [avg_succ_probs[L] for L in x]
            if error_bars == False:
                yerr = None
            else:
                yerr = [self.error_data[j]['success_probs_stds'][L] for L in x]
        
            # perform best fit
            popt, _ = curve_fit(fit_func, x, y)
            yfit = fit_func(xfit, *popt)
            plt.errorbar(x, y, yerr=yerr, fmt='*', color=co, label=f'q{ind_probe}')
            plt.plot(xfit, yfit, '-', color=co)
        
        plt.title(title)
        plt.ylabel('Success Probability')
        plt.xlabel('Sequence Length')
        plt.xticks(ticks=x, labels=x)
        plt.ylim(ylim)
        plt.legend()
        plt.show()
        
        
    def display_results(self, error_bars=True, **kwargs):

        prec = kwargs.get('precision', 7)
        
        print('Average Infidelities \n' + '-'*30)
        for i, f_avg in enumerate(self.fid_avg):
            q = self.probe_qubits[i]
            message = f'qubit {q}: {round(1-f_avg, prec)}'
            if error_bars == True:
                f_std = self.error_data[i]['avg_fid_std']
                message += f' +/- {round(f_std, prec)}'
            print(message)
        avg_message = 'Qubit Average: '
        mean_infid = 1-self.mean_fid_avg
        avg_message += f'{round(mean_infid, prec)}'
        if error_bars == True:
            mean_fid_avg_std = self.mean_fid_avg_std
            avg_message += f' +/- {round(mean_fid_avg_std, prec)}'
        print('-'*30)
        print(avg_message)
            
                
def marginalize_hists(n_qubits, probe_qubits, hists):
    """ return list of hists of same length as number of qubits """
    
    
    mar_hists = []
    for q in probe_qubits:# range(n_qubits):
        hists_q = {}
        for name in hists:
            L, rep, exp_out = name[0], name[1], name[2][q]
            out = hists[name]
            mar_out = {}
            for b_str in out:
                counts = out[b_str]
                # marginalize bitstring
                mar_b_str = b_str[q]
                if mar_b_str in mar_out:
                    mar_out[mar_b_str] += counts
                elif mar_b_str not in mar_out:
                    mar_out[mar_b_str] = counts
            # append marginalized outcomes to hists
            hists_q[(L, rep, exp_out)] = mar_out
        mar_hists.append(hists_q)
    
    return mar_hists            


def get_success_probs(hists: dict):
    """ compute dictionary of surv probs for 1 qubit """
    
    
    # read in list of sequence lengths
    seq_len = list(set([sett[0] for sett in list(hists.keys())]))
    seq_len.sort()
    
    success_probs = {L:[] for L in seq_len}
    for sett in hists:
        L = sett[0]
        exp_out = sett[2]
        outcomes = hists[sett]
        shots = sum(outcomes.values())
        
        if exp_out in outcomes:
            prob = outcomes[exp_out]/shots
        else:
            prob = 0.0
        
        success_probs[L].append(prob)
        
        
    return success_probs


def get_avg_success_probs(success_probs: dict):
    
    avg_success_probs = {}
    for L in success_probs:
        avg_success_probs[L] = float(np.mean(success_probs[L]))
    
    return avg_success_probs
        
    
def estimate_fidelity(avg_success_probs):
    
    # define fit function
    def fit_func(L, a, f):
        return a*f**L + 1/2
    
    x = [L for L in avg_success_probs]
    x.sort()
        
    y = [avg_success_probs[L] for L in x]
    
    # perform best fit
    popt, _ = curve_fit(fit_func, x, y, p0=[0.4, 0.9], bounds=([0,0], [0.5,1]))
    avg_fidelity = 1 - 1*(1-popt[1])/2
    
    
    return avg_fidelity


def compute_error_bars(hists: dict):
    
    boot_hists = bootstrap(hists)
    boot_avg_succ_probs = [get_avg_success_probs(get_success_probs(b_h)) for b_h in boot_hists]
    boot_avg_fids = [estimate_fidelity(avg_succ_prob)
                     for avg_succ_prob in boot_avg_succ_probs]
    
    
    # read in seq_len and list of Paulis
    seq_len = list(boot_avg_succ_probs[0].keys())
    seq_len.sort()
    
    # process bootstrapped data
    probs_stds = {}
    for L in seq_len:
        probs_stds[L] = float(np.std([b_p[L] for b_p in boot_avg_succ_probs]))
    
    avg_fid_std = float(np.std([f for f in boot_avg_fids]))
    error_data = {'success_probs_stds':probs_stds,
                  'avg_fid_std':avg_fid_std}
    
    return error_data


def bootstrap(hists, num_resamples=100):
    """ non-parametric resampling from circuits
        parametric resampling from hists
    """
    
    # read in seq_len and input states
    seq_len = list(set([name[0] for name in hists]))
    
    boot_hists = []
    for i in range(num_resamples):
        
        # first do non-parametric resampling
        hists_resamp = {}
        for L in seq_len:
            # make list of exp names to resample from
            circ_list = []
            for name in hists:
                if name[0] == L:
                    circ_list.append(name)
            # resample from circ_list
            seq_reps = len(circ_list)
            resamp_circs = np.random.choice(seq_reps, size=seq_reps)
            for rep, rep2 in enumerate(resamp_circs):
                circ = circ_list[rep2]
                name_resamp = (L, rep, circ[2])
                outcomes = hists[circ]
                hists_resamp[name_resamp] = outcomes
        
        # do parametric resample
        boot_hists.append(bs.resample_hists(hists_resamp))
    
    return boot_hists








