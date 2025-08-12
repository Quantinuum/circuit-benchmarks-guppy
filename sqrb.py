# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:17:43 2025

Single qubit Clifford randomized benchmarking

@author: Karl.Mayer
"""

import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

from guppylang import guppy
from guppylang.std.builtins import array, barrier, comptime
from guppylang.std.quantum import qubit, x
from hugr.package import FuncDefnPointer

from analysis_tools import postselect_leakage, get_postselection_rates
from experiment import Experiment
from Clifford_tools import apply_SQ_Clifford
from leakage_measurement import measure_and_record_leakage
import bootstrap as bs


# load SQ Clifford group
module_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(module_dir, 'SQ_Clifford_group.p')
with open(file_path, 'rb') as f:
    SQ_Clifford_group = pickle.load(f)
    
Clifford_group_list = list(SQ_Clifford_group.keys())


class SQRB_Experiment(Experiment):
    
    def __init__(self, n_qubits, seq_lengths, seq_reps, **kwargs):
        super().__init__(**kwargs)
        self.protocol = 'SQRB'
        self.parameters = {'n_qubits':n_qubits,
                           'seq_lengths':seq_lengths,
                           'seq_reps':seq_reps}
        
        self.n_qubits = n_qubits
        self.seq_lengths = seq_lengths
        self.seq_reps = seq_reps
        self.setting_labels = ('seq_len', 'seq_rep', 'surv_state')
        
        self.options['SQ_type'] = 'Clifford'
        self.options['transport'] = kwargs.get('transport', False)
        
        
    def add_settings(self):
        
        for seq_len in self.seq_lengths:
            for rep in range(self.seq_reps):
                
                # choose random survival state
                surv_state = ''
                for _ in range(self.n_qubits):
                    surv_state += str(np.random.choice(['0', '1']))
                
                sett = (seq_len, rep, surv_state)
                self.add_setting(sett)
        
    
    def make_circuit(self, setting: tuple) -> FuncDefnPointer:
        """ 
        seq_len (int): number of Cliffords in circuit
        surv_state (str): expected outcome
        """
        
        seq_len = setting[0]
        surv_state = setting[2]
        meas_leak = self.options['measure_leaked']
        n_qubits = self.n_qubits
        
        assert n_qubits == len(surv_state), "len(surv_state) must equal n_qubits"
    
        Cliff_U_list = [np.diag([1,1]) for q in range(n_qubits)]
        gate_list = []
    
        for i in range(seq_len):
            
            rand_Cliffords = [str(g) for g in np.random.choice(Clifford_group_list, size=n_qubits)]
            gate_list.append([Clifford_group_list.index(C) for C in rand_Cliffords])
            
            # update sequence Clifford for qubit q_i
            for q_i in range(n_qubits):
                C = rand_Cliffords[q_i]
                Cliff_U_list[q_i] = SQ_Clifford_group[C] @ Cliff_U_list[q_i]
    
        # inverse Cliffords
        inverse_gates = []
        for q_i in range(n_qubits):
            U = Cliff_U_list[q_i]
    
            # find inverse
            for g_inv in Clifford_group_list:
                V = SQ_Clifford_group[g_inv]
                dist = 1 - (np.abs(np.trace(U @ V))/2)**2
                if dist < 10**(-8):
                    break
            
            inverse_gates.append(Clifford_group_list.index(g_inv))
        gate_list.append(inverse_gates)
    
        # list of qubits to apply final X to
        final_Xs = []
        for i in range(n_qubits):
            if surv_state[i] == '1':
                final_Xs.append(i)
        
        @guppy
        def main() -> None:
    
            q = array(qubit() for _ in range(comptime(n_qubits)))
    
            for gates in comptime(gate_list):
                for q_i in range(comptime(n_qubits)):
                    gate_index = gates[q_i]
                    apply_SQ_Clifford(q[q_i], gate_index)
                
                barrier(q)
    
            # final X's
            for q_i in comptime(final_Xs):
                x(q[q_i])
            
            # measure
            measure_and_record_leakage(q, comptime(meas_leak))
    
        # return the compiled program (HUGR)
        return main.compile()
    
    
    # Analysis methods
    
    def analyze_results(self, error_bars=True, plot=True, display=True, **kwargs):
        
        
        marginal_results = marginalize_hists(self.n_qubits, self.results)
        
        # postselect leakage
        if self.options['measure_leaked'] == True:
            self.marginal_results = [postselect_leakage(mar_re) for mar_re in marginal_results]
            self.postselection_rates = []
            self.postselection_rates_stds = []
            for mar_re in marginal_results:
                ps_rates, ps_stds = get_postselection_rates(mar_re, self.setting_labels)
                self.postselection_rates.append(ps_rates)
                self.postselection_rates_stds.append(ps_stds)
        else:
            self.marginal_results = marginal_results
            
        self.success_probs = []
        self.avg_success_probs = []
        for j, hists in enumerate(self.marginal_results):
            succ_probs_j = get_success_probs(hists)
            avg_succ_probs_j = get_avg_success_probs(succ_probs_j)
            self.success_probs.append(succ_probs_j)
            self.avg_success_probs.append(avg_succ_probs_j)
            
        # estimate fidelity
        self.fid_avg = [estimate_fidelity(avg_succ_probs) for avg_succ_probs in self.avg_success_probs]
        self.mean_fid_avg = np.mean(self.fid_avg)
        
        # compute error bars
        if error_bars == True:
            self.error_data = [compute_error_bars(hists) for hists in self.marginal_results]
            self.fid_avg_std = [data['avg_fid_std'] for data in self.error_data]
            self.mean_fid_avg_std = np.sqrt(sum([s**2 for s in self.fid_avg_std]))/len(self.fid_avg_std)
            
            
        if plot == True:
            self.plot_results(error_bars=error_bars, **kwargs)
            
        if display == True:
            self.display_results(error_bars=error_bars, **kwargs)
            
        # leakage analysis
        if self.options['measure_leaked'] == True:
            self.plot_postselection_rates(display=display, **kwargs)
            
            
    def plot_results(self, error_bars=True, **kwargs):
        
        ylim = kwargs.get('ylim', None)
        
        title = kwargs.get('title', f'{self.protocol} Decays')
        
        # define fit function
        def fit_func(L, a, f):
            return a*f**L+1/2
        
        # Create a colormap
        cmap = cm.turbo

        # Normalize color range from 0 to num_lines-1
        cnorm = mcolors.Normalize(vmin=0, vmax=self.n_qubits-1)
        
        x = self.seq_lengths
        xfit = np.linspace(x[0], x[-1], 100)
        
        for j, avg_succ_probs in enumerate(self.avg_success_probs):
        
            y = [avg_succ_probs[L] for L in x]
            if error_bars == False:
                yerr = None
            else:
                yerr = [self.error_data[j]['success_probs_stds'][L] for L in x]
        
            # perform best fit
            popt, pcov = curve_fit(fit_func, x, y, p0=[0.4, 0.9], bounds=([0,0], [0.5,1]))
            yfit = fit_func(xfit, *popt)
            plt.errorbar(x, y, yerr=yerr, fmt='o', color=cmap(cnorm(j)), label=f'q{j}')
            plt.plot(xfit, yfit, '-', color=cmap(cnorm(j)))
        
        plt.title(title)
        plt.ylabel('Success Probability')
        plt.xlabel('Sequence Length')
        plt.xticks(ticks=x, labels=x)
        plt.ylim(ylim)
        if self.n_qubits <= 16:
            plt.legend()
        plt.show()
        
    
    def plot_postselection_rates(self, display=True, **kwargs):
        
        ylim = kwargs.get('ylim2', None)
        title = kwargs.get('title2', f'{self.protocol} Leakage Postselection Rates')
        
        # define fit function
        def fit_func(L, a, f):
            return a*f**L
        
        # Create a colormap
        cmap = cm.turbo

        # Normalize color range from 0 to num_lines-1
        cnorm = mcolors.Normalize(vmin=0, vmax=self.n_qubits-1)
        
        x = self.seq_lengths
        xfit = np.linspace(x[0], x[-1], 100)
        leakage_rates = []
        leakage_stds = []
        
        for j, ps_rates in enumerate(self.postselection_rates):
        
            y = [ps_rates[L] for L in x]
            yerr = [self.postselection_rates_stds[j][L] for L in x]
        
            # perform best fit
            popt, pcov = curve_fit(fit_func, x, y, p0=[0.4, 0.9], bounds=([0,0], [1,1]), sigma=yerr)
            leakage_rates.append(1-popt[1])
            leakage_stds.append(float(np.sqrt(pcov[1][1])))
            yfit = fit_func(xfit, *popt)
            plt.errorbar(x, y, yerr=yerr, fmt='o', color=cmap(cnorm(j)), label=f'q{j}')
            plt.plot(xfit, yfit, '-', color=cmap(cnorm(j)))
        
        plt.title(title)
        plt.ylabel('Postselection Rate')
        plt.xlabel('Sequence Length')
        plt.xticks(ticks=x, labels=x)
        plt.ylim(ylim)
        if self.n_qubits <= 16:
            plt.legend()
        plt.show()
        
        self.leakage_rates = leakage_rates
        self.leakage_rates_stds = leakage_stds
        self.mean_leakage_rate = float(np.mean(leakage_rates))
        self.mean_leakage_std = float(np.sqrt(sum([s**2 for s in leakage_stds]))/len(leakage_stds))
        
        if display:
            leak_rate = self.mean_leakage_rate
            leak_std = self.mean_leakage_std
            print(f'Qubit average leakge rate: {round(leak_rate, 6)} +/- {round(leak_std, 6)}')
        
        
        
    def display_results(self, error_bars=True, **kwargs):
        
        prec = kwargs.get('precision', 6)
        verbose = kwargs.get('verbose', True)
        
        if verbose:
            print('Average Infidelities\n' + '-'*30)
            for q, f_avg in enumerate(self.fid_avg):
                message = f'qubit {q}: {round(1-f_avg, prec)}'
                if error_bars == True:
                    f_std = self.error_data[q]['avg_fid_std']
                    message += f' +/- {round(f_std, prec)}'
                print(message)
        avg_message = 'Qubit Average: '
        mean_infid = 1-self.mean_fid_avg
        avg_message += f'{round(mean_infid,prec)}'
        if error_bars == True:
            mean_fid_avg_std = self.mean_fid_avg_std
            avg_message += f' +/- {round(mean_fid_avg_std, prec)}'
        print('-'*30)
        print(avg_message)
        
        
# analysis functions

def marginalize_hists(n_qubits, hists):
    """ return list of hists of same length as number of qubits """
    
    
    mar_hists = []
    for q in range(n_qubits):
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
    popt, pcov = curve_fit(fit_func, x, y, p0=[0.4, 0.9], bounds=([0,0], [0.5,1]))
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
        probs_stds[L] = np.std([b_p[L] for b_p in boot_avg_succ_probs])
    
    avg_fid_std = np.std([f for f in boot_avg_fids])
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




