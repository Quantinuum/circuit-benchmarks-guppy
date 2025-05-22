# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:51:15 2025

Fully random MB

@author: Karl.Mayer
"""


import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from guppylang import guppy
from guppylang.std.builtins import array, comptime, py, result
from guppylang.std.quantum import measure_array, qubit, x, z, t, tdg
from guppylang.std.qsystem.random import RNG
from guppylang.std.qsystem.utils import get_current_shot
from hugr.package import FuncDefnPointer


from experiment import Experiment
from Clifford_tools import apply_SQ_Clifford, apply_SQ_Clifford_inv
from randomized_compiling import rand_comp_rzz

# load SQ Clifford group
module_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(module_dir, 'SQ_Clifford_group.p')
with open(file_path, 'rb') as f:
    SQ_Clifford_group = pickle.load(f)
    
Clifford_group_list = list(SQ_Clifford_group.keys())


class FullyRandomMB_Experiment(Experiment):
    
    def __init__(self, n_qubits, seq_lengths, **kwargs):
        super().__init__()
        self.protocol = 'Fully Random Mirror Benchmarking'
        self.parameters = {'n_qubits':n_qubits,
                           'seq_lenghts':seq_lengths}
        
        self.n_qubits = n_qubits
        self.seq_lengths = seq_lengths # list of seqence lengths
        self.setting_labels = ('seq_len', 'surv_state')
        
        # optional keyword arguments
        self.layer_depth = kwargs.get('layer_depth', 1) # number of TQ gates per layer
        self.parameters['layer_depth'] = self.layer_depth
        self.permute = kwargs.get('permute', True) # random permutation before TQ
        self.SQ_type = kwargs.get('SQ_type', 'Clifford') # or 'SU(2)' or 'Clifford+T'
        self.Pauli_twirl = kwargs.get('Pauli_twirl', True) # Pauli randomizations
        self.filename = kwargs.get('filename', None)
        
        # options
        self.options['init_seed'] = kwargs.get('init_seed', 12345)
        self.options['permute'] = self.permute
        self.options['SQ_type'] = self.SQ_type
        self.options['Pauli_twirl'] = self.Pauli_twirl
        #self.options['arbZZ'] = False
        #self.options['mix_TQ_gates'] = False
        
        
    def add_settings(self):
        
        for seq_len in self.seq_lengths:
                
            # choose random survival state
            surv_state = ''
            for _ in range(self.n_qubits):
                surv_state += str(np.random.choice(['0', '1']))
            
            sett = (seq_len, surv_state)
            self.add_setting(sett)
                
                
    def make_circuit(self, setting: tuple) -> FuncDefnPointer:
        
        n_qubits = self.n_qubits
        seq_len = setting[0]
        surv_state = setting[1]
        n_pairs = int(np.floor(n_qubits/2))
        init_seed = self.options['init_seed']
        
        if self.options['SQ_type'] == 'Clifford+T':
            include_T_gates = True
        else:
            include_T_gates = False
    
        # list of qubits to apply final X to
        final_Xs = []
        for i in range(n_qubits):
            if surv_state[i] == '0':
                final_Xs.append(0)
            elif surv_state[i] == '1':
                final_Xs.append(1)
        
        @guppy
        def rand_SQ_gates(rng: RNG) -> array[int, comptime(n_qubits)]:
            """Returns an array of random integers between 0 and 23. """
        
            return array(rng.random_int_bounded(24) for _ in range(comptime(n_qubits)))
    
        @guppy
        def rand_SQ_layers(rng: RNG) -> array[array[int, comptime(n_qubits)], comptime(seq_len)]:
            """ Returns an array of arrays of random intergers between 0 and 23. """
            
            comptime(n_qubits) # "use" n_qubits value
            
            return array(rand_SQ_gates(rng) for _ in range(comptime(seq_len)))
        
        @guppy
        def rand_T_qubits(rng:RNG) -> array[array[int, comptime(n_qubits)], comptime(seq_len)]:
            
            return array(array(rng.random_int_bounded(2) for _ in range(comptime(n_qubits))) for _ in range(comptime(seq_len)))
    
        @guppy
        def get_random_order(rng: RNG) -> array[int, comptime(n_qubits)]:
            """ Randomly order the n_qubits """
            
            arr = array(q_i for q_i in range(comptime(n_qubits)))
            for k in range(comptime(n_qubits)):
                i = comptime(n_qubits) - 1 - k
                j = rng.random_int_bounded(i + 1)
                if i != j:
                    mem_swap(arr[i], arr[j])
    
            return arr
    
        @guppy
        def rand_TQ_layers(rng: RNG) -> array[array[int, comptime(n_qubits)], comptime(seq_len)]:
            """ Returns an array of random orderings of qubit indices """
            
            comptime(n_qubits) # "use" n_qubits value
            
            return array(get_random_order(rng) for _ in range(comptime(seq_len)))
            
        
        @guppy
        def main() -> None:
    
            rng = RNG(comptime(init_seed) + get_current_shot())
            
            # create initial array of gate indices
            SQ_gate_indices = rand_SQ_layers(rng)
            TQ_gate_indices = rand_TQ_layers(rng)
            if comptime(include_T_gates):
                T_gate_qubits = rand_T_qubits(rng)
            else:
                T_gate_qubits = array(array(0 for _ in range(py(n_qubits))) for _ in range(py(seq_len)))
    
            q = array(qubit() for _ in range(py(n_qubits)))
    
            # front half of circuit
            for i in range(py(seq_len)):
                
                # SQ_gates
                for q_i in range(py(n_qubits)):
                    gate_id = SQ_gate_indices[i][q_i]
                    apply_SQ_Clifford(q[q_i], gate_id)
                
                # optional T gates
                for q_i in range(py(n_qubits)):
                    if T_gate_qubits[i][q_i] == 1:
                        t(q[q_i])
                
                # TQ gates
                for j in range(py(n_pairs)):
                    q0 = TQ_gate_indices[i][2*j]
                    q1 = TQ_gate_indices[i][2*j+1]
                    rand_comp_rzz(q[q0], q[q1], rng)
    
    
            # inverse half of circuit
            for i in range(py(seq_len)):
    
                # TQ_gates
                for j in range(py(n_pairs)):
                    q0 = TQ_gate_indices[py(seq_len)-1-i][2*j]
                    q1 = TQ_gate_indices[py(seq_len)-1-i][2*j+1]
                    rand_comp_rzz(q[q0], q[q1], rng)
                    z(q[q0])
                    z(q[q1])
                    
                # optional T gates
                for q_i in range(py(n_qubits)):
                    if T_gate_qubits[py(seq_len)-1-i][q_i] == 1:
                        tdg(q[q_i])
                
                # SQ_gates
                for q_i in range(py(n_qubits)):
                    gate_id = SQ_gate_indices[py(seq_len)-1-i][q_i]
                    apply_SQ_Clifford_inv(q[q_i], gate_id)
    
            # final X's
            for q_i in range(py(n_qubits)):
                if py(final_Xs)[q_i] == 1:
                    x(q[q_i])
            
            # measure
            b_str = measure_array(q)
            rng.discard()
    
            # report measurement outcomes
            for b in b_str:
                result("c", b)
    
        # return the compiled program (HUGR)
        return main.compile()
    
    
    # Analysis methods
    
    def analyze_results(self, error_bars=True, plot=True, display=True, **kwargs):
        
        
        n = self.n_qubits
        
        success_probs = self.get_success_probs()
        
        # define decay function
        def fit_func(L, a, b):
            return a*b**(L-1) + 1/2**n
        
        # estimate unitarity and TQ gate fidelity
        x_data = list(self.seq_lengths)
        y_data = [success_probs[L] for L in x_data]
        # perform best fit
        popt, pcov = curve_fit(fit_func, x_data, y_data, p0=[0.9, 0.9], bounds=(0,1))
        unitarity = popt[1]
        self.unitarity = unitarity
        self.fid_avg = unitarity2TQ_fidelity(unitarity, n)
        self.effective_depth = effective_depth(popt[0], popt[1], n)
        
        # bootstrap for error bars
        if error_bars == True:
            self.compute_error_bars()
        
        # plot results
        if plot == True:
            self.plot_results(error_bars=error_bars, **kwargs)
        
        # display results
        if display == True:
            self.display_results(error_bars=error_bars)
            
        # save results
        self.save()
            
    
    def get_success_probs(self):
        
        results = self.results
        
        # read in list of sequence lengths
        #seq_len = self.seq_lengths
        
        success_probs = {}
        for setting in results:
            L = setting[0]
            #exp_out = self.surv_state[setting]
            exp_out = setting[1]
            outcomes = results[setting]
            p = success_probability(exp_out, outcomes)
            success_probs[L] = p
        
        self.success_probs = success_probs
        
        return success_probs
    
    
    def compute_error_bars(self, num_resamples=100):
        
        n = self.n_qubits
        shots = sum(list(self.results.values())[0].values())
        
        # define decay function
        def fit_func(L, a, b):
            return a*b**(L-1) + 1/2**n
        
        stds = {}
        for L in self.success_probs:
            p = self.success_probs[L]
            if p < 1.0:
                p_std = float(np.sqrt(p*(1-p)/shots))
            elif p == 1.0:
                p_eff = shots/(shots+2) # rule of 2
                p_std = float(np.sqrt(p_eff*(1-p_eff)/shots))
            
            
            stds[L] = p_std
        self.success_probs_stds = stds   
        boot_unitarity = []
        x = self.seq_lengths
        
        # bootstrap
        boot_probs = []
        for _ in range(num_resamples):
            b_probs = {}
            for L in self.seq_lengths:
                p = self.success_probs[L]
                if p < 1.0:
                    p_eff = p
                elif p == 1.0:
                    p_eff = shots/(shots+2) # rule of 2
                b_probs[L] = float(np.random.binomial(shots, p_eff)/shots)
            boot_probs.append(b_probs)
                
            
        for b_probs in boot_probs:
            b_y = [b_probs[L] for L in x]
            # best fit the bootstrapped success probabilities
            b_popt, b_pcov = curve_fit(fit_func, x, b_y, p0=[0.9, 0.9], bounds=(0,1))
            boot_unitarity.append(b_popt[1])
        
        self.unitarity_std = float(np.std(boot_unitarity))
        
        # estimate F_avg_std
        self.fid_avg_std = float(np.std([unitarity2TQ_fidelity(u, n) for u in boot_unitarity]))
    
    
    def plot_results(self, error_bars=True, **kwargs):
        
        ylim = kwargs.get('ylim', None)
        
        n = self.n_qubits
        
        # define decay function
        def fit_func(L, a, b):
            return a*b**(L-1) + 1/2**n
        
        x_data = list(self.seq_lengths)
        y_data = [self.success_probs[L] for L in x_data]
        # perform best fit
        popt, pcov = curve_fit(fit_func, x_data, y_data, p0=[0.9, 0.9], bounds=(0,1))
            
        if error_bars == True:
            stds = self.success_probs_stds
            yerr = [stds[L] for L in x_data]
        elif error_bars == False:
            yerr = None
        
        
        xfit = np.linspace(x_data[0]-2,x_data[-1]+2,100)
        yfit = fit_func(xfit, *popt)
        
        plt.errorbar(x_data, y_data, yerr=yerr, fmt='bo')
        plt.errorbar(xfit, yfit, fmt='b-')
        
        plt.xticks(x_data)
        plt.xlabel('Sequence Length')
        plt.ylabel('Average Success')
        plt.ylim(ylim)
        #plt.legend()
        plt.title(f'N={n} Mirror Benchmarking Results' )
        plt.show()
        
    
    def display_results(self, error_bars=True):
        
        print('Success Probabilities\n' + '-'*22)
        succ_probs = self.success_probs
        for L in self.seq_lengths:
            if error_bars == True:
                stds = self.success_probs_stds
                fid_avg_std = self.fid_avg_std
                err_str = f' +/- {round(stds[L],4)}'
                err_str2 = f' +/- {round(fid_avg_std, 4)}'
            else:
                err_str, err_str2 = '', ''
            print(f'{L}: {round(succ_probs[L],4)}'+err_str)
        eff_depth = self.effective_depth
        print(f'\nMax circuit depth with survival > 2/3: {eff_depth}')
        print(f'\nTQ Average Fidelity (for depolarizing error) = {round(self.fid_avg, 4)}'+err_str2)
            
            
                
# analysis functions

def success_probability(exp_out, outcomes):
    
    shots = sum(list(outcomes.values()))
    
    if exp_out in outcomes:
        p = outcomes[exp_out]/shots
    else:
        p = 0.0
    
    return p




def true_unitarity(TQ_err, n):
    """ TQ_err: depolarizing parameter
             n: number of qubits (must be even)
    """
    
    from scipy.special import comb
    
    d = 2**n
    n_pairs = int(n/2) # number of qubit pairs
    
    # unitarity
    u = 0.0
    
    # sum over weights of Paulis
    for w in range(1,n_pairs+1):
        c1 = comb(n_pairs, w, exact=True)
        S = 0.0
        for j in range(n_pairs-w+1):
            c2 = comb(n_pairs-w, j, exact=True)
            S += c2*(TQ_err**j)*(1-TQ_err)**(n_pairs-j)
        u += (15**w)*c1*S**2/(d**2-1)
    
    return u


def unitarity2TQ_fidelity(u, n, lay_depth=1):
    """ TQ fidelity assuming TQ depolarizing error
        u : unitarity
        n : number of qubits (must be even)
        
        returns F_avg : average fidelity
    """
    
    # first, find TQ depolarizing parameter that gives correct unitarity
    tol = 10**(-6)
    finished = False
    
    # initialize lower and upper limits on depolarizing parameter
    p_0, p_1 = 0.0, 1.0
    
    while finished == False:
        p = (p_0+p_1)/2
        u_true = true_unitarity(p, n)
        if abs(u_true-u) < tol:
            finished = True
        else:
            if u_true > u:
                p_0 = p
            if u_true < u:
                p_1 = p
        #print(p) # for debugging
    
    # convert to TQ average fidelity
    #rescale according to layer depth
    p_rs = 1 - ((1-p)**(1/lay_depth))
    F_avg = 1-3*p_rs/4
    
    return F_avg


def effective_depth(a, b, n):
    """ depth at which survival > 2/3
        a, b: fit parameters a*b^(L-1) + 1/2^n
    """
    
    L = 0
    while a*b**(L-1) + 1/2**n > 2/3:
        L += 1
    
    return 2*L
    
    
