# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:27:23 2025

Mirror benchmarking

@author: Karl.Mayer
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, py, result
from guppylang.std.quantum import measure_array, qubit, z, x, t, tdg
from guppylang.std.qsystem import zz_phase
from guppylang.std.qsystem.random import RNG
from guppylang.std.qsystem.utils import get_current_shot
from hugr.package import FuncDefnPointer


from experiment import Experiment
from Clifford_tools import apply_SQ_Clifford, apply_SQ_Clifford_inv
from randomized_compiling import rand_comp_rzz


class MB_Experiment(Experiment):
    
    def __init__(self, n_qubits, seq_lengths, seq_reps, **kwargs):
        super().__init__()
        self.protocol = 'Mirror Benchmarking'
        self.parameters = {'n_qubits':n_qubits,
                           'seq_lenghts':seq_lengths,
                           'seq_reps':seq_reps}
        
        self.n_qubits = n_qubits
        self.seq_lengths = seq_lengths # list of seqence lengths
        self.seq_reps = seq_reps # number of repetitions per seq len
        self.setting_labels = ('seq_len', 'seq_rep', 'surv_state')
        self.filename = kwargs.get('filename', None)
        
        # options
        self.options['permute'] = kwargs.get('permute', True) # random permutation before TQ
        self.options['SQ_type'] = kwargs.get('SQ_type', 'Clifford') # or 'SU(2)' or 'Clifford+T'
        self.options['Pauli_twirl'] = kwargs.get('Pauli_twirl', True) # Pauli randomizations
        #self.options['arbZZ'] = False
        #self.options['mix_TQ_gates'] = False
        
        
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
        
        n_qubits = self.n_qubits
        seq_len = setting[0]
        surv_state = setting[2]
        Pauli_twirl = self.options['Pauli_twirl']
        permute = self.options['permute']
        init_seed = 12345
        
        if self.options['SQ_type'] == 'Clifford+T':
            include_T_gates = True
        else:
            include_T_gates = False
        
        # generate sequence of random SQ gates
        rand_SQ_gates = []
        for _ in range(seq_len):
            SQ_gates = [int(r) for r in np.random.choice(24, size=n_qubits)]
            rand_SQ_gates.append(SQ_gates)
            
        # generate list of qubits to apply T gates to (0 means no T, 1 means T)
        rand_T_qubits = []
        for _ in range(seq_len):
            if include_T_gates:
                rand_qubits = [int(b) for b in np.random.choice(2, size=n_qubits)]
                rand_T_qubits.append(rand_qubits)
            else:
                rand_T_qubits.append([0 for q_i in range(n_qubits)])
                
    
        # generate list of random pairings in each round
        TQ_pairings = []
        for _ in range(seq_len):
            if permute == True:
                rand_order = [int(q_i) for q_i in np.random.permutation(n_qubits)]
                pairs = []
                for i in range(int(np.floor(n_qubits/2))):
                    pairs.append([rand_order[2*i], rand_order[2*i+1]])
            elif permute == False:
                pairs = [[2*i,2*i+1] for i in range(int(np.floor(n_qubits/2)))]
            TQ_pairings.append(pairs)
    
        # list of qubits to apply final X to
        final_Xs = []
        for i in range(n_qubits):
            if surv_state[i] == '1':
                final_Xs.append(i)
    
        @guppy
        def main() -> None:
            
            q = array(qubit() for _ in range(py(n_qubits)))
            rng = RNG(comptime(init_seed) + get_current_shot())
            
            for i in range(py(seq_len)):
                
                # SQ gates
                SQ_gates = py(rand_SQ_gates)[i]
                for q_i in range(py(n_qubits)):
                    gate_id = SQ_gates[q_i]
                    apply_SQ_Clifford(q[q_i], gate_id)
                
                # optional T gates
                rand_qubits = py(rand_T_qubits)[i]
                for q_i in range(py(n_qubits)):
                    if rand_qubits[q_i] == 1:
                        t(q[q_i])
    
                # TQ gates
                pairings = py(TQ_pairings)[i]
                for pair in pairings:
                    if comptime(Pauli_twirl):
                        rand_comp_rzz(q[pair[0]], q[pair[1]], rng)
                    else:
                        zz_phase(q[pair[0]], q[pair[1]], angle(0.5))
                    
    
            # inverse half of circuit
            for i in range(py(seq_len)):
    
                # TQ gates
                inv_pairings = py(TQ_pairings)[py(seq_len)-1-i]
                for pair in inv_pairings:
                    if comptime(Pauli_twirl):
                        rand_comp_rzz(q[pair[0]], q[pair[1]], rng)
                    else:
                        zz_phase(q[pair[0]], q[pair[1]], angle(0.5))
                    z(q[pair[0]])
                    z(q[pair[1]])
                    
                # optional T gates
                rand_qubits = py(rand_T_qubits)[py(seq_len)-1-i]
                for q_i in range(py(n_qubits)):
                    if rand_qubits[q_i] == 1:
                        tdg(q[q_i])
    
                # SQ gates
                inv_SQ_gates = py(rand_SQ_gates)[py(seq_len)-1-i]
                for q_i in range(py(n_qubits)):
                    gate_id = inv_SQ_gates[q_i]
                    apply_SQ_Clifford_inv(q[q_i], gate_id)
            
            
            # final X's
            for q_i in py(final_Xs):
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
        
        avg_success_probs = self.get_avg_success_probs()
        
        # define decay function
        def fit_func(L, a, b):
            return a*b**(L-1) + 1/2**n
        
        # estimate unitarity and TQ gate fidelity
        x_data = list(self.seq_lengths)
        y_data = [avg_success_probs[L] for L in x_data]
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
        seq_len = self.seq_lengths
        success_probs = {L:[] for L in seq_len}
        
        for setting in results:
            L = setting[0]
            #exp_out = self.surv_state[setting]
            exp_out = setting[2]
            outcomes = results[setting]
            p = success_probability(exp_out, outcomes)
            success_probs[L].append(p)
        
        self.success_probs = success_probs
        
        return success_probs
    
    
    def get_avg_success_probs(self):
        
        success_probs = self.get_success_probs()
        
        self.avg_success_probs = {}
        for L in success_probs:
            self.avg_success_probs[L] = float(np.mean(success_probs[L]))
        
        return self.avg_success_probs
    
    
    def compute_error_bars(self):
        
        n = self.n_qubits
        shots = sum(list(self.results.values())[0].values())
        
        # define decay function
        def fit_func(L, a, b):
            return a*b**(L-1) + 1/2**n
        
        boot_probs = bootstrap(self.success_probs, shots=shots)
        stds = {}
        for L in self.avg_success_probs:
            stds[L] = np.std([b_prob[L] for b_prob in boot_probs])
        self.avg_success_probs_stds = stds   
        boot_unitarity = []
        x = self.seq_lengths
        for b_prob in boot_probs:
            b_y = [b_prob[L] for L in x]
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
        y_data = [self.avg_success_probs[L] for L in x_data]
        # perform best fit
        popt, pcov = curve_fit(fit_func, x_data, y_data, p0=[0.9, 0.9], bounds=(0,1))
            
        if error_bars == True:
            stds = self.avg_success_probs_stds
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
        avg_succ_probs = self.avg_success_probs
        for L in self.seq_lengths:
            if error_bars == True:
                stds = self.avg_success_probs_stds
                fid_avg_std = self.fid_avg_std
                err_str = f' +/- {round(stds[L],4)}'
                err_str2 = f' +/- {round(fid_avg_std, 4)}'
            else:
                err_str, err_str2 = '', ''
            print(f'{L}: {round(avg_succ_probs[L],4)}'+err_str)
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


def bootstrap(success_probs, num_resamples=100, shots=100):
    """ succ_probs (dict): keys are sequence lengths,
                           values are lists of circuit success probs
    """
    
    # read in sequence lengths
    seq_len = list(success_probs.keys())
    seq_len.sort()
    
    boot_probs = []
    
    for samp in range(num_resamples):
        
        b_succ_probs = {}
        for L in seq_len:
            probs = success_probs[L]
            # non-parametric resample from circuits
            re_probs = [float(p) for p in np.random.choice(probs, size=len(probs))]
            # parametric sample from success probs
            re_probs = [float(n_succ/shots) for n_succ in np.random.binomial(shots, re_probs)]
            b_succ_probs[L] = re_probs
        
        # take average success probs
        b_avg_succ_probs = {L:float(np.mean(b_succ_probs[L])) for L in seq_len}
        
        boot_probs.append(b_avg_succ_probs)
    

    return boot_probs
            

def resample_outcomes(outcomes):
    
    shots = sum(list(outcomes.values()))
    b_strs = list(outcomes.keys())
    p = np.array(list(outcomes.values()))/shots # probability distribution
    r = list(np.random.choice(b_strs, size=shots, p=p))
    re_out = {b_str:r.count(b_str) for b_str in set(r)}
    
    return re_out


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
    
    


