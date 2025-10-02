# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 14:23:58 2025

Binary RB with optional mid-circuit measurements

@author: Karl.Mayer
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from guppylang import guppy
from guppylang.std.builtins import array, barrier, comptime, result
from guppylang.std.quantum import qubit
from guppylang.std.qsystem import zz_phase, measure_and_reset
from guppylang.std.qsystem.random import RNG
from guppylang.std.qsystem.utils import get_current_shot
from guppylang.std.angles import angle

from hugr.package import FuncDefnPointer

from experiment import Experiment
from analysis_tools import postselect_leakage, get_postselection_rates
from leakage_measurement import measure_and_record_leakage
from Clifford_tools import apply_SQ_Clifford, update_stab_SQ, update_stab_ZZ, Clifford_list
from randomized_compiling import rand_comp_rzz

        
        
class BinaryRB_Experiment(Experiment):
    
    def __init__(self, n_qubits, seq_lengths, seq_reps, **kwargs):
        super().__init__(**kwargs)
        self.protocol = 'Binary RB'
        self.parameters = {'n_qubits':n_qubits,
                           'seq_lengths':seq_lengths,
                           'seq_reps':seq_reps}
        self.n_qubits = n_qubits
        self.seq_lengths = seq_lengths # list of seqence lengths
        self.seq_reps = seq_reps # number of repetitions per seq len
        self.setting_labels = ('n_meas', 'seq_len', 'seq_rep')
        #self.layer_depth = kwargs.get('layer_depth', 1) # number of TQ gates per layer
        self.n_meas_per_layer = kwargs.get('n_meas_per_layer', [0])
        #self.n_TQ_per_layer = kwargs.get('n_TQ_per_layer', int(np.floor(n_qubits/2)))
        self.parameters['n_meas_per_layer'] = self.n_meas_per_layer
        #self.parameters['n_TQ_per_layer'] = self.n_TQ_per_layer
        #self.parameters['layer_depth'] = self.layer_depth
        self.stabilizers = {}
        
        # options
        self.options['barriers'] = True
        self.options['init_seed'] = kwargs.get('init_seed', 12345)
        self.options['permute'] = kwargs.get('permute', True) # random permutation before TQ
        self.options['Pauli_twirl'] = kwargs.get('Pauli_twirl', False) # Pauli randomizations
        
        
    def add_settings(self):
        
        for n_meas in self.n_meas_per_layer:
            for seq_len in self.seq_lengths:
                for rep in range(self.seq_reps):
                
                    sett = (n_meas, seq_len, rep)
                    self.add_setting(sett)
                    
                    
    def make_circuit(self, setting: tuple) -> FuncDefnPointer:
        
        n_qubits = self.n_qubits
        n_meas = setting[0]
        seq_len = setting[1]
        barriers = self.options['barriers']
        meas_leak = self.options['measure_leaked']
        #layer_depth = self.layer_depth
        twirl = self.options['Pauli_twirl']
        permute = self.options['permute']
        init_seed = self.options['init_seed']
        
        # initialize stabilizer
        init_stab = np.random.choice(['I', 'Z'], size=n_qubits, p=[0.25, 0.75])
        stab = '+'
        for q_i in range(n_qubits):
            stab = stab + str(init_stab[q_i])
        
        if n_meas > 0:
            mcmr_stab = ''
        
        # generate sequence of random SQ gates, random 2Q pairings, and mcmr qubits
        rand_SQ_gates = []
        rand_pairings = []
        mcmr_qubits = [[0] for _ in range(seq_len)]
        mcmr_rots = [[0] for _ in range(seq_len)]
        final_rots = [0 for _ in range(n_qubits)]
    
        for i in range(seq_len):
    
            # add SQ gates and update stabilizer
            SQ_gates = [int(r) for r in np.random.choice(24, size=n_qubits)]
            rand_SQ_gates.append(SQ_gates)
            C = Clifford_list(SQ_gates)
            stab = update_stab_SQ(C, stab)
    
            # add TQ gates and update stabilizer
            pairs = random_qubit_pairs(n_qubits, permute=permute)
            rand_pairings.append(pairs)
            stab = update_stab_ZZ(stab, pairs)
                
            # add MCMR
            if n_meas > 0:
                mcmr_q = [int(q_i) for q_i in np.random.choice(n_qubits, size=n_meas, replace=False)]
                mcmr_qubits[i] = mcmr_q
                mcmr_r = [0 for _ in range(n_qubits)]
                for q_i in mcmr_q:
                    # rotate stabilizer into Z basis
                    if stab[q_i+1] == 'X':
                        mcmr_r[q_i] = 1
                    elif stab[q_i+1] == 'Y':
                        mcmr_r[q_i] = 8
                        
                    # reset stabilizer Pauli on measured qubit
                    mcmr_stab = mcmr_stab + stab[q_i+1]
                    stab = stab[:q_i+1] + str(np.random.choice(['I', 'Z'], p=[0.25, 0.75])) + stab[q_i+2:]
                mcmr_rots[i] = mcmr_r
    
        # measure final stabilizer
        for q_i, basis in enumerate((stab[1:])):
            if basis == 'X':
                final_rots[q_i] = 1
            elif basis == 'Y':
                final_rots[q_i] = 8
                
        if n_meas > 0:
            self.stabilizers[setting] = (stab, mcmr_stab)
        else:
            self.stabilizers[setting] = stab
            
    
        @guppy
        def main() -> None:
            q = array(qubit() for _ in range(comptime(n_qubits)))
            rng = RNG(comptime(init_seed) + get_current_shot())
            
            for i in range(comptime(seq_len)):
                
                # SQ gates
                SQ_gates = comptime(rand_SQ_gates)[i]
                for q_i in range(comptime(n_qubits)):
                    gate_id = SQ_gates[q_i]
                    apply_SQ_Clifford(q[q_i], gate_id)
                    
                # TQ gates
                pairings = comptime(rand_pairings)[i]
                for pair in pairings:
                    if comptime(twirl):
                        rand_comp_rzz(q[pair[0]], q[pair[1]], rng)
                    else:
                        zz_phase(q[pair[0]], q[pair[1]], angle(0.5))
                
                if comptime(barriers):
                    barrier(q)
                        
                # MCMR
                if comptime(n_meas) > 0:
                    mcmr_qs = comptime(mcmr_qubits)[i]
                    mcmr_r = comptime(mcmr_rots)[i]
                    for q_i in mcmr_qs:
                        gate_id = mcmr_r[q_i]
                        apply_SQ_Clifford(q[q_i], gate_id)
                    mcmr_array = array(measure_and_reset(q[q_i]) for q_i in mcmr_qs)
                    for b_mid in mcmr_array:
                        result("c_mid", b_mid)
                
                if comptime(barriers):
                    barrier(q)
                
            # final rotations
            for q_i in range(comptime(n_qubits)):
                gate_id = comptime(final_rots)[q_i]
                if gate_id > 0:
                    apply_SQ_Clifford(q[q_i], gate_id)
            
            
            # measure
            barrier(q)
            measure_and_record_leakage(q, comptime(meas_leak))
            rng.discard()
            
    
        # return the compiled program (HUGR)
        return main.compile()
    
    
    
    def analyze_results(self, error_bars=True, plot=True, display=True,
                        **kwargs):
        
        
        if self.options['measure_leaked'] == True:
            
            results = dict(self.results)
            self.raw_results = dict(results)
            self.results = postselect_leakage(results)
            self.leakage_rates = {}
            self.leakage_stds = {}
            for n_meas in self.n_meas_per_layer:
                
                results_n_meas = {}
                for sett in results:
                    if sett[0] == n_meas:
                        results_n_meas[sett] = results[sett]
                    
                ps_rates, ps_stds = get_postselection_rates(results_n_meas, self.setting_labels)
            
                self.leakage_rates[n_meas] = {L:1-ps_rates[L] for L in ps_rates}
                self.leakage_stds[n_meas] = ps_stds
        
        
        avg_success_probs = self.get_avg_success_probs()
        self.spam_param = {}
        self.layer_fidelity = {}
        
        # compute fit params
        x = list(self.seq_lengths)
        for n_meas in self.n_meas_per_layer:
            y = [self.avg_success_probs[n_meas][L] for L in x]
            # perform best fit
            try:
                spam_param, layer_fid = decay_fit_params(x, y)
                self.spam_param[n_meas] = spam_param
                self.layer_fidelity[n_meas] = layer_fid
                if n_meas ==0:
                    self.fid_avg = effective_TQ_fidelity(layer_fid, self.n_qubits)
            except:
                continue
        
        # estimate MCMR error
        if len(self.n_meas_per_layer) > 1:
            a, b = estimate_MCMR_params(self.layer_fidelity)
            self.MCMR_error = 2*(1-b)/3
            
            
        if error_bars:
            self.compute_error_bars(**kwargs)
        
        if plot:
            self.plot_results(**kwargs)
        
        if display:
            if plot:
                print('Max depth with success > 2/3')
                for n_meas in self.n_meas_per_layer:
                    print(f'MCMR/layer = {n_meas}: {self.effective_depth[n_meas]}')
            
            if 0 in self.n_meas_per_layer:
                message1 = f'Effective TQ avg fidelity: {round(self.fid_avg,5)}'
                if error_bars:
                    message1 += f' +/- {round(self.fid_avg_std,5)}'
                print(message1)
            
            if len(self.n_meas_per_layer) > 1:
                message2 = f'Effective MCMR error: {round(self.MCMR_error,5)}'
                if error_bars:
                    message2 += f' +/- {round(self.MCMR_error_std,5)}'
                print(message2)
        
        # leakage analysis
        if self.options['measure_leaked'] == True:
            if display:
                print('\nLeakage rates:')
                for n_meas in self.leakage_rates:
                    for L in self.leakage_rates[n_meas]:
                        rate = self.leakage_rates[n_meas][L]
                        std = self.leakage_stds[n_meas][L]
                        print(f'n_meas={n_meas}, L={L}: {round(rate, 3)} +/- {round(std, 3)}')
        
                
    def get_success_probs(self):
        
        results = self.results
        
        # read in list of sequence lengths
        n_meas_per_layer = self.n_meas_per_layer
        seq_lengths = self.seq_lengths
        success_probs = {n_meas:{seq_len:[] for seq_len in seq_lengths}
                         for n_meas in n_meas_per_layer}
        
        for setting in results:
            n_meas = setting[0]
            seq_len = setting[1]
            outcomes = results[setting]
            if n_meas == 0:
                stab = self.stabilizers[setting]
                p = success_probability(outcomes, stab)
            elif n_meas > 0:
                stab, mcmr_stab = self.stabilizers[setting]
                p = success_probability(outcomes, stab,
                                        mcmr_stab=mcmr_stab)
            success_probs[n_meas][seq_len].append(p)
        
        self.success_probs = success_probs
        
        return success_probs
    
    
    def get_avg_success_probs(self):
        
        success_probs = self.get_success_probs()
        
        avg_success_probs = {}
        for n_meas in self.n_meas_per_layer:
            avg_success_probs[n_meas] = {}
            for seq_len in self.seq_lengths:
                avg_success_probs[n_meas][seq_len] = float(np.mean(success_probs[n_meas][seq_len]))
        
        self.avg_success_probs = avg_success_probs
        
        return avg_success_probs
    
    
    def compute_error_bars(self, **kwargs):
        
        num_resamples = kwargs.get('num_resamples', 100)
        
        succ_probs = self.success_probs
        shots = sum(list(self.results.values())[0].values())
        #shots = self.shots
        n = self.n_qubits
        n_meas_per_layer = self.n_meas_per_layer
        
        boot_avg_succ_probs = {n_m:{L:[] for L in self.seq_lengths} for n_m in n_meas_per_layer}
        for r in range(num_resamples):
            for n_m in n_meas_per_layer:
                for L in self.seq_lengths:
                    # first do non-parametric boostrap
                    probs = succ_probs[n_m][L]
                    b_probs = np.random.choice(probs, size=len(probs))
                    # then do parametric resample
                    b_probs2 = [np.random.binomial(shots, p)/shots for p in b_probs]
                    b_avg_probs = np.mean(b_probs2)
                    boot_avg_succ_probs[n_m][L].append(b_avg_probs)
        
        self.avg_success_stds = {n_m:{} for n_m in n_meas_per_layer}
        for n_m in n_meas_per_layer:
            for L in succ_probs[n_m]:
                self.avg_success_stds[n_m][L] = float(np.std(boot_avg_succ_probs[n_m][L]))
    
        # compute error bars for spam param and layer fidelity
        x = list(self.seq_lengths)
        boot_spam_params = {n_m:[] for n_m in n_meas_per_layer}
        boot_layer_fids = {n_m:[] for n_m in n_meas_per_layer}
        if 0 in n_meas_per_layer:
            boot_avg_fid = []
        for n_m in n_meas_per_layer:
            for r in range(num_resamples):
                b_y = [boot_avg_succ_probs[n_m][L][r] for L in x]
                b_spam_param, b_layer_fid = decay_fit_params(x, b_y)
                boot_spam_params[n_m].append(b_spam_param)
                boot_layer_fids[n_m].append(b_layer_fid)
                if n_m == 0:
                    b_avg_fid = effective_TQ_fidelity(b_layer_fid, n)
                    boot_avg_fid.append(b_avg_fid)
        if len(self.n_meas_per_layer) > 1:
            boot_MCMR_error = []
            for i in range(num_resamples):
                b_layer_fid = {n_m: boot_layer_fids[n_m][i] for n_m in self.n_meas_per_layer}
                boot_MCMR_error.append(2*(1-estimate_MCMR_params(b_layer_fid)[1])/3)
            self.MCMR_error_std = float(np.std(boot_MCMR_error))
            
        
        self.spam_param_std = {n_m:float(np.std(boot_spam_params[n_m])) for n_m in n_meas_per_layer}
        self.layer_fidelity_std = {n_m:float(np.std(boot_layer_fids[n_m])) for n_m in n_meas_per_layer}
        if 0 in n_meas_per_layer:
            self.fid_avg_std = float(np.std(boot_avg_fid))
            
        
        
    def plot_results(self, error_bars=True, **kwargs):
        
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        n = self.n_qubits
        self.effective_depth = {}
        
        def fit_func(L, a, b):
            return a*b**L
        
        colors = ['g', 'b', 'r', 'c', 'm', 'y']
        
        x = list(self.seq_lengths)
        for i, n_meas in enumerate(self.n_meas_per_layer):
            co = colors[i]
            y = [2*self.avg_success_probs[n_meas][L]-1 for L in x] # polarization
            
            if error_bars == True:
                yerr = [2*self.avg_success_stds[n_meas][L] for L in x]
            else:
                yerr = None
                
            # perform best fit
            plt.errorbar(x, y, yerr=yerr, fmt=co+'o', label=f'{n_meas} meas/layer')
            try:
                popt, pcov = curve_fit(fit_func, x, y, p0=[0.9, 0.9])
                self.effective_depth[n_meas] = effective_depth(popt)
                if xlim:
                    xfit = np.linspace(xlim[0],xlim[1],100)
                else:
                    xfit = np.linspace(x[0]-2,x[-1]+2,100)
                yfit = fit_func(xfit, *popt)
                
                plt.errorbar(xfit, yfit, fmt=co+'-')
            except:
                continue
        
        plt.xticks(x)
        plt.xlabel('Sequence Length')
        plt.ylabel('Polarization')
        plt.ylim(ylim)
        plt.legend()
        plt.title(f'N={n} binary RB Results' )
        plt.show()
        
        
        # plot layer fidelity versus MCMR number
        if len(self.n_meas_per_layer) > 1:
            
            x_data = self.n_meas_per_layer
            y_data = [float(self.layer_fidelity[n_meas]) for n_meas in x_data]
            yerr = [float(self.layer_fidelity_std[n_meas]) for n_meas in x_data]
            
            xfit = np.linspace(x_data[0],x_data[-1],100)
            popt, pcov = curve_fit(fit_func, x_data, y_data)
            yfit = fit_func(xfit, *popt)

            plt.errorbar(x_data, y_data, yerr=yerr, fmt='o', color='g')
            plt.plot(xfit, yfit, color='g')
            plt.xticks(ticks=x_data, labels=x_data)
            plt.xlabel('Meas/Layer')
            plt.ylabel('Layer Fidelity')
            #plt.ylim(0.94,0.98)
            plt.show()
                

# circuit building functions

def random_qubit_pairs(n, permute=True):
    """ return random list of qubit pairs: ex. [[1,3], [0,2]]
        permute: if True, perform random qubit permutations
                 if False, applies TQ gates to nearest-neighbor pairs
    """
    
    if permute == True:
        r = [int(q_i) for q_i in list(np.random.permutation(n))]    
        qubit_pairs = []
        for i in range(int(np.floor(n/2))):
            q0 = r[2*i]
            q1 = r[2*i+1]
            # sort
            pair = [min([q0, q1]), max([q0, q1])]
            qubit_pairs.append(pair)
    
    elif permute == False:
        qubit_pairs = [[2*i,2*i+1] for i in range(int(n/2))]
    
    return qubit_pairs


# Analysis functions

def success_probability(outcomes, stab, mcmr_stab=''):
    
    shots = sum(outcomes.values())
    #n_qubits = len(stab)-1
    
    p = 0.0
    if stab[0] == '+':
        ideal_parity = 1
    elif stab[0] == '-':
        ideal_parity = -1
    
    full_stab = mcmr_stab + stab[1:]
    for b_str in outcomes:
        parity = 1
        for j, Pauli in enumerate(full_stab):
            if Pauli in ['X', 'Y', 'Z'] and b_str[j] == '1':
                parity *= -1
        if parity == ideal_parity:
            p += outcomes[b_str]/shots
    
    return round(p,8)


def decay_fit_func(L, a, b):
    return a*b**L


def decay_fit_params(x, y):
    
    # y is success probability, convert to polarizaltion
    y_pol = [2*y_succ-1 for y_succ in y]
    
    # perform best fit
    popt, pcov = curve_fit(decay_fit_func, x, y_pol, p0=[0.9, 0.9])
    spam_param = float(popt[0])
    layer_fid = float(popt[1])
        
    return spam_param, layer_fid


def effective_TQ_fidelity(layer_fid, n_qubits):
    """ compute effective TQ average fidelity from layer fidelity """
    
    F = layer_fid
    n_pairs = np.floor(n_qubits/2)
    F_pro_TQ = float(F**(1/(n_pairs)))
    F_avg_TQ = (4*F_pro_TQ+1)/5
    
    return F_avg_TQ


def effective_depth(popt):
    """ using fit function a*b**L+1/2 """
    
    a, b = popt[0], popt[1]
    L = 0
    
    while a*b**L+1/2 > 2/3:
        L += 1
        
    return L


def estimate_MCMR_params(layer_fidelity: dict):
    
    x_data = [int(n_meas) for n_meas in layer_fidelity]
    y_data = [float(layer_fidelity[n_meas]) for n_meas in layer_fidelity]
    
    popt, pcov = curve_fit(decay_fit_func, x_data, y_data)
    a, b = float(popt[0]), float(popt[1])
    
    return a, b
    
    

        
                

