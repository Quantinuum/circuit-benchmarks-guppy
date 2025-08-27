# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 14:23:58 2025

Binary RB with optional mid-circuit measurements

@author: Karl.Mayer
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, least_squares

from guppylang import guppy
from guppylang.std.builtins import array, barrier, comptime, result
from guppylang.std.quantum import qubit
from guppylang.std.qsystem import zz_phase, measure_and_reset
from guppylang.std.qsystem.random import RNG
from guppylang.std.qsystem.utils import get_current_shot
from guppylang.std.angles import angle

from hugr.package import FuncDefnPointer

from circuit_benchmarks_guppy.benchmarks.experiment import Experiment
from circuit_benchmarks_guppy.tools.analysis import postselect_leakage, get_postselection_rates
from circuit_benchmarks_guppy.tools.leakage_measurement import measure_and_record_leakage
from circuit_benchmarks_guppy.tools.clifford import apply_SQ_Clifford, update_stab_SQ, update_stab_ZZ, Clifford_list
from circuit_benchmarks_guppy.tools.randomized_compiling import rand_comp_rzz

        
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
        self.TQ_density = kwargs.get('TQ_density', 1.0)
        #self.n_TQ_per_layer = kwargs.get('n_TQ_per_layer', int(np.floor(n_qubits/2)))
        self.parameters['n_meas_per_layer'] = self.n_meas_per_layer
        self.parameters['TQ_density'] = self.TQ_density
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
        TQ_density = self.TQ_density
        
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
            pairs = random_qubit_pairs(n_qubits, permute=permute, TQ_density=TQ_density)
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
        
        num_resamples = kwargs.get('num_resamples', 500)
        prec = kwargs.get('precision', 5)
        
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
        
        # fix y-intercept for all decay curves
        self.layer_fidelity = {}
        if not self.n_meas_per_layer == [0]:
            spam_param, TQ_error, MCMR_error = estimate_fit_params(avg_success_probs, self.n_qubits, TQ_density=self.TQ_density)
            for n_meas in self.n_meas_per_layer:
                x = list(avg_success_probs[n_meas].keys())
                y = [avg_success_probs[n_meas][L] for L in x]
                layer_fid = decay_fit_params_fixed_spam(x, y, spam_param)
                self.layer_fidelity[n_meas] = layer_fid
            
        elif self.n_meas_per_layer == [0]:
            x = list(avg_success_probs[0].keys())
            y = [avg_success_probs[0][L] for L in x]
            spam_param, layer_fid = decay_fit_params(x, y)
            TQ_error = 1 - effective_TQ_fidelity(layer_fid, self.n_qubits, self.TQ_density)
            self.layer_fidelity[0] = layer_fid
        
        self.spam_param = spam_param
        self.TQ_error = TQ_error
        self.fid_avg = 1 - TQ_error
        
        
        # estimate MCMR error
        if len(self.n_meas_per_layer) > 1:
            a, b = estimate_MCMR_params(self.layer_fidelity)
            self.MCMR_error = 2*(1-b)/3
            
            
        if error_bars:
            self.compute_error_bars(num_resamples)
        
        if plot:
            self.plot_results(error_bars=error_bars, **kwargs)
            if len(self.n_meas_per_layer) > 1:
                self.plot_layer_fidelity(error_bars=error_bars, **kwargs)
        
        if display:
            if plot:
                try:
                    print('Max depth with success > 2/3')
                    for n_meas in self.n_meas_per_layer:
                        print(f'MCMR/layer = {n_meas}: {self.effective_depth[n_meas]}')
                except:
                    pass
            
            message1 = f'\nEffective TQ avg infidelity: {round(self.TQ_error,prec)}'
            if error_bars:
                message1 += f' +/- {round(self.TQ_error_std,prec)}'
            print(message1)
            
            if not self.n_meas_per_layer == [0]:
                message2 = f'Effective MCMR error: {round(self.MCMR_error,prec)}'
                if error_bars:
                    message2 += f' +/- {round(self.MCMR_error_std,prec)}'
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
    
    
    def compute_error_bars(self, num_resamples=500):
        
        succ_probs = self.success_probs
        shots = self.shots
        n = self.n_qubits
        n_meas_per_layer = self.n_meas_per_layer
        TQ_density = self.TQ_density
        
        # bootstrap success probabilities
        boot_avg_succ_probs = {}
        for n_m in n_meas_per_layer:
            boot_avg_succ_probs[n_m] = {}
            for L in succ_probs[n_m]:
                if len(succ_probs[n_m][L]) > 0:
                    boot_avg_succ_probs[n_m][L] = []
        
        for r in range(num_resamples):
            for n_m in n_meas_per_layer:
                for L in succ_probs[n_m]:
                    if len(succ_probs[n_m][L]) > 0:
                        # first do non-parametric boostrap
                        probs = succ_probs[n_m][L]
                        b_probs = np.random.choice(probs, size=len(probs))
                        # then do parametric resample
                        b_probs2 = [np.random.binomial(shots, p)/shots for p in b_probs]
                        b_avg_probs = np.mean(b_probs2)
                        boot_avg_succ_probs[n_m][L].append(b_avg_probs)
        
        # compute error bars for success probs
        self.avg_success_stds = {n_m:{} for n_m in n_meas_per_layer}
        for n_m in n_meas_per_layer:
            for L in boot_avg_succ_probs[n_m]:
                self.avg_success_stds[n_m][L] = float(np.std(boot_avg_succ_probs[n_m][L]))
        
        
        # compute error bars for fit parameters
        # from bootstrapped success probs
        boot_spam_params = []
        boot_TQ_error = []
        boot_layer_fids = {n_m:[] for n_m in n_meas_per_layer}
        
        if not n_meas_per_layer == [0]: 
            for r in range(num_resamples):
                b_avg_probs = {n_m:{L:boot_avg_succ_probs[n_m][L][r] for L in boot_avg_succ_probs[n_m]} for n_m in n_meas_per_layer}
                b_spam_param, b_TQ_error, MCMR_error = estimate_fit_params(b_avg_probs, self.n_qubits, TQ_density=self.TQ_density)
                boot_spam_params.append(b_spam_param)
                boot_TQ_error.append(b_TQ_error)
                for n_m in n_meas_per_layer:
                    b_x = [L for L in boot_avg_succ_probs[n_m]]
                    b_y = [boot_avg_succ_probs[n_m][L][r] for L in b_x]
                    b_layer_fid = decay_fit_params_fixed_spam(b_x, b_y, b_spam_param)
                    boot_layer_fids[n_m].append(b_layer_fid)
        
        elif n_meas_per_layer == [0]:
            for r in range(num_resamples):
                b_x = [L for L in boot_avg_succ_probs[0]]
                b_y = [boot_avg_succ_probs[0][L][r] for L in b_x]
                b_spam_param, b_layer_fid = decay_fit_params(b_x, b_y)
                boot_layer_fids[0].append(b_spam_param)
                boot_spam_params.append(b_spam_param)
                boot_TQ_error.append(1 - effective_TQ_fidelity(b_layer_fid, n, TQ_density))
                
            
        
        self.layer_fidelity_std = {n_m:float(np.std(boot_layer_fids[n_m])) for n_m in n_meas_per_layer}
        self.spam_param_std = float(np.std(boot_spam_params))
        self.TQ_error_std = float(np.std(boot_TQ_error))
        self.fid_avg_std = self.TQ_error_std
        
        # estimate MCMR error
        if len(self.n_meas_per_layer) > 1:
            boot_MCMR_error = []
            for r in range(num_resamples):
                a, b = estimate_MCMR_params({n_m:boot_layer_fids[n_m][r] for n_m in n_meas_per_layer})
                boot_MCMR_error.append(float(2*(1-b)/3))
            self.MCMR_error_std = float(np.std(boot_MCMR_error))
        
        
    def plot_results(self, error_bars=True, **kwargs):
        
        n = self.n_qubits
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        title = kwargs.get('title', f'N={n} binary RB Results' )
        
        self.effective_depth = {}
        
        a = self.spam_param
        def fit_func(L, b):
            return a*b**L
        
        colors = ['g', 'c', 'b', 'm', 'r']
        
        for i, n_meas in enumerate(self.n_meas_per_layer):
            co = colors[i]
            x = list(self.avg_success_probs[n_meas].keys())
            y = [2*self.avg_success_probs[n_meas][L]-1 for L in x] # polarization
            
            if error_bars == True:
                yerr = [2*self.avg_success_stds[n_meas][L] for L in x]
            else:
                yerr = None
                
            # perform best fit
            plt.errorbar(x, y, yerr=yerr, fmt=co+'o', label=f'{n_meas} measurements per layer')
            try:
                popt, pcov = curve_fit(fit_func, x, y, p0=[0.9])
                self.effective_depth[n_meas] = effective_depth((self.spam_param, popt[0]))
                if xlim:
                    xfit = np.linspace(xlim[0],xlim[1],100)
                else:
                    xfit = np.linspace(x[0]-2,x[-1]+2,100)
                yfit = fit_func(xfit, *popt)
                
                plt.errorbar(xfit, yfit, fmt=co+'-')
            except:
                continue
        
        if 0 in self.seq_lengths:
            xticks = self.seq_lengths
        else:
            xticks = [0] + self.seq_lengths
        plt.xticks(ticks=xticks, labels=xticks)
        plt.xlabel('Sequence Length')
        plt.ylabel('Polarization')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.grid(True, linestyle='--', which='both', axis='both')
        plt.title(title)
        plt.show()
        
        
    def plot_layer_fidelity(self, error_bars=True, **kwargs):
        
        ylim = kwargs.get('ylim2', None)
        
        x_data = self.n_meas_per_layer
        y_data = [float(self.layer_fidelity[n_meas]) for n_meas in x_data]
        if error_bars:
            yerr = [float(self.layer_fidelity_std[n_meas]) for n_meas in x_data]
        else:
            yerr = None
            
        
        xfit = np.linspace(x_data[0],x_data[-1],100)
        popt, pcov = curve_fit(decay_fit_func, x_data, y_data)
        yfit = decay_fit_func(xfit, *popt)

        plt.errorbar(x_data, y_data, yerr=yerr, fmt='o', color='g')
        plt.plot(xfit, yfit, color='g')
        plt.xticks(ticks=x_data, labels=x_data)
        plt.xlabel('Measurements per Layer')
        plt.ylabel('Layer Fidelity')
        plt.ylim(ylim)
        plt.show()
        
                

# circuit building functions

def random_qubit_pairs(n, permute=True, TQ_density=1.0):
    """ return random list of qubit pairs: ex. [[1,3], [0,2]]
        permute: if True, perform random qubit permutations
                 if False, applies TQ gates to nearest-neighbor pairs
        TQ_density: ratio of number of TQ pairs per layer to n/2
    """
    
    n_pairs = int(TQ_density*n/2)
    
    if permute == True:
        r = [int(q_i) for q_i in list(np.random.permutation(n))]    
        qubit_pairs = []
        for i in range(n_pairs):
            q0 = r[2*i]
            q1 = r[2*i+1]
            # sort
            pair = [min([q0, q1]), max([q0, q1])]
            qubit_pairs.append(pair)
    
    elif permute == False:
        qubit_pairs = [[2*i,2*i+1] for i in range(n_pairs)]
    
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


def decay_fit_params_fixed_spam(x, y, spam_param):
    
    a = spam_param
    
    def fit_func(L, b):
        return a*b**L
    
    # y is success probability, convert to polarizaltion
    y_pol = [2*y_succ-1 for y_succ in y]
    
    # perform best fit
    popt, pcov = curve_fit(fit_func, x, y_pol, p0=[0.9])
    layer_fid = float(popt[0])
        
    return layer_fid


def effective_TQ_fidelity(layer_fid, n_qubits, TQ_density):
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


# functions for doing least squares fit


def estimate_fit_params(avg_success_probs: dict, n_qubits: int, TQ_density=1.0):
    
    n = n_qubits
    n_meas = list(avg_success_probs.keys())
    
    # define fit model
    # a: spam param
    # b: gate error param
    # c: MCMR error param
    def residuals(params, x, y, z):
        a, b, c = params
        return a*(1-b)**(x*((TQ_density*n)//2))*(1-c)**(x*y) - z
    
    x_data, y_data, z_data = [], [], []
    
    for n_m in n_meas:
        if len(avg_success_probs[n_m]) > 0:
            seq_lengths = list(avg_success_probs[n_m].keys())
            for L in seq_lengths:
                x_data.append(L)
                y_data.append(n_m)
                y_pol = 2*avg_success_probs[n_m][L]-1 # polarization
                z_data.append(y_pol)
    
    initial_guess = [0.95, 0.003, 0.003]
    
    fit_result = least_squares(residuals, initial_guess, args=(np.array(x_data), np.array(y_data), np.array(z_data)))
    a, b, c = fit_result.x
    
    spam_param = float(a)
    TQ_err = float(4*b/5)
    MCMR_err = float(2*c/3)
    
    return spam_param, TQ_err, MCMR_err


def bootstrap_fit_params(success_probs: dict, shots: int, n_qubits: int, TQ_density=1.0, num_resamples=500):
    
    n = n_qubits
    n_meas = list(success_probs.keys())
    
    # define fit model
    # a: spam param
    # b: gate error param
    # c: MCMR error param
    def residuals(params, x, y, z):
        a, b, c = params
        return a*(1-b)**(x*((TQ_density*n)//2))*(1-c)**(x*y) - z
    
    initial_guess = [0.95, 0.003, 0.003]
    
    boot_spam_param = []
    boot_TQ_err = []
    boot_MCMR_err = []
    
    x_data, y_data = [], []
    for n_m in n_meas:
        seq_lengths = list(success_probs[n_m].keys())
        for L in seq_lengths:
            if len(success_probs[n_m][L]) > 0:
                x_data.append(L)
                y_data.append(n_m)
    
    for _ in range(num_resamples):
    
        boot_avg_succ_probs = {}
        boot_z_data = []
        for n_m in n_meas:
            boot_avg_succ_probs[n_m] = {}
            seq_lengths = list(success_probs[n_m].keys())
            for L in seq_lengths:
                if len(success_probs[n_m][L]) > 0:
                    boot_succ_probs = []
                    for i in range(len(success_probs[n_m][L])):
                        p_samp = np.random.choice(success_probs[n_m][L]) # non-parametric resample
                        p_sim = np.random.binomial(shots, p_samp)/shots # parametric resample
                        boot_succ_probs.append(p_sim)
                        
                    boot_avg_succ_probs[n_m][L] = np.mean(boot_succ_probs)
                    boot_y_pol = 2*boot_avg_succ_probs[n_m][L]-1
                    boot_z_data.append(boot_y_pol)
                    
        boot_fit_result = least_squares(residuals, initial_guess, args=(np.array(x_data), np.array(y_data), np.array(boot_z_data)))
        boot_fit_params = boot_fit_result.x
        
        b_spam_param = float(boot_fit_params[0])
        b_TQ_err = float(4*boot_fit_params[1]/5)
        b_MCMR_err = float(2*boot_fit_params[2]/3)

        boot_spam_param.append(b_spam_param)
        boot_TQ_err.append(b_TQ_err)
        boot_MCMR_err.append(b_MCMR_err)
        
    spam_param_std = float(np.std(boot_spam_param))
    TQ_err_std = float(np.std(boot_TQ_err))
    MCMR_err_std = float(np.std(boot_MCMR_err))
    
    return spam_param_std, TQ_err_std, MCMR_err_std
    
    

        
                

