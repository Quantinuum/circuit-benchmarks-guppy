# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:12:55 2025

Fully random binary RB with optional mid-circuit measurements

@author: Karl.Mayer
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, owned, result
from guppylang.std.quantum import measure_array, qubit
from guppylang.std.qsystem import zz_phase, measure_and_reset
from guppylang.std.qsystem.random import RNG
from guppylang.std.qsystem.utils import get_current_shot
from hugr.package import FuncDefnPointer

from guppylang.qsys_result import QsysResult
from selene_sim import build

from experiment import Experiment
from Clifford_tools import apply_SQ_Clifford
from randomized_compiling import rand_comp_rzz


Clifford_sign_table = [
    [0, 0, 0], # ('+X', '+Y', '+Z') # I
    [0, 1, 0], # ('+Z', '-Y', '+X') # H
    [0, 1, 0], # ('+Y', '-X', '+Z') # S
    [1, 0, 0], # ('-Y', '+X', '+Z') # Sdg
    [0, 1, 1], # ('+X', '-Y', '-Z') # X
    [1, 0, 1], # ('-X', '+Y', '-Z') # Y
    [1, 1, 0], # ('-X', '-Y', '+Z') # Z
    [1, 1, 0], # ('-Y', '-Z', '+X') # Ry(pi/2) Sdg
    [0, 0, 0], # ('+Y', '+Z', '+X') # Ry(pi/2) S
    [0, 0, 1], # ('+Z', '+Y', '-X') # Ry(-pi/2)
    [1, 1, 1], # ('-Z', '-Y', '-X') # Z Ry(pi/2)
    [1, 0, 0], # ('-Z', '+Y', '+X') # Ry(pi/2)
    [0, 0, 0], # ('+Z', '+X', '+Y') # Sdg Ry(-pi/2)
    [0, 0, 1], # ('+Y', '+X', '-Z') # Y S
    [1, 1, 1], # ('-Y', '-X', '-Z') # Y Sdg
    [0, 1, 1], # ('+Z', '-X', '-Y') # S Ry(-pi/2)
    [0, 0, 1], # ('+X', '+Z', '-Y') # Rx(pi/2)
    [1, 0, 1], # ('-Y', '+Z', '-X') # Ry(-pi/2) Sdg
    [0, 1, 1], # ('+Y', '-Z', '-X') # Ry(-pi/2) S
    [0, 1, 0], # ('+X', '-Z', '+Y') # Rx(-pi/2)
    [1, 0, 0], # ('-X', '+Z', '+Y') # Rx(-pi/2) Z
    [1, 0, 1], # ('-Z', '+X', '-Y') # Sdg Ry(pi/2)
    [1, 1, 0], # ('-Z', '-X', '+Y') # S Ry(pi/2)
    [1, 1, 1] # ('-X', '-Z', '-Y') # Rx(pi/2) Z
]


Clifford_lookup_table = [
    [1, 2, 3], # ('+X', '+Y', '+Z') # I
    [3, 2, 1], # ('+Z', '-Y', '+X') # H
    [2, 1, 3], # ('+Y', '-X', '+Z') # S
    [2, 1, 3], # ('-Y', '+X', '+Z') # Sdg
    [1, 2, 3], # ('+X', '-Y', '-Z') # X
    [1, 2, 3], # ('-X', '+Y', '-Z') # Y
    [1, 2, 3], # ('-X', '-Y', '+Z') # Z
    [2, 3, 1], # ('-Y', '-Z', '+X') # Ry(pi/2) Sdg
    [2, 3, 1], # ('+Y', '+Z', '+X') # Ry(pi/2) S
    [3, 2, 1], # ('+Z', '+Y', '-X') # Ry(-pi/2)
    [3, 2, 1], # ('-Z', '-Y', '-X') # Z Ry(pi/2)
    [3, 2, 1], # ('-Z', '+Y', '+X') # Ry(pi/2)
    [3, 1, 2], # ('+Z', '+X', '+Y') # Sdg Ry(-pi/2)
    [2, 1, 3], # ('+Y', '+X', '-Z') # Y S
    [2, 1, 3], # ('-Y', '-X', '-Z') # Y Sdg
    [3, 1, 2], # ('+Z', '-X', '-Y') # S Ry(-pi/2)
    [1, 3, 2], # ('+X', '+Z', '-Y') # Rx(pi/2)
    [2, 3, 1], # ('-Y', '+Z', '-X') # Ry(-pi/2) Sdg
    [2, 3, 1], # ('+Y', '-Z', '-X') # Ry(-pi/2) S
    [1, 3, 2], # ('+X', '-Z', '+Y') # Rx(-pi/2)
    [1, 3, 2], # ('-X', '+Z', '+Y') # Rx(-pi/2) Z
    [3, 1, 2], # ('-Z', '+X', '-Y') # Sdg Ry(pi/2)
    [3, 1, 2], # ('-Z', '-X', '+Y') # S Ry(pi/2)
    [1, 3, 2] # ('-X', '-Z', '-Y') # Rx(pi/2) Z
]


@guppy
def SQ_cliff_action(cliff_id: int, pauli_id: int) -> array[int, 2]:
    """ cliff_id: int between 0 and 23, representing SQ Clifford
        pauli_id: int between 0 and 3, representing SQ Pauli
    """
    
    if pauli_id == 0:
        out_pauli = 0
        out_sign = 0
    else:
        lookup_table = comptime(Clifford_lookup_table)[cliff_id]
        sign_table = comptime(Clifford_sign_table)[cliff_id]
        out_pauli = lookup_table[pauli_id-1]
        out_sign = sign_table[pauli_id-1]

    return array(out_pauli, out_sign)

@guppy
def TQ_cliff_action(q0: int, q1: int) -> tuple[int, int, int]:

    r0 = 0
    r1 = 0
    sign = 0
    
    if q0 == 0:
        if q1 == 0: # II -> II
            r0 = 0
            r1 = 0
            sign = 0
        elif q1 == 1: # IX -> ZY
            r0 = 3
            r1 = 2
            sign = 0
        elif q1 == 2: # IY -> -ZX
            r0 = 3
            r1 = 1
            sign = 1
        elif q1 == 3: # IZ -> IZ
            r0 = 0
            r1 = 3
            sign = 0

    elif q0 == 1:
        if q1 == 0: # XI -> YZ
            r0 = 2
            r1 = 3
            sign = 0
        elif q1 == 1: # XX -> XX
            r0 = 1
            r1 = 1
            sign = 0
        elif q1 == 2: # XY -> XY
            r0 = 1
            r1 = 2
            sign = 0
        elif q1 == 3: # XZ -> YI
            r0 = 2
            r1 = 0
            sign = 0

    elif q0 == 2:
        if q1 == 0: # YI -> -XZ
            r0 = 1
            r1 = 3
            sign = 1
        elif q1 == 1: # YX -> YX
            r0 = 2
            r1 = 1
            sign = 0
        elif q1 == 2: # YY -> YY
            r0 = 2
            r1 = 2
            sign = 0
        elif q1 == 3: # YZ -> -XI
            r0 = 1
            r1 = 0
            sign = 1

    elif q0 == 3:
        if q1 == 0: # ZI -> ZI
            r0 = 3
            r1 = 0
            sign = 0
        elif q1 == 1: # ZX -> IY
            r0 = 0
            r1 = 2
            sign = 0
        elif q1 == 2: # ZY -> -IX
            r0 = 0
            r1 = 1
            sign = 1
        elif q1 == 3: # ZZ -> ZZ
            r0 = 3
            r1 = 3
            sign = 0
    
    return r0, r1, sign


n = guppy.nat_var("n")      
        
@guppy
def update_stab_SQ_guppy(cliff_array: array[int, n], stab: array[int, n] @owned, sign: int) -> tuple[array[int, n], int]:
    """ cliff_array: array of integers between 0 and 23
        stab: array of integers between 0 and 3
        sign: 0 (+1) or 1 (-1)
    """
        
    for q_i in range(n):
        out_pauli, out_sign = SQ_cliff_action(cliff_array[q_i], stab[q_i])
        stab[q_i] = out_pauli
        sign = (sign + out_sign) % 2

    return stab, sign

@guppy
def update_stab_TQ_guppy(qubit_order: array[int, n], stab: array[int, n] @owned, n_q_pairs: int, sign: int) -> tuple[array[int,n], int]:

    for j in range(n_q_pairs):
        q0 = qubit_order[2*j]
        q1 = qubit_order[2*j+1]

        s0, s1, out_sign = TQ_cliff_action(stab[q0], stab[q1])
        stab[q0] = s0
        stab[q1] = s1
        sign = (sign + out_sign)%2

    return stab, sign


class FullyRandomBinaryRB_Experiment(Experiment):
    
    def __init__(self, n_qubits, seq_lengths, **kwargs):
        super().__init__(**kwargs)
        self.protocol = 'Fully Random Binary RB'
        self.parameters = {'n_qubits':n_qubits,
                           'seq_lengths':seq_lengths}
                           
        self.n_qubits = n_qubits
        self.seq_lengths = seq_lengths # list of seqence lengths
        self.setting_labels = ('n_meas', 'seq_len')
        #self.layer_depth = kwargs.get('layer_depth', 1) # number of TQ gates per layer
        self.n_meas_per_layer = kwargs.get('n_meas_per_layer', [0])
        #self.n_TQ_per_layer = kwargs.get('n_TQ_per_layer', int(np.floor(n_qubits/2)))
        self.parameters['n_meas_per_layer'] = self.n_meas_per_layer
        #self.parameters['n_TQ_per_layer'] = self.n_TQ_per_layer
        #self.parameters['layer_depth'] = self.layer_depth
        self.stabilizers = {}
        
        # options
        self.options['permute'] = kwargs.get('permute', True) # random permutation before TQ
        self.options['Pauli_twirl'] = kwargs.get('Pauli_twirl', True) # Pauli randomizations
        
        
    def add_settings(self):
        
        for n_meas in self.n_meas_per_layer:
            for seq_len in self.seq_lengths:
                
                # choose random survival state
                #surv_state = ''
                #for _ in range(self.n_qubits):
                    #surv_state += str(np.random.choice(['0', '1']))
                
                sett = (n_meas, seq_len)
                self.add_setting(sett)
                
    
    def make_circuit(self, setting: tuple) -> FuncDefnPointer:
        
        n_qubits = self.n_qubits
        n_q_pairs = int(np.floor(n_qubits/2))
        n_meas = setting[0]
        seq_len = setting[1]
        n_meas_total = n_meas*seq_len
        twirl = self.options['Pauli_twirl']
        permute = self.options['permute']
        init_seed = 12345
    
        @guppy
        def init_stabilizer(rng: RNG) -> array[int, comptime(n_qubits)]:
            # pick Z with prob 3/4 and I with prob 1/4
            stab = array(rng.random_int_bounded(4) for _ in range(comptime(n_qubits)))
            for q_i in range(comptime(n_qubits)):
                if stab[q_i] > 0:
                    stab[q_i] = 3
    
            return stab
    
        @guppy
        def sample_meas_qubits(rng: RNG) -> array[int, comptime(n_meas)]:
            
            qubit_order = array(q_i for q_i in range(comptime(n_qubits)))
            rng.shuffle(qubit_order)
            meas_qubits = array(qubit_order[i] for i in range(comptime(n_meas)))
    
            return meas_qubits
    
        
        @guppy
        def main() -> None:
    
            q = array(qubit() for _ in range(comptime(n_qubits)))
            rng = RNG(comptime(init_seed) + get_current_shot())
            stab = init_stabilizer(rng)
            mcmr_stab = array(0 for _ in range(comptime(n_meas_total)))
            sign = 0
            mcmr_index = 0
    
            # initial qubit order
            qubit_order = array(q_i for q_i in range(comptime(n_qubits)))
        
            for i in range(comptime(seq_len)):
    
                # SQ gates
                cliff_arr = array(rng.random_int_bounded(24) for _ in range(comptime(n_qubits)))
                stab, sign = update_stab_SQ_guppy(cliff_arr, stab, sign)
                for q_i in range(comptime(n_qubits)):
                    apply_SQ_Clifford(q[q_i], cliff_arr[q_i])
    
                # TQ gates
                if comptime(permute):
                    rng.shuffle(qubit_order)
                    
                stab, sign = update_stab_TQ_guppy(qubit_order, stab, comptime(n_q_pairs), sign)
                for j in range(comptime(n_q_pairs)):
                    q0 = qubit_order[2*j]
                    q1 = qubit_order[2*j+1]
                    if comptime(twirl):
                        rand_comp_rzz(q[q0], q[q1], rng)
                    else:
                        zz_phase(q[q0], q[q1], angle(0.5))
    
                # mid_circuit measurement measurements
                meas_qubits = sample_meas_qubits(rng)
                for q_i in meas_qubits:
                    if stab[q_i] == 1:
                        apply_SQ_Clifford(q[q_i], 1)
                    elif stab[q_i] == 2:
                        apply_SQ_Clifford(q[q_i], 8)
    
                    # update the MCMR stabilizer
                    mcmr_stab[mcmr_index] = stab[q_i]
                    mcmr_index += 1
                        
                    # measure
                    b_mid = measure_and_reset(q[q_i])
                    result("c_mid", b_mid)
    
                    # reset stabilizer on measured qubit
                    r = rng.random_int_bounded(4)
                    if r > 0:
                        stab[q_i] = 3
                    elif r == 0:
                        stab[q_i] = 0
    
            # measure final stabilizer
            for q_i in range(comptime(n_qubits)):
                if stab[q_i] == 1:
                    apply_SQ_Clifford(q[q_i], 1)
                elif stab[q_i] == 2:
                    apply_SQ_Clifford(q[q_i], 8)
                    
            b_str = measure_array(q)
            rng.discard()
            
            # report measurement outcomes
            for b in b_str:
                result("c", b)
            # report sign and stabilizer
            result("sign", sign)
            for pauli_id in stab:
                result("stab", pauli_id)
            for pauli_id in mcmr_stab:
                result("mcmr_stab", pauli_id)
            
        return main.compile()
    
    
    def sim(self, shots, error_model, simulator, verbose=True):
        """ simulate experiment using selene_sim simulator
            simulator: Stim() or Quest()
        """
        
        protocol = self.protocol
        n_qubits = self.n_qubits
        
        self.shots = shots
        self.raw_results = {}
        print('Simulating ...')
        for j, sett in enumerate(self.settings):
            setting = self.settings[j]
            prog = self.make_circuit(setting)
            runner = build(prog, f'{protocol} circuit {j}')
            shot_results = QsysResult(runner.run_shots(simulator,
                                    n_qubits=n_qubits,
                                    n_shots=shots,
                                    error_model=error_model))
            #outcomes = dict(Counter("".join(f"{e[1]}" for e in shot.entries) for shot in shot_results.results))
            
            raw_results = []
            for shot_result in shot_results.results:
                entries = shot_result.entries
                raw_result = {'c':'', 'c_mid':'', 'stab':'', 'mcmr_stab': '', 'sign':''}
                for entry in entries:
                    raw_result[entry[0]] += str(entry[1])
                raw_results.append(raw_result)
            self.raw_results[sett] = raw_results
            if verbose:
                print(f'{j+1}/{len(self.settings)} circuits complete')
                
    
    def analyze_results(self, error_bars=True, plot=True, display=True,
                        **kwargs):
        
        success_probs = self.get_success_probs()
        
        self.spam_param = {}
        self.layer_fidelity = {}
        
        # compute fit params
        x = list(self.seq_lengths)
        for n_meas in self.n_meas_per_layer:
            y = [self.success_probs[n_meas][L] for L in x]
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
            if len(self.n_meas_per_layer) > 1:
                message1 = f'Effective TQ avg fidelity: {round(self.fid_avg,5)}'
                if error_bars:
                    message1 += f' +/- {round(self.fid_avg_std,5)}'
                message2 = f'Effective MCMR error: {round(self.MCMR_error,5)}'
                if error_bars:
                    message2 += f' +/- {round(self.MCMR_error_std,5)}'
                print(message1)
                print(message2)
        
        
    def get_success_probs(self):
        
        success_probs = {}
        for setting in self.raw_results:
            n_meas = setting[0]
            seq_len = setting[1]
            if n_meas not in success_probs:
                success_probs[n_meas] = {}    
            p = success_probability(self.raw_results[setting])
            success_probs[n_meas][seq_len] = p
        
        self.success_probs = success_probs
        
        return success_probs
    
    
    def compute_error_bars(self, **kwargs):
        
        num_resamples = kwargs.get('num_resamples', 100)
        
        succ_probs = self.success_probs
        #shots = sum(list(self.results.values())[0].values())
        shots = self.shots
        n_qubits = self.n_qubits
        n_meas_per_layer = self.n_meas_per_layer
        
        b_succ_probs = {n_m:{L:[] for L in self.seq_lengths} for n_m in n_meas_per_layer}
        for r in range(num_resamples):
            for n_m in n_meas_per_layer:
                for L in self.seq_lengths:
                    # first do non-parametric boostrap
                    p = succ_probs[n_m][L]
                    if p < 1.0:
                        p_eff = p
                    elif p == 1.0:
                        p_eff = shots/(shots+2) # rule of 2
                    
                    b_succ_probs[n_m][L].append(float(np.random.binomial(shots, p_eff)/shots))
                        
                        
        self.success_stds = {n_m:{} for n_m in n_meas_per_layer}
        for n_m in n_meas_per_layer:
            for L in succ_probs[n_m]:
                self.success_stds[n_m][L] = float(np.std(b_succ_probs[n_m][L]))
    
        # compute error bars for spam param and layer fidelity
        x = list(self.seq_lengths)
        boot_spam_params = {n_m:[] for n_m in n_meas_per_layer}
        boot_layer_fids = {n_m:[] for n_m in n_meas_per_layer}
        if 0 in n_meas_per_layer:
            boot_avg_fid = []
        for n_m in n_meas_per_layer:
            for r in range(num_resamples):
                b_y = [b_succ_probs[n_m][L][r] for L in x]
                b_spam_param, b_layer_fid = decay_fit_params(x, b_y)
                boot_spam_params[n_m].append(b_spam_param)
                boot_layer_fids[n_m].append(b_layer_fid)
                if n_m == 0:
                    b_avg_fid = effective_TQ_fidelity(b_layer_fid, n_qubits)
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
            y = [2*self.success_probs[n_meas][L]-1 for L in x] # polarization
            
            if error_bars == True:
                yerr = [2*self.success_stds[n_meas][L] for L in x]
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
                
                
# Analysis functions

def is_success(c: str, c_mid: str, stab: str, mcmr_stab: str, sign: str):

    tot_c = c_mid + c
    tot_stab = mcmr_stab + stab
    parity = 0
    for j in range(len(tot_stab)):
        if tot_stab[j] in ['1', '2', '3'] and tot_c[j] == '1':
            parity = (parity + 1)%2

    if parity == int(sign):
        success = True
    elif parity != int(sign):
        success = False
    
    return success


def success_probability(raw_results: list):
    
    shots = len(raw_results)
    num_success = 0
    for raw_result in raw_results:
        c = raw_result['c']
        c_mid = raw_result['c_mid']
        stab = raw_result['stab']
        mcmr_stab = raw_result['mcmr_stab']
        sign = raw_result['sign']
        if is_success(c, c_mid, stab, mcmr_stab, sign):
            num_success += 1
    succ_prob = num_success/shots
    
    return succ_prob


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
        
        
        
        

