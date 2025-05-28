# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:53:50 2025

TQ Cycle Benchmarking

@author: Karl.Mayer
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from guppylang import guppy
from guppylang.std.builtins import array, barrier, comptime, py, result
from guppylang.std.quantum import measure_array, qubit, h, x, z, s, sdg
from guppylang.std.qsystem import zz_phase
from guppylang.std.qsystem.random import RNG
from guppylang.std.qsystem.utils import get_current_shot
from hugr.package import FuncDefnPointer

from experiment import Experiment
from analysis_tools import marginalize_hists
import bootstrap as bs
from randomized_compiling import rand_comp_rzz


Paulis = ['IX','IY','IZ','XI','XX','XY','XZ','YI','YX','YY','YZ','ZI','ZX','ZY','ZZ']


class CB_Experiment(Experiment):
    
    def __init__(self, qubits, seq_lengths, **kwargs):
        super().__init__(**kwargs)
        self.protocol = 'TQ Cycle Benchmarking'
        self.parameters = {'qubits':qubits,
                           'seq_lengths':seq_lengths}
        
        
        n_qubits = max([max(q_pair) for q_pair in qubits]) + 1
        self.n_qubits = n_qubits
        self.qubits = qubits
        self.seq_lengths = seq_lengths
        self.setting_labels = ('seq_len', 'init_state')
        
        self.options['experiment_size'] = kwargs.get('experiment_size', 'small')
        self.options['init_seed'] = kwargs.get('init_seed', 12345)
        self.options['transport'] = kwargs.get('transport', False)
        
        
    def add_settings(self, **kwargs):
        
        exp_size = kwargs.get('experiment_size', None)
        if exp_size:
            self.options['experiment_size'] = exp_size
        
        # choice of input states can override 'experiment_size' argument
        input_states = kwargs.get('input_states', None)
        if not input_states:
            experiment_size = self.options['experiment_size']
            if experiment_size == 'small':
                input_states = ['00', '01', '10', '11', '++', '+-', '-+', '--']
            elif experiment_size == 'medium':
                input_states = ['00', '01', '10', '11', '++', '+-', '-+', '--', 'RR', 'RL', 'LR', 'LL']
            elif experiment_size == 'large':
                input_states = ['00', '01', '10', '11', '++', '+-', '-+', '--',
                                '+R', '+L', '-R', '-L', 'R+', 'R-', 'L+', 'L-', 'RR', 'RL', 'LR', 'LL']
        
        self.input_states = input_states
                
        
        for seq_len in self.seq_lengths:
            for state in input_states:
                init_state = tuple((state for _ in range(len(self.qubits))))
                sett = (seq_len, init_state)
                self.add_setting(sett)
                
                
    def make_circuit(self, setting:tuple) -> FuncDefnPointer:
        """ 
        seq_len (int): number of TQ gates per qubit pair in circuit
        init_state: initial state to each qubit pair, i.e.m ('01', '+R', ...)
        """
        
        seq_length = setting[0]
        init_state = setting[1]
        qubits = self.qubits
        n_qubits = self.n_qubits
        init_seed = self.options['init_seed']
        
        assert len(qubits) == len(init_state), "len(qubits) must equal len(init_state)"
    
        init_commands = []
        meas_commands = []
    
        # state prep
        for j, q_pair in enumerate(qubits):
            for k, q_id in enumerate(q_pair):
                state = init_state[j][k]
                if state == '0':
                    init_commands.append((0, q_id)) # I
                elif state == '1':
                    init_commands.append((1, q_id)) # X
                elif state == '+':
                    init_commands.append((2, q_id)) # H
                elif state == '-':
                    init_commands.append((2, q_id)) # H
                    init_commands.append((3, q_id)) # Z
                elif state == 'R':
                    init_commands.append((2, q_id)) # H
                    init_commands.append((4, q_id)) # S 
                elif state == 'L':
                    init_commands.append((2, q_id)) # H
                    init_commands.append((5, q_id)) # Sdg
    
        # meas prep
        for j, q_pair in enumerate(qubits):
            for k, q_id in enumerate(q_pair):
                state = init_state[j][k]
                if state == '0':
                    meas_commands.append((0, q_id))
                elif state == '1':
                    meas_commands.append((0, q_id))
                elif state == '+':
                    meas_commands.append((2, q_id))
                elif state == '-':
                    meas_commands.append((2, q_id))
                elif state == 'R':
                    meas_commands.append((5, q_id))
                    meas_commands.append((2, q_id)) 
                elif state == 'L':
                    meas_commands.append((5, q_id))
                    meas_commands.append((2, q_id))
    
        @guppy
        def main() -> None:
            q = array(qubit() for _ in range(py(n_qubits)))
            rng = RNG(comptime(init_seed) + get_current_shot())
    
            for gate_id, q_id in py(init_commands):
                if gate_id == 1:
                    x(q[q_id])
                elif gate_id == 2:
                    h(q[q_id])
                elif gate_id == 3:
                    z(q[q_id])
                elif gate_id == 4:
                    s(q[q_id])
                elif gate_id == 5:
                    sdg(q[q_id])
            
            for _ in range(py(seq_length)):
                for q0, q1 in py(qubits):
                    rand_comp_rzz(q[q0], q[q1], rng)
                
                barrier(q)
    
            for gate_id, q_id in py(meas_commands):
                if gate_id == 2:
                    h(q[q_id])
                elif gate_id == 4:
                    s(q[q_id])
                elif gate_id == 5:
                    sdg(q[q_id])
            
            b_str = measure_array(q)
            rng.discard()
            # report measurement outcomes
            for b in b_str:
                result("c", b)
    
        # return the compiled program (HUGR)
        return main.compile()
    
    
    def analyze_results(self, error_bars=True, plot=True, display=True, **kwargs):
        

        self.mar_results = marginalize_hists(self.qubits, self.results, mar_exp_out=True)
        self.exp_values = [results_to_exp_values(mar_re) for mar_re in self.mar_results]
        self.Pauli_fids = [estimate_Pauli_fids(exp_vals) for exp_vals in self.exp_values]
        self.Pauli_probs = [estimate_Pauli_probs(P_fids) for P_fids in self.Pauli_fids]
        self.fid_avg = [average_fidelity(P_probs) for P_probs in self.Pauli_probs]
        
        # zone averages
        if len(self.qubits) > 1:
            self.mean_Pauli_probs = {P:float(np.mean([P_probs[P] for P_probs in self.Pauli_probs])) for P in Paulis}
            self.zone_mean_fid_avg = float(np.mean(self.fid_avg))
        
        
        if error_bars:
            self.error_data = [compute_error_bars(mar_re) for mar_re in self.mar_results]
            self.fid_avg_std = [err_data['fid_avg_std'] for err_data in self.error_data]
            
            if len(self.qubits) > 1:
                self.zone_mean_fid_avg_std = float(np.sqrt(sum([s**2 for s in self.fid_avg_std])))/len(self.fid_avg_std)
                self.mean_Pauli_stds = {}
                for P in Paulis:
                    P_std = float(np.sqrt(sum([err_data['Pauli_probs_stds'][P]**2 for err_data in self.error_data])))/len(self.qubits)
                    self.mean_Pauli_stds[P] = P_std
                
        if plot:
            
            # plot decays
            for j, exp_vals in enumerate(self.exp_values):
                q_pair = self.qubits[j]
                if error_bars:
                    exp_vals_stds = self.error_data[j]['exp_values_stds']
                else:
                    exp_vals_stds = None
                plot_exp_value_decays(exp_vals, exp_values_stds=exp_vals_stds,
                                      title=f'Qubits {q_pair}', **kwargs)
            
            # plot Pauli errors
            for j, q_pair in enumerate(self.qubits):
                P_probs = self.Pauli_probs[j]
                if error_bars:
                    yerr = list(self.error_data[j]['Pauli_probs_stds'].values())
                else:
                    yerr = None
                plot_Pauli_probs(P_probs, yerr=yerr, title=f'Qubits {q_pair}', **kwargs)
            
            # plot zone-average
            if len(self.qubits) > 1:
                mean_P_probs = self.mean_Pauli_probs
                if error_bars:
                    yerr = list(self.mean_Pauli_stds.values())
                else:
                    yerr = None
                plot_Pauli_probs(mean_P_probs, yerr=yerr, title='Zone Average', **kwargs)
                
                
            # display results
            if display:
                print('Average Fidelity\n' + '-'*26)
                for j, q_pair in enumerate(self.qubits):
                    F_avg = self.fid_avg[j]
                    if error_bars:
                        F_avg_std = self.error_data[j]['fid_avg_std']
                        print(f'{q_pair}: {round(F_avg,5)} +/- {round(F_avg_std, 5)}')
                    else:
                        print(f'{q_pair}: {round(F_avg,5)}')
                print('\nZone Average:')
                mean_fid = self.zone_mean_fid_avg
                if error_bars:
                    mean_fid_std = self.zone_mean_fid_avg_std
                    print(f'{round(mean_fid,5)} +/- {round(mean_fid_std, 5)}')
                else:
                    print(f'{round(mean_fid,5)}')    
                    
    
    
# analysis functions

def results_to_exp_values(mar_results):
    
    # read in seq lengths and input states from mar_results dictionary
    input_states = []
    seq_lengths = []
    for sett in mar_results:
        seq_len = sett[0]
        init_state = sett[1]
        if seq_len not in seq_lengths:
            seq_lengths.append(seq_len)
        if init_state not in input_states:
            input_states.append(init_state)
    
    
    exp_values = {}
    Paulis = []
    if '00' in input_states and '01' in input_states and '10' in input_states and '11' in input_states:
        Paulis += ['IZ', 'ZI', 'ZZ']

    if '++' in input_states and '+-' in input_states and '-+' in input_states and '--' in input_states:
        Paulis += ['IX', 'XI', 'XX']

    if 'RR' in input_states and 'RL' in input_states and 'LR' in input_states and 'LL' in input_states:
        Paulis += ['IY', 'YI', 'YY']

    if '+R' in input_states and '+L' in input_states and '-R' in input_states and '-L' in input_states:
        Paulis += ['XY']

    if 'R+' in input_states and 'R-' in input_states and 'L+' in input_states and 'L-' in input_states:
        Paulis += ['YX']

    Had_trans_dict = {
        
        'IZ':{'00':1, '01':-1, '10':1, '11':-1},
        'ZI':{'00':1, '01':1, '10':-1, '11':-1},
        'ZZ':{'00':1, '01':-1, '10':-1, '11':1},
        'IX':{'++':1, '+-':-1, '-+':1, '--':-1},
        'XI':{'++':1, '+-':1, '-+':-1, '--':-1},
        'XX':{'++':1, '+-':-1, '-+':-1, '--':1},
        'IY':{'RR':1, 'RL':-1, 'LR':1, 'LL':-1},
        'YI':{'RR':1, 'RL':1, 'LR':-1, 'LL':-1},
        'YY':{'RR':1, 'RL':-1, 'LR':-1, 'LL':1},
        'XY':{'+R':1, '+L':-1, '-R':-1, '-L':1},
        'YX':{'R+':1, 'R-':-1, 'L+':-1, 'L-':1},
    }
    
    for P in Paulis:

        if P in ['IZ', 'ZI', 'ZZ']:
                outcomes_dict = {'00':'00', '01':'01', '10':'10', '11':'11'}
        elif P in ['IX', 'XI', 'XX']:
            outcomes_dict = {'00':'++', '01':'+-', '10':'-+', '11':'--'}
        elif P in ['IY', 'YI', 'YY']:
            outcomes_dict = {'00':'RR', '01':'RL', '10':'LR', '11':'LL'}
        elif P == 'XY':
            outcomes_dict = {'00':'+R', '01':'+L', '10':'-R', '11':'-L'}
        elif P == 'YX':
            outcomes_dict = {'00':'R+', '01':'R-', '10':'L+', '11':'L-'}
        
        exp_values[P] = {}
        for seq_len in seq_lengths:
            exp_val = 0.0
            for state in [str(psi) for psi in list(Had_trans_dict[P].keys())]:
                outcomes = mar_results[(seq_len, state)]
                shots = sum(outcomes.values())
                ev = 0.0
                for b_str in outcomes:
                    counts = outcomes[b_str]
                    ev += Had_trans_dict[P][outcomes_dict[b_str]]*counts/shots
                exp_val += Had_trans_dict[P][state]*ev/4

            exp_values[P][seq_len] = exp_val

    return exp_values
            

def estimate_Pauli_fids(exp_values):

    def fit_func(L, a, f):
        return a*f**L
    
    Pauli_fids = {}
    for P in exp_values:
        
        x = [int(seq_len) for seq_len in list(exp_values[P].keys())]
        y = [exp_values[P][seq_len] for seq_len in x]
    
        # perform best fit
        popt, pcov = curve_fit(fit_func, x, y, p0=[0.9, 0.9], bounds=([0,0], [1,1]))
        P_fid = float(popt[1])
        Pauli_fids[P] = P_fid

    return Pauli_fids


def extend_Pauli_fids(Pauli_fids):

    ext_Pauli_fids = {}
    for P in Pauli_fids:
        ext_Pauli_fids[P] = Pauli_fids[P]
        
    # first add X-Y symmetry
    if 'IX' in Pauli_fids and 'IY' not in Pauli_fids:
        ext_Pauli_fids['IY'] = Pauli_fids['IX']

    if 'XI' in Pauli_fids and 'YI' not in Pauli_fids:
        ext_Pauli_fids['YI'] = Pauli_fids['XI']

    if 'XX' in Pauli_fids and 'YY' not in Pauli_fids:
        ext_Pauli_fids['YY'] = Pauli_fids['XX']

    if 'IY' in Pauli_fids and 'IX' not in Pauli_fids:
        ext_Pauli_fids['IX'] = Pauli_fids['IY']

    if 'YI' in Pauli_fids and 'XI' not in Pauli_fids:
        ext_Pauli_fids['XI'] = Pauli_fids['YI']

    if 'YY' in Pauli_fids and 'XX' not in Pauli_fids:
        ext_Pauli_fids['XX'] = Pauli_fids['YY']

    if 'XX' in Pauli_fids and 'XY' not in Pauli_fids:
        ext_Pauli_fids['XY'] = Pauli_fids['XX']

    if 'XX' in Pauli_fids and 'YX' not in Pauli_fids:
        ext_Pauli_fids['YX'] = Pauli_fids['XX']

    if 'YY' in Pauli_fids and 'XY' not in Pauli_fids:
        ext_Pauli_fids['XY'] = Pauli_fids['YY']

    if 'YY' in Pauli_fids and 'YX' not in Pauli_fids:
        ext_Pauli_fids['YX'] = Pauli_fids['YY']

    # add unlearnable error symmetry
    if 'IX' in ext_Pauli_fids and 'ZY' not in Pauli_fids:
        ext_Pauli_fids['ZY'] = ext_Pauli_fids['IX']

    if 'XI' in ext_Pauli_fids and 'YZ' not in Pauli_fids:
        ext_Pauli_fids['YZ'] = ext_Pauli_fids['XI']

    if 'IY' in ext_Pauli_fids and 'ZX' not in Pauli_fids:
        ext_Pauli_fids['ZX'] = ext_Pauli_fids['IY']

    if 'YI' in ext_Pauli_fids and 'XZ' not in Pauli_fids:
        ext_Pauli_fids['XZ'] = ext_Pauli_fids['YI']

    
    ext_Pauli_fids = {P:ext_Pauli_fids[P] for P in ext_Pauli_fids}

    return ext_Pauli_fids


def commute(P1: str, P2: str):
    """ compute whether P1 and P2 commute """

    assert len(P1) == len(P2), "len(P1) must equal len(P2)!"
    
    out = 1
    for j in range(len(P1)):
        if P1[j] != 'I' and P2[j] != 'I' and P1[j] != P2[j]:
            out = (out+1)%2

    return bool(out)


def estimate_Pauli_probs(Pauli_fids):


    ext_Pauli_fids = extend_Pauli_fids(Pauli_fids)

    Pauli_probs = {}
    for P in Paulis:

        commute_fids = []
        anticommute_fids = []

        for P2 in ext_Pauli_fids:
            if commute(P, P2):
                commute_fids.append(ext_Pauli_fids[P2])
            elif not commute(P, P2):
                anticommute_fids.append(ext_Pauli_fids[P2])

        P_prob_est = float(((7*np.mean(commute_fids)+1) - 8*np.mean(anticommute_fids))/16)
        Pauli_probs[P] = max([0, P_prob_est])
        
    return Pauli_probs


def average_fidelity(Pauli_probs):

    F_pro = 1 - sum(Pauli_probs.values())
    F_avg = (4*F_pro + 1)/5
                    
    return F_avg


def plot_exp_value_decays(exp_values, exp_values_stds=None, title=None, ylim=None, **kwargs):
    
    def fit_func(L, a, f):
        return a*f**L
    
    x = [int(seq_len) for seq_len in list(list(exp_values.values())[0].keys())]
    x.sort()
    xfit = np.linspace(x[0], x[-1], 100)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    for j, P in enumerate(exp_values):
        co = colors[j%6]
        y = [exp_values[P][seq_len] for seq_len in x]
        if exp_values_stds:
            yerr = [exp_values_stds[P][seq_len] for seq_len in x]
        else:
            yerr = None
        popt, pcov = curve_fit(fit_func, x, y, p0=[0.9, 0.9], bounds=([0,0], [1,1]))
        yfit = fit_func(xfit, *popt)
        plt.errorbar(x, y, yerr=yerr, fmt='o', color=co, label=P)
        plt.plot(xfit, yfit, '-', color=co)
    plt.xticks(ticks=x, labels=x)
    plt.xlabel('Sequence Length')
    plt.ylabel('Expectation Value')
    plt.title(title)
    plt.ylim(ylim)
    plt.legend()
    plt.show()


def plot_Pauli_probs(Pauli_probs: dict, yerr=None, err_lim=None, title=None, **kwargs):
    
    plot_orbits = kwargs.get('plot_orbits', False)
    
    if plot_orbits == False:
        x = [str(P) for P in list(Pauli_probs.keys())]
        y = [Pauli_probs[P] for P in x]
    elif plot_orbits == True:
        x = ['IX,ZY', 'IY,ZX', 'IZ', 'XI,YZ', 'XX', 'XY', 'XZ,YI', 'YX', 'YY','ZI', 'ZZ']
        y = [Pauli_probs[P] for P in ['IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YX', 'YY', 'ZI', 'ZZ']]
        if yerr:
            yerr = [yerr[j] for j in [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 14]]
    
    
    # create plot
    fig, ax = plt.subplots()
    ax.bar(range(len(y)), y, yerr=yerr, width=0.75)
    ax.set_xticks(range(len(y)))
    ax.set_xticklabels(x)
    ax.set_title(title)
    ax.set_ylim(err_lim)
    ax.set_xlabel('Pauli Error')
    ax.set_ylabel('Probability')
    plt.show()
    
    
    
# error analysis functions

def compute_error_bars(mar_results):
    
    # read in seq_lengths from marginal results
    seq_lengths = []
    for sett in mar_results:
        seq_len = sett[0]
        if seq_len not in seq_lengths:
            seq_lengths.append(seq_len)
    
    boot_results = bootstrap(mar_results)
    boot_exp_values = [results_to_exp_values(b_results) for b_results in boot_results]
    boot_Pauli_fids = [estimate_Pauli_fids(b_exp_values) for b_exp_values in boot_exp_values]
    boot_ext_Pauli_fids = [extend_Pauli_fids(b_Pauli_fids) for b_Pauli_fids in boot_Pauli_fids]
    boot_Pauli_probs = [estimate_Pauli_probs(b_ext_Pauli_fids) for b_ext_Pauli_fids in boot_ext_Pauli_fids]
    
    boot_fid_avg = [average_fidelity(b_P_p) for b_P_p in boot_Pauli_probs]
    
    # process bootstrapped data
    exp_values_stds = {}
    for P in list(boot_exp_values[0].keys()):
        exp_values_stds[P] = {}
        for L in seq_lengths:
            exp_values_stds[P][L] = float(np.std([b_ev[P][L] for b_ev in boot_exp_values]))
    
    #P_fids_stds = {P:np.std([b_P_fids[P] for b_P_fids in boot_P_fids]) for P in Paulis}
    fid_avg_std = float(np.std([f for f in boot_fid_avg]))
    Pauli_probs_stds = {P:float(np.std([b_P_probs[P] for b_P_probs in boot_Pauli_probs])) for P in Paulis}
    
    error_data = {'exp_values_stds':exp_values_stds,
                  'Pauli_probs_stds':Pauli_probs_stds,
                  'fid_avg_std':fid_avg_std}
    
    return error_data


def bootstrap(results: dict, num_resamples=100):
    """ 
    parametric resampling from hists
    """
    
    boot_results = []
    for i in range(num_resamples):
        b_results = bs.resample_hists(results)
        boot_results.append(b_results)
    
    return boot_results
    
        
        
        
