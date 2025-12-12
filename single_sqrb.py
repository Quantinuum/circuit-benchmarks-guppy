# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:17:43 2025

Single qubit Clifford randomized benchmarking

@author: Karl.Mayer
"""

import os
import pickle
from collections import defaultdict
import json
from typing import Optional
import datetime as datetime

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from guppylang import guppy
from guppylang.std.builtins import array, comptime, result, barrier, mem_swap
from guppylang.std.angles import pi
from guppylang.std.quantum import rz, rx, ry, qubit, discard_array
from guppylang.std.qsystem import zz_phase, measure_and_reset
from guppylang.std.qsystem.random import RNG
from guppylang.std.qsystem.utils import get_current_shot
# from qtm_platform.ops import order_in_zones, sleep
from hugr.package import FuncDefnPointer

from experiment import Experiment
import bootstrap as bs

n = guppy.nat_var("n")
T = guppy.type_var("T", copyable=False, droppable=False)


class Single_SQRB_Experiment(Experiment):
    
    def __init__(self, 
                 n_qubits: int, 
                 seq_lengths: list[int], 
                 seq_reps: int,
                 qubit_length_groups: Optional[dict] = None,
                 interleave_operation: str = 0,
                 barriers: bool = False,
                 delay_time: int = 0,
                 **kwargs):
        
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

        if qubit_length_groups is not None:
            self.qubit_length_groups = qubit_length_groups
        else:
            self.qubit_length_groups = {q: 1 for q in self.n_qubits}

        self.length_groups = defaultdict(list)
        for q, length in self.qubit_length_groups.items():
            self.length_groups[length].append(q)

        self.barriers = barriers
        self.interleave_operation = interleave_operation
        self.delay_time = delay_time
        
        
    def add_settings(self):
        
        for seq_len in self.seq_lengths:
            for rep in range(self.seq_reps):
                
                # choose random survival state
                surv_state = '0'*self.n_qubits  # randomized and updated in guppy
                # for _ in range(self.n_qubits):
                #     surv_state += str(np.random.choice(['0', '1']))
                
                sett = (seq_len, rep, surv_state)
                self.add_setting(sett)
        
    
    def make_circuit(self, setting: tuple, seed: int = 0) -> FuncDefnPointer:
        """ 
        seq_len (int): number of Cliffords in circuit
        surv_state (str): expected outcome
        """
        
        seq_len = setting[0]
        surv_state = setting[2]
        n_qubits = self.n_qubits
        barriers = self.barriers
        delay_time = self.delay_time
        if self.interleave_operation == 'transport':
            interleave_operation = 1
        elif self.interleave_operation == 'sleep':
            interleave_operation = 2
        else:
            interleave_operation = 0
        
        assert n_qubits == len(surv_state), "len(surv_state) must equal n_qubits"
    
        with open(f'n1_lookup_tables.json', 'r') as f:
            lookup_table = json.load(f)
        
        clifford_matrix = lookup_table['clifford_matrix']
        inversion_list = lookup_table['inversion_list']
        paulis = lookup_table['paulis']
        flips = lookup_table['flips']

        num_cliffs = 24
        num_paulis = 4

        @guppy
        def clifford_gates_1Q(cliff_ind: int, qubit0: qubit) -> None:

            if cliff_ind == 0:
                pass
            elif cliff_ind == 1:
                rx(qubit0, -0.5*pi)
            elif cliff_ind == 2:
                rx(qubit0, 1*pi)
            elif cliff_ind == 3:
                rx(qubit0, 0.5*pi)
            elif cliff_ind == 4:
                ry(qubit0, -0.5*pi)
            elif cliff_ind == 5:
                ry(qubit0, 1*pi)
            elif cliff_ind == 6:
                ry(qubit0, 0.5*pi)
            elif cliff_ind == 7:
                rz(qubit0, -0.5*pi)
            elif cliff_ind == 8:
                rz(qubit0, 1*pi)
            elif cliff_ind == 9:
                rz(qubit0, 0.5*pi)
            elif cliff_ind == 10:
                rx(qubit0, -0.5*pi)
                ry(qubit0, -0.5*pi)
            elif cliff_ind == 11:
                rx(qubit0, -0.5*pi)
                ry(qubit0, 1*pi)
            elif cliff_ind == 12:
                rx(qubit0, -0.5*pi)
                ry(qubit0, 0.5*pi)
            elif cliff_ind == 13:
                rx(qubit0, -0.5*pi)
                rz(qubit0, -0.5*pi)
            elif cliff_ind == 14:
                rx(qubit0, -0.5*pi)
                rz(qubit0, 1*pi)
            elif cliff_ind == 15:
                rx(qubit0, -0.5*pi)
                rz(qubit0, 0.5*pi)
            elif cliff_ind == 16:
                rx(qubit0, 1*pi)
                ry(qubit0, -0.5*pi)
            elif cliff_ind == 17:
                rx(qubit0, 1*pi)
                ry(qubit0, 0.5*pi)
            elif cliff_ind == 18:
                rx(qubit0, 1*pi)
                rz(qubit0, -0.5*pi)
            elif cliff_ind == 19:
                rx(qubit0, 1*pi)
                rz(qubit0, 0.5*pi)
            elif cliff_ind == 20:
                rx(qubit0, 0.5*pi)
                ry(qubit0, -0.5*pi)
            elif cliff_ind == 21:
                rx(qubit0, 0.5*pi)
                ry(qubit0, 0.5*pi)
            elif cliff_ind == 22:
                rx(qubit0, 0.5*pi)
                rz(qubit0, -0.5*pi)
            elif cliff_ind == 23:
                rx(qubit0, 0.5*pi)
                rz(qubit0, 0.5*pi)

        @guppy
        def shuffle(array: array[T, n], rng: RNG) -> None:
            """Randomly shuffle the elements of a possibly linear array in place.
            Uses the Fisher-Yates algorithm."""
            for k in range(n):
                i = n - 1 - k
                j = rng.random_int_bounded(i + 1)
                if i != j:
                    mem_swap(array[i], array[j])

        @guppy
        def depth_one(qarray: array[qubit, n], new_order: array[int, n]) -> None:

            for i in range(n):
                if i % 2 == 0:
                    zz_phase(qarray[new_order[i]], qarray[new_order[i+1]], 0*pi)

        @guppy
        def main() -> None:
            g_num_cliffs = comptime(num_cliffs)
            g_paulis = comptime(paulis)
            g_num_paulis = comptime(num_paulis)
            g_clifford_matrix = comptime(clifford_matrix)
            g_inversion_list = comptime(inversion_list)
            g_flips = comptime(flips)

            q = array(qubit() for _ in range(comptime(n_qubits)))
            rng = RNG(comptime(seed) + get_current_shot())

            # make `seq_len` random cliffords and track state
            clifford_state = array(0 for _ in range(comptime(n_qubits)))
            for _ in range(comptime(seq_len)):
                for q_i in range(comptime(n_qubits)):
                    randval = rng.random_int_bounded(g_num_cliffs)
                    clifford_state[q_i] = g_clifford_matrix[randval][clifford_state[q_i]]
                    clifford_gates_1Q(randval, q[q_i])

                if comptime(interleave_operation) == 1:
                    order = array(i for i in range(comptime(n_qubits)))
                    shuffle(order, rng)
                    depth_one(q, order)
                # elif comptime(interleave_operation) == 2:
                #     sleep(q, comptime(delay_time))
                
                if comptime(barriers):
                        barrier(q)

            # randomize final state by adding extra Pauli gate
            for q_i in range(comptime(n_qubits)):
                p = rng.random_int_bounded(g_num_paulis)
                inverse_ind = g_clifford_matrix[g_paulis[p]][g_inversion_list[clifford_state[q_i]]]
                # result("final", p)  # could comment in to get pauli but makes one large list
                clifford_gates_1Q(inverse_ind, q[q_i])

                b = measure_and_reset(q[q_i])
                if g_flips[p][0] == 0:
                    result("c", b)
                else:
                    result("c", not b)

            discard_array(q)
            rng.discard()

        return main.compile()
    
    
    # Analysis methods
    
    def analyze_results(self, error_bars=True, plot=True, display=True, **kwargs):
        
        
        self.marginal_results = marginalize_hists(self.n_qubits, self.results, self.qubit_length_groups)
        self.success_probs = []
        self.avg_success_probs = []
        for j, hists in enumerate(self.marginal_results):
            succ_probs_j = get_success_probs(hists)
            avg_succ_probs_j = get_avg_success_probs(succ_probs_j)
            self.success_probs.append(succ_probs_j)
            self.avg_success_probs.append(avg_succ_probs_j)
            
        # estimate fidelity
        self.fid_avg = [estimate_fidelity(avg_succ_probs) for avg_succ_probs in self.avg_success_probs]
        self.mean_fid_avg = {
            length: np.mean([self.fid_avg[i] for i in qubits]) 
            for length, qubits in self.length_groups.items()
        }
        
        # compute error bars
        if error_bars == True:
            self.error_data = [compute_error_bars(hists) for hists in self.marginal_results]
            self.fid_avg_std = [data['avg_fid_std'] for data in self.error_data]
            self.mean_fid_avg_std = {
                length: np.sqrt(sum([self.fid_avg_std[i]**2 for i in qubits]))/len(qubits)
                for length, qubits in self.length_groups.items()
            }
            
            
        if plot == True:
            self.plot_results(error_bars=error_bars, **kwargs)
            
        if display == True:
            self.display_results(error_bars=error_bars, **kwargs)

        if github_save == True:
            date = str(datetime.now().strftime("%Y_%m_%d-%H-%M-%S"))
            survival = {}
            leakage_postselect = {}
            for q, res in enumerate(self.marginal_results):
                survival[q] = {}
                leakage_postselect[q] = {}
                for (length, rep, s), val in res.items():
                    try:
                        survival[q][length][rep] = val[s]
                        leakage_postselect[q][length][rep] = sum(val.values())
                    except KeyError:
                        survival[q][length] = {rep: val[s]}
                        leakage_postselect[q][length] = {rep: sum(val.values())}


            raw_results = {}
            survival = {}
            for i, ((length, rep, s), res) in enumerate(self.results.items()):
                raw_results[f'SQ_RB ({length}, {rep}) [{i}]'] = res
                survival[f'SQ_RB ({length}, {rep}) [{i}]'] = s
                leakage_postselect[f'SQ_RB ({length}, {rep}) [{i}]']

            data = {
                'shots': self.shots,
                'survival': survival,
                'sequence_info': {length: self.seq_reps for length in self.seq_lengths},
                'raw_data': raw_results,
                'expected_output': {(length, rep): exp_output for length, rep, exp_output in self.settings},
                'leakage_postselect': leakage_postselect
            }
            with open(f'SQ_RB_data_helios-1_{date}.json', 'w+') as f:
                json.dump(data, f, indent=2)            
            
    def plot_results(self, error_bars=True, **kwargs):
        
        ylim = kwargs.get('ylim', None)
        
        title = kwargs.get('title', f'{self.protocol} Decays')
        
        # define fit function
        def fit_func(L, a, f):
            return a*f**L+1/2
        
        cmap = plt.cm.turbo  # define the colormap
        colors = [
            cmap(i) 
            for i in range(0, cmap.N, cmap.N//self.n_qubits)
        ]
        
        x = self.seq_lengths
        xfit = np.linspace(x[0], x[-1], 100)
        
        for j, avg_succ_probs in enumerate(self.avg_success_probs):
            
            co = colors[j]
            x = list(avg_succ_probs.keys())
            y = list(avg_succ_probs.values())
            if error_bars == False:
                yerr = None
            else:
                yerr = [self.error_data[j]['success_probs_stds'][L] for L in x]
        
            # perform best fit
            xfit = np.linspace(x[0], x[-1], 100)
            popt, pcov = curve_fit(fit_func, x, y, p0=[0.4, 0.9], bounds=([0,0], [0.5,1]))
            yfit = fit_func(xfit, *popt)
            plt.errorbar(x, y, yerr=yerr, fmt='o', color=co, label=f'q{j}')
            plt.plot(xfit, yfit, '-', color=co)
        
        plt.title(title)
        plt.ylabel('Success Probability')
        plt.xlabel('Sequence Length')
        plt.ylim(ylim)
        plt.legend()
        plt.show()

    
    def plot_scaling(self,**kwargs):
        
        ylim = kwargs.get('ylim', None)
        
        title = kwargs.get('SQRB scaling', f'{self.protocol} Decays')
        
        plt.errorbar(
            list(self.mean_fid_avg.keys()),
            [1 - fid for fid in self.mean_fid_avg.values()],
            yerr=list(self.mean_fid_avg_std.values())
        )
        
        plt.title(title)
        plt.ylabel('Infidelity')
        plt.xlabel('Delay depth')
        plt.ylim(ylim)
        plt.show()
        
        
    def display_results(self, error_bars=True):
        
        print('Average Fidelities\n' + '-'*30)
        for q, f_avg in enumerate(self.fid_avg):
            message = f'qubit {q}: {round(f_avg, 6)}'
            if error_bars == True:
                f_std = self.error_data[q]['avg_fid_std']
                message += f' +/- {round(f_std, 6)}'
            print(message)

        print('-'*30)
        for length in self.length_groups:
            avg_message = f'Qubit length {length} Average: '
            mean_fid_avg = self.mean_fid_avg[length]
            avg_message += f'{round(mean_fid_avg,6)}'
            if error_bars == True:
                mean_fid_avg_std = self.mean_fid_avg_std[length]
                avg_message += f' +/- {round(mean_fid_avg_std, 6)}'
            print(avg_message)
        
        
# analysis functions

def marginalize_hists(n_qubits, hists, qubit_length_groups):
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
            hists_q[(L/qubit_length_groups[q], rep, exp_out)] = mar_out
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




