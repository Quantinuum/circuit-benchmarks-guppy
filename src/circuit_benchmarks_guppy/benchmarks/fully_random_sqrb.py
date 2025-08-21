# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:17:43 2025

Single qubit Clifford randomized benchmarking

@author: Karl.Mayer
"""


from collections import defaultdict
import json
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

from guppylang import guppy
from guppylang.std.builtins import array, comptime, result, barrier, mem_swap
from guppylang.std.angles import pi
from guppylang.std.quantum import measure_array, rz, rx, ry, qubit
from guppylang.std.qsystem import measure_leaked, zz_phase
from guppylang.std.qsystem.random import RNG
from guppylang.std.qsystem.utils import get_current_shot
# from qtm_platform.ops import order_in_zones, sleep
from hugr.package import FuncDefnPointer

from analysis_tools import postselect_leakage, get_postselection_rates
from experiment import Experiment
import bootstrap as bs

n = guppy.nat_var("n")
T = guppy.type_var("T", copyable=False, droppable=False)


class FullyRandomSQRB_Experiment(Experiment):
    
    def __init__(self, 
                 n_qubits: int, 
                 seq_lengths: list[int], 
                 seq_reps: int,
                 qubit_transport_depths: Optional[dict] = None,
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
        #self.options['transport'] = kwargs.get('transport', False)
        self.options['barriers'] = barriers
        self.options['interleave_operation'] = interleave_operation
        self.options['delay_time'] = delay_time

        if qubit_transport_depths is not None:
            self.qubit_transport_depths = qubit_transport_depths
        else:
            self.qubit_transport_depths = {q: 1 for q in range(self.n_qubits)}

        self.length_groups = defaultdict(list)
        for q, length in self.qubit_transport_depths.items():
            self.length_groups[length].append(q)
            
        # check that qubit_transport_depths is right size
        for q_i in range(n_qubits):
            assert q_i in self.qubit_transport_depths, "qubit_transport_depths must be of length n_qubits"
        
        # check that transport_depths divide sequence lengths
        for L in seq_lengths:
            for q_i in self.qubit_transport_depths:
                t_depth = self.qubit_transport_depths[q_i]
                assert L%t_depth == 0, "Sequence lengths must be a multiple of transport depths"

        
        
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
        meas_leak = self.options['measure_leaked']
        n_qubits = self.n_qubits
        barriers = self.options['barriers']
        #delay_time = self.delay_time
        if self.options['interleave_operation'] == 'transport':
            interleave_operation = 1
        elif self.options['interleave_operation'] == 'sleep':
            interleave_operation = 2
        else:
            interleave_operation = 0
        
        assert n_qubits == len(surv_state), "len(surv_state) must equal n_qubits"
    
        with data_path('n1_lookup_tables.json').open('r') as f:
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
            p_array = array(0 for _ in range(comptime(n_qubits)))
            for q_i in range(comptime(n_qubits)):
                p = rng.random_int_bounded(g_num_paulis)
                inverse_ind = g_clifford_matrix[g_paulis[p]][g_inversion_list[clifford_state[q_i]]]
                # result("final", p)  # could comment in to get pauli but makes one large list
                clifford_gates_1Q(inverse_ind, q[q_i])
            
            
            rng.discard()
            
            # measure and report measurement outcomes
            if comptime(meas_leak):
                meas_leaked_array = array(measure_leaked(q_i) for q_i in q)
                i = 0
                for m in meas_leaked_array:
                    p = p_array[i]
                    if m.is_leaked():
                        m.discard()
                        result("c", 2)
                    else:
                        b = m.to_result().unwrap()
                        if g_flips[p][0] == 0:
                            result("c", b)
                        else:
                            result("c", not b)
                    i += 1
                            
            else:
                b_str = measure_array(q)
                for i in range(comptime(n_qubits)):
                    p = p_array[i]
                    b = b_str[i]
                    if g_flips[p][0] == 0:
                        result("c", b)
                    else:
                        result("c", not b)
                        

        return main.compile()
    
    
    # Analysis methods
    
    def analyze_results(self, error_bars=True, plot=True, display=True, save=True, **kwargs):
        
        num_resamples = kwargs.get('num_resamples', 100)
        
        marginal_results = marginalize_hists(self.n_qubits, self.results, self.qubit_transport_depths)
        
        # postselect leakage
        if self.options['measure_leaked'] == True:
            self.marginal_results = [postselect_leakage(mar_re) for mar_re in marginal_results]
            self.postselection_rates = []
            self.postselection_rates_stds = []
            for mar_re in marginal_results:
                ps_rates, ps_stds = get_postselection_rates(mar_re, self.setting_labels)
                self.postselection_rates.append(ps_rates)
                self.postselection_rates_stds.append(ps_stds)
            leakage_rates, leakage_stds = estimate_leakage_rates(self.postselection_rates,
                                                                 self.postselection_rates_stds)
            self.leakage_rates = leakage_rates
            self.leakage_stds = leakage_stds
            self.mean_leakage_rates = {}
            self.mean_leakage_stds = {}
            for j in range(self.n_qubits):
                rate = leakage_rates[j]
                std = leakage_stds[j]
                length = self.qubit_transport_depths[j]
                if length not in self.mean_leakage_rates:
                    self.mean_leakage_rates[length] = [rate]
                    self.mean_leakage_stds[length] = [std]
                else:
                    self.mean_leakage_rates[length].append(rate)
                    self.mean_leakage_stds[length].append(std)
                    
            self.mean_leakage_rates = {length:float(np.mean(self.mean_leakage_rates[length])) for length in self.mean_leakage_rates}
            self.mean_leakage_stds = {length: float(np.sqrt(sum([s**2 for s in self.mean_leakage_stds[length]]))/len(self.mean_leakage_stds[length]))
                                      for length in self.mean_leakage_stds}
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
        fid_avg = [estimate_fidelity(avg_succ_probs) for avg_succ_probs in self.avg_success_probs]
        if self.options['measure_leaked'] == True:
            self.fid_avg = [fid_avg[j] - leakage_rates[j] for j in range(self.n_qubits)]
        else:
            self.fid_avg = fid_avg
            
        self.mean_fid_avg = {
            length: np.mean([self.fid_avg[i] for i in qubits]) 
            for length, qubits in self.length_groups.items()
        }
        
        # compute error bars
        if error_bars == True:
            self.error_data = [compute_error_bars(hists, num_resamples) for hists in self.marginal_results]
            fid_avg_std = [data['avg_fid_std'] for data in self.error_data]
            if self.options['measure_leaked'] == True:
                self.fid_avg_std = [float(np.sqrt(fid_avg_std[j]**2 + leakage_stds[j]**2)) for j in range(self.n_qubits)]
            else:
                self.fid_avg_std = fid_avg_std
            
            self.mean_fid_avg_std = {
                length: np.sqrt(sum([self.fid_avg_std[i]**2 for i in qubits]))/len(qubits)
                for length, qubits in self.length_groups.items()
            }
            
            
        if plot == True:
            self.plot_results(error_bars=error_bars, **kwargs)
            if self.qubit_transport_depths != {q:1 for q in range(self.n_qubits)}:
                self.plot_scaling(error_bars=error_bars, **kwargs)
            if self.options['measure_leaked'] == True:
                self.plot_postselection_rates(**kwargs)
                if self.qubit_transport_depths != {q:1 for q in range(self.n_qubits)}:
                    self.plot_leakage_scaling(**kwargs)
            
        if display == True:
            self.display_results(error_bars=error_bars, **kwargs)
            
        if save:
            self.save()
            
            
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
        plt.xlabel('Sequence Length (number of Cliffords)')
        plt.ylim(ylim)
        if self.n_qubits <= 16:
            plt.legend()
        plt.show()
        
        

    
    def plot_scaling(self, error_bars=True, **kwargs):
        
        fit_model = kwargs.get('fit_model', 'linear') # or quadratic
        prec = kwargs.get('precision', 5)
        ylim = kwargs.get('ylim', None)
        title = kwargs.get('Memory Error Scaling', f'{self.protocol} Decays')
        
        def fit_func(x, a, b):
            return a*x + b
        
        def fit_func2(x, a, b, c):
            return b*x**2 + a*x + c
        
        x_data = list(self.mean_fid_avg.keys())
        y_data = [1 - fid for fid in self.mean_fid_avg.values()]
        
        if fit_model == 'linear':
            if error_bars:
                yerr = list(self.mean_fid_avg_std.values())
                popt, pcov = curve_fit(fit_func, x_data, y_data, sigma=yerr)
            else:
                popt, pcov = curve_fit(fit_func, x_data, y_data)
                
        elif fit_model == 'quadratic':
            if error_bars:
                yerr = list(self.mean_fid_avg_std.values())
                popt, pcov = curve_fit(fit_func2, x_data, y_data, sigma=yerr)
            else:
                popt, pcov = curve_fit(fit_func2, x_data, y_data)
        
        xfit = np.linspace(x_data[0], x_data[-1], 100)
        yfit = fit_func(xfit, *popt)
        
        plt.errorbar(x_data, y_data, yerr=yerr, fmt='bo')
        plt.plot(xfit, yfit, '-', color='b')
        plt.title(title)
        plt.ylabel('Infidelity')
        plt.xlabel('RB Sequence Length (number of Cliffords)')
        plt.ylim(ylim)
        plt.show()
        
        
        try:
            lin_mem_err = float(popt[0])
            lin_mem_err_std = np.sqrt(pcov[0][0])
            if fit_model == 'quadratic':
                quad_mem_err = float(popt[1])
                quad_mem_err_std = np.sqrt(pcov[1][1])
            
            print('Depth-1 Linear Memory Error:')
            print(f'{round(lin_mem_err, prec)} +/- {round(lin_mem_err_std, prec)}\n' + '-'*30)
            if fit_model == 'quadratic':
                print('Depth-1 Quadratic Memory Error:')
                print(f'{round(quad_mem_err, prec)} +/- {round(quad_mem_err_std, prec)}\n' + '-'*30)
        except:
            pass
        
    
    def plot_postselection_rates(self, **kwargs):
        
        ylim = kwargs.get('ylim3', None)
        title = kwargs.get('title3', f'{self.protocol} Leakage Postselection Rates')
        
        # define fit function
        def fit_func(L, a, f):
            return a*f**L
        
        # Create a colormap
        cmap = cm.turbo

        # Normalize color range from 0 to num_lines-1
        cnorm = mcolors.Normalize(vmin=0, vmax=self.n_qubits-1)
        
        for j, ps_rates in enumerate(self.postselection_rates):
            
            x = list(ps_rates.keys())
            y = list(ps_rates.values())
            
            yerr = [self.postselection_rates_stds[j][L] for L in x]
        
            # perform best fit
            popt, pcov = curve_fit(fit_func, x, y, p0=[0.4, 0.9], bounds=([0,0], [1,1]), sigma=yerr)
            xfit = np.linspace(x[0], x[-1], 100)
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
                
    
    def plot_leakage_scaling(self, **kwargs):
        
        fit_model = kwargs.get('fit_model', 'linear') # or quadratic
        prec = kwargs.get('precision', 5)
        ylim = kwargs.get('ylim4', None)
        title = kwargs.get('title4', f'{self.protocol} Leakage Rate Scaling')
        
        def fit_func(x, a, b):
            return a*x + b
        
        def fit_func2(x, a, b, c):
            return b*x**2 + a*x + c
        
        x_data = list(self.mean_leakage_rates.keys())
        y_data = list(self.mean_leakage_rates.values())
        yerr = list(self.mean_leakage_stds.values())
        
        # begin plot
        plt.errorbar(x_data, y_data, yerr=yerr, fmt='bo')
        xfit = np.linspace(x_data[0], x_data[-1], 100)
        if fit_model == 'linear' and len(x_data) > 1:
            popt, pcov = curve_fit(fit_func, x_data, y_data, sigma=yerr)
            yfit = fit_func(xfit, *popt)
            plt.plot(xfit, yfit, '-', color='b')
                
        elif fit_model == 'quadratic' and len(x_data) > 2:
            popt, pcov = curve_fit(fit_func2, x_data, y_data, sigma=yerr)
            yfit = fit_func2(xfit, *popt)
            plt.plot(xfit, yfit, '-', color='b')
        
        plt.title(title)
        plt.ylabel('Leakage Rate')
        plt.xlabel('Transport Depth')
        plt.ylim(ylim)
        plt.show()
        
        try:
            lin_leak_err = float(popt[0])
            lin_leak_err_std = float(np.sqrt(pcov[0][0]))
            if fit_model == 'quadratic':
                quad_leak_err = float(popt[1])
                quad_leak_err_std = float(np.sqrt(pcov[1][1]))
            
            print('Depth-1 Linear Leakage Error:')
            print(f'{round(lin_leak_err, prec)} +/- {round(lin_leak_err_std, prec)}\n' + '-'*30)
            if fit_model == 'quadratic':
                print('Depth-1 Quadratic Leakage Error:')
                print(f'{round(quad_leak_err, prec)} +/- {round(quad_leak_err_std, prec)}\n' + '-'*30)
        except:
            pass
        
        
    def display_results(self, error_bars=True, **kwargs):
        
        prec = kwargs.get('precision', 6)
        verbose = kwargs.get('verbose', True)
        
        print('Average Infidelities\n' + '-'*30)
        if verbose:
            for q, f_avg in enumerate(self.fid_avg):
                message = f'qubit {q}: {round(1-f_avg, prec)}'
                if error_bars == True:
                    f_std = self.error_data[q]['avg_fid_std']
                    message += f' +/- {round(f_std, prec)}'
                print(message)
            print('-'*30)
            
        for length in self.length_groups:
            avg_message = f'Qubit length {length} Average: '
            mean_infid = 1-self.mean_fid_avg[length]
            avg_message += f'{round(mean_infid,prec)}'
            if error_bars == True:
                mean_fid_avg_std = self.mean_fid_avg_std[length]
                avg_message += f' +/- {round(mean_fid_avg_std, prec)}'
            print(avg_message)
            
        if self.options['measure_leaked'] == True:
            print('Qubit average leakge rates:')
            for length in self.mean_leakage_rates:
                leak_rate = self.mean_leakage_rates[length]
                leak_std = self.mean_leakage_stds[length]
                print(f'Transport Depth {length}: {round(leak_rate, prec)} +/- {round(leak_std, prec)}')
        
        
# analysis functions

def marginalize_hists(n_qubits, hists, qubit_transport_depths):
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
            hists_q[(L/qubit_transport_depths[q], rep, exp_out)] = mar_out
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


def estimate_leakage_rates(post_rates, post_stds):
    
    leakage_rates = []
    leakage_stds = []
    
    # define fit function
    def fit_func(L, a, f):
        return a*f**L
    
    for j, ps_rates in enumerate(post_rates):
        
        seq_lengths = list(ps_rates.keys())
        y = [ps_rates[L] for L in seq_lengths]
        yerr = [post_stds[j][L] for L in seq_lengths]
        # perform best fit
        popt, pcov = curve_fit(fit_func, seq_lengths, y, p0=[0.9, 0.9], bounds=([0,0], [1,1]), sigma=yerr)
        leakage_rates.append(1-popt[1])
        leakage_stds.append(float(np.sqrt(pcov[1][1])))
    
    return leakage_rates, leakage_stds


def compute_error_bars(hists: dict, num_resamples=100):
    
    boot_hists = bootstrap(hists, num_resamples)
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
    
    
    boot_hists = []
    for i in range(num_resamples):
        boot_hists.append(bs.resample_hists(hists))
        
    
    return boot_hists



