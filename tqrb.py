# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 16:20:52 2025

Two qubit Clifford randomized benchmarking

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
from guppylang.std.angles import angle
from guppylang.std.builtins import array, barrier, comptime
from guppylang.std.quantum import qubit, h, z, x, y, s, sdg
from guppylang.std.qsystem import zz_phase
from hugr.package import FuncDefnPointer

from experiment import Experiment
import analysis_tools as at
from leakage_measurement import measure_and_record_leakage
import bootstrap as bs


# load TQ Clifford group
module_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(module_dir, 'TQ_Clifford_group.p')
with open(file_path, 'rb') as f:
    TQ_Clifford_group = pickle.load(f)
    
Clifford_group_list = list(TQ_Clifford_group.keys())


class TQRB_Experiment(Experiment):
    
    def __init__(self, qubits, seq_lengths, seq_reps, **kwargs):
        super().__init__(**kwargs)
        self.protocol = 'TQRB'
        self.parameters = {'qubits':qubits,
                           'seq_lengths':seq_lengths,
                           'seq_reps':seq_reps}
        
        n_qubits = max([max(q_pair) for q_pair in qubits]) + 1
        self.n_qubits = n_qubits
        self.qubits = qubits
        self.seq_lengths = seq_lengths
        self.seq_reps = seq_reps
        self.setting_labels = ('seq_len', 'seq_rep', 'surv_state')
        
        self.options['barriers'] = True
        self.options['parallel'] = False
        self.options['transport'] = kwargs.get('transport', False)
        
        
    def add_settings(self):
        
        for seq_len in self.seq_lengths:
            for rep in range(self.seq_reps):
                
                # choose random survival state
                surv_state = tuple([str(np.random.choice(['00', '01', '10', '11'])) for q_pair in self.qubits])
                
                sett = (seq_len, rep, surv_state)
                self.add_setting(sett)
                
                
    def make_circuit(self, setting:tuple) -> FuncDefnPointer:
        """ 
        seq_len (int): number of Cliffords in circuit
        surv_state (str): expected outcome
        """
        
        seq_len = setting[0]
        surv_state = setting[2]
        barriers = self.options['barriers']
        meas_leak = self.options['measure_leaked']
        parallel = self.options['parallel']
        n_qubits = self.n_qubits
        n_q_pairs = len(self.qubits)
        qubits = self.qubits
         
        
        # track unitaries in each gate zone
        unitary_list = [np.diag([1,1,1,1]) for _ in range(len(qubits))]
        command_list = [] # list of lists of commands for Clifford group elements
    
        for _ in range(seq_len):
            
            # Cliffords for the ith layer
            command_list_i = []
    
            # sample random Cliffords
            if parallel == False:
                rand_Cliffords = [str(g) for g in np.random.choice(Clifford_group_list, size=len(qubits))]
            elif parallel == True:
                # same Clifford for every qubit pair
                g = str(np.random.choice(Clifford_group_list))
                rand_Cliffords = [g for _ in range(len(qubits))]
            
            for j, cliff_str in enumerate(rand_Cliffords):
                
                # update sequence Clifford for qubit q
                unitary_list[j] = TQ_Clifford_group[cliff_str] @ unitary_list[j]
    
                # convert to command for Guppy program
                q0, q1 = qubits[j][0], qubits[j][1]
                commands_q_pair_j = cliff_str_to_list(cliff_str, q0, q1)
                command_list_i.append(commands_q_pair_j)
            
            command_list.append(command_list_i)
    
        # apply inverse Cliffords
        inv_commands = []
        for j in range(len(qubits)):
            q0, q1 = qubits[j][0], qubits[j][1]
            U = unitary_list[j]
            
            # find inverse
            for g_inv in Clifford_group_list:
                V = TQ_Clifford_group[g_inv]
                dist = 1 - (np.abs(np.trace(U @ V))/4)**2
                if dist < 10**(-8):
                    break
            
            inv_cliff_list_j = cliff_str_to_list(g_inv, q0, q1)
            inv_commands.append(inv_cliff_list_j)
        
        command_list.append(inv_commands)
        
        # pad command_list with dummy commands
        max_co_length = 0
        for i in range(seq_len+1):
            for j in range(n_q_pairs):
                co_list = command_list[i][j]
                if len(co_list) > max_co_length:
                    max_co_length = len(co_list)
        for i in range(seq_len+1):
            for j in range(n_q_pairs):
                co_list = command_list[i][j]
                for k in range(max_co_length-len(co_list)):
                    command_list[i][j].append((0,0,0))
                        
    
        # apply final X's based on chosen survival state
        final_Xs = [0 for _ in range(n_qubits)]
        for j, q_pair in enumerate(qubits):
            for i in [0, 1]:
                if surv_state[j][i] == '1':
                    q_i = q_pair[i]
                    final_Xs[q_i] = 1
                    
        
        @guppy
        def main() -> None:
            q = array(qubit() for _ in range(comptime(n_qubits)))
            
            for i in range(comptime(seq_len)+1):
                for j in range(comptime(n_q_pairs)):
                    commands = comptime(command_list)[i][j]
                    for gate_id, q0_id, q1_id in commands:
                        if gate_id == 1:
                            x(q[q0_id])
                        elif gate_id == 2:
                            y(q[q0_id])
                        elif gate_id == 3:
                            z(q[q0_id])
                        elif gate_id == 4:
                            h(q[q0_id])
                        elif gate_id == 5:
                            s(q[q0_id])
                        elif gate_id == 6:
                            sdg(q[q0_id])
                        elif gate_id == 7:
                            zz_phase(q[q0_id], q[q1_id], angle(0.5))
                
                if comptime(barriers):
                    barrier(q)
            
            # final Xs
            for q_i in range(comptime(n_qubits)):
                if comptime(final_Xs)[q_i] == 1:
                    x(q[q_i])
            
            # measure
            measure_and_record_leakage(q, comptime(meas_leak))
    
        # return the compiled program (HUGR)
        return main.compile()
    
        
    # Analysis methods
    
    def analyze_results(self, error_bars=True, plot=True, display=True, save=True, **kwargs):
        
        num_resamples = kwargs.get('num_resamples', 100)
        
        results = self.results
        marginal_results = at.marginalize_hists(self.qubits, results, mar_exp_out=True)
        
        # postselect leakage
        if self.options['measure_leaked'] == True:
            self.marginal_results = [at.postselect_leakage(mar_re) for mar_re in marginal_results]
            raw_marginal_results = marginal_results
            self.postselection_rates = []
            self.postselection_rates_stds = []
            for mar_re in marginal_results:
                ps_rates, ps_stds = at.get_postselection_rates(mar_re, self.setting_labels)
                self.postselection_rates.append(ps_rates)
                self.postselection_rates_stds.append(ps_stds)
            leakage_rates, leakage_stds = estimate_leakage_rates(self.postselection_rates,
                                                                 self.postselection_rates_stds,
                                                                 self.seq_lengths)
            self.leakage_rates = leakage_rates
            self.leakage_rates_stds = leakage_stds
            self.mean_leakage_rate = float(np.mean(leakage_rates))
            self.mean_leakage_std = float(np.sqrt(sum([s**2 for s in leakage_stds]))/len(leakage_stds))
            
        else:
            self.marginal_results = marginal_results
        
        
        self.success_probs = [at.get_success_probs(hists) for hists in self.marginal_results]
        self.avg_success_probs = [at.get_avg_success_probs(hists) for hists in self.marginal_results]
        if self.options['measure_leaked'] == True:
            raw_success_probs = [at.get_success_probs(hists) for hists in raw_marginal_results]
            raw_avg_success_probs = [at.get_avg_success_probs(hists) for hists in raw_marginal_results]
        
        # estimate fidelity
        fid_avg = [at.estimate_fidelity(avg_succ_probs, rescale_fidelity=True) for avg_succ_probs in self.avg_success_probs]
        mean_fid_avg = float(np.mean(fid_avg))
        if self.options['measure_leaked'] == True:
            self.raw_fid_avg = [at.estimate_fidelity(raw_avg_succ_probs, rescale_fidelity=True) for raw_avg_succ_probs in raw_avg_success_probs]
            self.raw_mean_fid_avg = float(np.mean(self.raw_fid_avg))
            self.fid_avg = [fid_avg[j] - leakage_rates[j] for j in range(len(self.qubits))]
            self.mean_fid_avg = mean_fid_avg - self.mean_leakage_rate
        else:
            self.fid_avg = fid_avg
            self.mean_fid_avg = mean_fid_avg
        
        # compute error bars
        if error_bars == True:
            self.error_data = [compute_error_bars(hists, num_resamples) for hists in self.marginal_results]
            fid_avg_std = [data['avg_fid_std'] for data in self.error_data]
            mean_fid_avg_std = float(np.sqrt(sum([s**2 for s in fid_avg_std])))/len(fid_avg_std)
            
            if self.options['measure_leaked'] == True:
                raw_error_data = [compute_error_bars(hists, num_resamples) for hists in raw_marginal_results]
                self.raw_fid_avg_std = [data['avg_fid_std'] for data in raw_error_data]
                self.raw_mean_fid_avg_std = float(np.sqrt(sum([s**2 for s in self.raw_fid_avg_std]))/len(self.raw_fid_avg_std))
                self.fid_avg_std = [float(np.sqrt(fid_avg_std[j]**2 + self.leakage_rates_stds[j]**2)) for j in range(len(self.qubits))]
                self.mean_fid_avg_std = float(np.sqrt(mean_fid_avg_std**2 + self.mean_leakage_std**2))
            else:
                self.fid_avg_std = fid_avg_std
                self.mean_fid_avg_std = mean_fid_avg_std
        
        # make plots
        if plot == True:
            self.plot_results(error_bars=error_bars, **kwargs)
            if self.options['measure_leaked'] == True:
                self.plot_postselection_rates(**kwargs)
            
        # display results
        if display == True:
            self.display_results(error_bars=error_bars, **kwargs)
    
            
        if save:
            self.save()
            
    
    def plot_results(self, error_bars=True, **kwargs):
        
        
        title = kwargs.get('title', self.protocol + ' Decays')
        labels = self.qubits
        
        seq_len = self.seq_lengths
        avg_success_probs = self.avg_success_probs
        
        if error_bars:
            avg_success_stds = [{L:error_data['success_probs_stds'][L] for L in seq_len}
                                for error_data in self.error_data]
        else:
            avg_success_stds = None
            
        at.plot_TQ_decays(seq_len, avg_success_probs, avg_success_stds,
                          title=title, labels=labels, **kwargs)
    
    
    def plot_postselection_rates(self, display=True, **kwargs):
        
        ylim = kwargs.get('ylim2', None)
        title = kwargs.get('title2', f'{self.protocol} Leakage Postselection Rates')
        
        # define fit function
        def fit_func(L, a, f):
            return a*f**L
        
        # Create a colormap
        cmap = cm.turbo

        # Normalize color range from 0 to num_lines-1
        cnorm = mcolors.Normalize(vmin=0, vmax=len(self.qubits)-1)
        
        x = self.seq_lengths
        xfit = np.linspace(x[0], x[-1], 100)
        
        for j, ps_rates in enumerate(self.postselection_rates):
        
            y = [ps_rates[L] for L in x]
            yerr = [self.postselection_rates_stds[j][L] for L in x]
            q_pair = self.qubits[j]
        
            # perform best fit
            popt, pcov = curve_fit(fit_func, x, y, p0=[0.4, 0.9], bounds=([0,0], [1,1]), sigma=yerr)
            yfit = fit_func(xfit, *popt)
            plt.errorbar(x, y, yerr=yerr, fmt='o', color=cmap(cnorm(j)), label=f'{q_pair}')
            plt.plot(xfit, yfit, '-', color=cmap(cnorm(j)))
        
        plt.title(title)
        plt.ylabel('Postselection Rate')
        plt.xlabel('Sequence Length')
        plt.xticks(ticks=x, labels=x)
        plt.ylim(ylim)
        plt.legend()
        plt.show()
        
            
    
    
    def display_results(self, error_bars=True, **kwargs):
        
        prec = kwargs.get('precision', 5)
        verbose = kwargs.get('verbose', True)
        
        print('TQ Average Infidelities:\n' + '-'*34)
        for j, f_avg in enumerate(self.fid_avg):
            q_pair = self.qubits[j]
            message = f'qubits {q_pair}: {round(1-f_avg, prec)}'
            if error_bars == True:
                f_std = self.error_data[j]['avg_fid_std']
                message += f' +/- {round(f_std, prec)}'
            print(message)
        avg_message = '-'*34 + '\nZone Average Infidelity:  '
        mean_infid = 1-self.mean_fid_avg
        avg_message += f'{round(mean_infid,prec)}'
        if error_bars == True:
            mean_fid_avg_std = self.mean_fid_avg_std
            avg_message += f' +/- {round(mean_fid_avg_std, prec)}'
        print(avg_message)
        
        if self.options['measure_leaked'] == True:
            leak_rate = self.mean_leakage_rate
            leak_std = self.mean_leakage_std
            if verbose:
                print('\nTQ Leakage Rates:\n' + '-'*34)
                for j, leak_r in enumerate(self.leakage_rates):
                    q_pair = self.qubits[j]
                    leak_s = self.leakage_rates_stds[j]
                    print(f'qubits {q_pair}: {round(leak_r, prec)} +/- {round(leak_s, prec)}')
            print('-'*34 + f'\nZone Average Leakage Rate: {round(leak_rate, prec)} +/- {round(leak_std, prec)}')



def cliff_str_to_list(cliff_str: str, q0: int, q1: int):
    """ convert Clifford str representation into list representation
        for SQ gates: (gate_id, q_id, 0)
        for TQ gates: (gate_id, q0_id, q1_id)
        x: 1
        y: 2
        z: 3
        h: 4
        s: 5
        sdg: 6
        rzz: 7
    """

    gate_list = []
    i0 = 0
    for i in range(len(cliff_str)):
        if cliff_str[i] == '*':
            gate_list.append(cliff_str[i0:i])
            i0 = i+1
    gate_list.append(cliff_str[i0:])

    cliff_list = []
    for gate in reversed(gate_list):
        if 'rzz' not in gate:
            if '0' in gate:
                q = q0
            elif '1' in gate:
                q = q1

            if 'x' in gate and 'cx' not in gate:
                cliff_list.append((1,q,0))
            elif 'y' in gate:
                cliff_list.append((2,q,0))
            elif 'z' in gate and 'rzz' not in gate:
                cliff_list.append((3,q,0))
            elif 'h' in gate:
                cliff_list.append((4,q,0))
            elif 's' in gate and 'sdg' not in gate:
                cliff_list.append((5,q,0))
            elif 'sdg' in gate:
                cliff_list.append((6,q,0))

        elif 'rzz' in gate:
            cliff_list.append((7, q0,q1))

    return cliff_list


### Analysis functions

def estimate_leakage_rates(post_rates, post_stds, seq_lengths):
    
    leakage_rates = []
    leakage_stds = []
    
    # define fit function
    def fit_func(L, a, f):
        return a*f**L
    
    for j, ps_rates in enumerate(post_rates):
        
        y = [ps_rates[L] for L in seq_lengths]
        yerr = [post_stds[j][L] for L in seq_lengths]
        # perform best fit
        popt, pcov = curve_fit(fit_func, seq_lengths, y, p0=[0.9, 0.9], bounds=([0,0], [1,1]), sigma=yerr)
        leakage_rates.append(2*(1-popt[1])/3) # 1.5 TQ per Clifford
        leakage_stds.append(float(2*np.sqrt(pcov[1][1]))/3)
    
    return leakage_rates, leakage_stds



def compute_error_bars(hists, num_resamples=100):
    
    
    boot_hists = bootstrap(hists, num_resamples=num_resamples)
    boot_avg_succ_probs = [at.get_avg_success_probs(b_h) for b_h in boot_hists]
    boot_avg_fids = [at.estimate_fidelity(avg_succ_prob, rescale_fidelity=True)
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
    seq_len = list(set([name[-3] for name in hists]))
    
    boot_hists = []
    for i in range(num_resamples):
        
        # first do non-parametric resampling
        hists_resamp = {}
        for L in seq_len:
            # make list of exp names to resample from
            circ_list = []
            for name in hists:
                if name[-3] == L:
                    circ_list.append(name)
            # resample from circ_list
            seq_reps = len(circ_list)
            resamp_circs = np.random.choice(seq_reps, size=seq_reps)
            for rep, rep2 in enumerate(resamp_circs):
                circ = circ_list[rep2]
                name_resamp = (L, rep, circ[-1])
                outcomes = hists[circ]
                hists_resamp[name_resamp] = outcomes
        
        # do parametric resample
        boot_hists.append(bs.resample_hists(hists_resamp))
    
    return boot_hists



