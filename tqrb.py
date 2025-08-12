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
from guppylang.std.builtins import array, comptime, result
from guppylang.std.quantum import measure_array, qubit, h, z, x, y, s, sdg
from guppylang.std.qsystem import measure_leaked, zz_phase
from hugr.package import FuncDefnPointer

from experiment import Experiment
import analysis_tools as at
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
        meas_leak = self.options['measure_leaked']
        n_qubits = self.n_qubits
        qubits = self.qubits
        
        # track unitaries in each gate zone
        unitary_list = [np.diag([1,1,1,1]) for _ in range(len(qubits))]
        command_list = [] # commands for Guppy program
    
        for _ in range(seq_len):
    
            # sample random Cliffords
            rand_Cliffords = [str(g) for g in np.random.choice(Clifford_group_list, size=len(qubits))]
            for j, cliff_str in enumerate(rand_Cliffords):
                
                # update sequence Clifford for qubit q
                unitary_list[j] = TQ_Clifford_group[cliff_str] @ unitary_list[j]
    
                # convert to command for Guppy program
                q0, q1 = qubits[j][0], qubits[j][1]
                cliff_list = cliff_str_to_list(cliff_str, q0, q1)
                for com in cliff_list:
                    command_list.append(com)
    
        # apply inverse Cliffords
        for j in range(len(qubits)):
            q0, q1 = qubits[j][0], qubits[j][1]
            U = unitary_list[j]
            
            # find inverse
            for g_inv in Clifford_group_list:
                V = TQ_Clifford_group[g_inv]
                dist = 1 - (np.abs(np.trace(U @ V))/4)**2
                if dist < 10**(-8):
                    break
            
            cliff_list = cliff_str_to_list(g_inv, q0, q1)
            for com in cliff_list:
                command_list.append(com)
    
        # apply final X's based on chosen survival state
        for j, q_pair in enumerate(qubits):
            for i in [0, 1]:
                if surv_state[j][i] == '1':
                    q_i = q_pair[i]
                    command_list.append((1,q_i,0))
        
        @guppy
        def main() -> None:
            q = array(qubit() for _ in range(comptime(n_qubits)))
    
            for gate_id, q0_id, q1_id in comptime(command_list):
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
                    #cx(q[q0_id], q[q1_id])
                    
            
            # measure
            if comptime(meas_leak):
                meas_leaked_array = array(measure_leaked(q_i) for q_i in q)
                for m in meas_leaked_array:
                    if m.is_leaked():
                        m.discard()
                        result("c", 2)
                    else:
                        result("c", m.to_result().unwrap())
            else:
                b_str = measure_array(q)
                # report measurement outcomes
                for b in b_str:
                    result("c", b)
    
        # return the compiled program (HUGR)
        return main.compile()
    
        
    # Analysis methods
    
    def analyze_results(self, error_bars=True, plot=True, display=True, **kwargs):
        
        
        results = self.results
        marginal_results = at.marginalize_hists(self.qubits, results, mar_exp_out=True)
        # postselect leakage
        if self.options['measure_leaked'] == True:
            self.marginal_results = [at.postselect_leakage(mar_re) for mar_re in marginal_results]
            self.postselection_rates = []
            self.postselection_rates_stds = []
            for mar_re in marginal_results:
                ps_rates, ps_stds = at.get_postselection_rates(mar_re, self.setting_labels)
                self.postselection_rates.append(ps_rates)
                self.postselection_rates_stds.append(ps_stds)
        else:
            self.marginal_results = marginal_results
        
        
        self.success_probs = [at.get_success_probs(hists) for hists in self.marginal_results]
        self.avg_success_probs = [at.get_avg_success_probs(hists) for hists in self.marginal_results]
        self.fid_avg = [at.estimate_fidelity(avg_succ_probs, rescale_fidelity=True) for avg_succ_probs in self.avg_success_probs]
        self.mean_fid_avg = float(np.mean(self.fid_avg))
        
        # compute error bars
        if error_bars == True:
            self.error_data = [compute_error_bars(hists) for hists in self.marginal_results]
            self.fid_avg_std = [data['avg_fid_std'] for data in self.error_data]
            self.mean_fid_avg_std = float(np.sqrt(sum([s**2 for s in self.fid_avg_std])))/len(self.fid_avg_std)
        
        # make plots
        if plot == True:
            self.plot_results(error_bars=error_bars, **kwargs)
            
        # display results
        if display == True:
            self.display_results(error_bars=error_bars, **kwargs)
            
        # leakage analysis
        if self.options['measure_leaked'] == True:
            self.plot_postselection_rates(display=display, **kwargs)
            
    
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
        leakage_rates = []
        leakage_stds = []
        
        for j, ps_rates in enumerate(self.postselection_rates):
        
            y = [ps_rates[L] for L in x]
            yerr = [self.postselection_rates_stds[j][L] for L in x]
            q_pair = self.qubits[j]
        
            # perform best fit
            popt, pcov = curve_fit(fit_func, x, y, p0=[0.4, 0.9], bounds=([0,0], [1,1]), sigma=yerr)
            leakage_rates.append(1-popt[1])
            leakage_stds.append(float(np.sqrt(pcov[1][1])))
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
        
        self.leakage_rates = leakage_rates
        self.leakage_rates_stds = leakage_stds
        self.mean_leakage_rate = float(np.mean(leakage_rates))
        self.mean_leakage_std = float(np.sqrt(sum([s**2 for s in leakage_stds]))/len(leakage_stds))
        
        if display:
            leak_rate = self.mean_leakage_rate
            leak_std = self.mean_leakage_std
            print(f'Zone average leakge rate: {round(leak_rate, 5)} +/- {round(leak_std, 5)}')
    
    
    def display_results(self, error_bars=True, **kwargs):
        
        prec = kwargs.get('precision', 5)
        
        print('TQ Average Infidelities\n' + '-'*34)
        for j, f_avg in enumerate(self.fid_avg):
            q_pair = self.qubits[j]
            message = f'qubits {q_pair}: {round(1-f_avg, prec)}'
            if error_bars == True:
                f_std = self.error_data[j]['avg_fid_std']
                message += f' +/- {round(f_std, prec)}'
            print(message)
        avg_message = '-'*34 + '\nZone average:  '
        mean_infid = 1-self.mean_fid_avg
        avg_message += f'{round(mean_infid,prec)}'
        if error_bars == True:
            mean_fid_avg_std = self.mean_fid_avg_std
            avg_message += f' +/- {round(mean_fid_avg_std, prec)}'
        print(avg_message)



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

def compute_error_bars(hists):
    
    
    boot_hists = bootstrap(hists)
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



