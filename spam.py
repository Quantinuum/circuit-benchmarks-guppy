# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:44:00 2025

simple state prep and measurement (SPAM) experiment

@author: Karl.Mayer
"""


import numpy as np
import matplotlib.pyplot as plt

from guppylang import guppy
from guppylang.std.builtins import array, barrier, comptime, result
from guppylang.std.quantum import measure_array, qubit, x
from guppylang.std.qsystem import measure_leaked
from hugr.package import FuncDefnPointer

from experiment import Experiment
from leakage_measurement import measure_and_record_leakage


class SPAM_Experiment(Experiment):
    
    def __init__(self, n_qubits, **kwargs):
        super().__init__(**kwargs)
        self.protocol = 'SPAM'
        self.parameters = {'n_qubits':n_qubits}
        self.n_qubits = n_qubits
        self.rounds = kwargs.get('rounds', 1)
        self.setting_labels = ('round_index', 'surv_state')
        
        self.options['barriers'] = True
        
    
    def add_settings(self, **kwargs):
        
        self.rounds = kwargs.get('rounds', self.rounds)
        n = self.n_qubits
        
        for r in range(self.rounds):
    
            # sample random outputs
            rand_bits = np.random.choice(['0','1'], n)
            surv_state = ''
            for j in range(n):
                surv_state += str(rand_bits[j])
            
            surv_state2 = ''
            for j in range(n):
                if surv_state[j] == '0':
                    surv_state2 += '1'
                elif surv_state[j] == '1':
                    surv_state2 += '0'
    
            
            self.add_setting((r, surv_state))
            self.add_setting((r, surv_state2))
            
    
    def make_circuit(self, setting: tuple) -> FuncDefnPointer:
        
        surv_state = setting[1]
        barriers = self.options['barriers']
        meas_leak = self.options['measure_leaked']
        n_qubits = self.n_qubits
        
        assert n_qubits == len(surv_state), "len(surv_state) must equal n_qubits"
        
        # convert string to list of ints
        survival_state = []
        for j in range(n_qubits):
            survival_state.append(int(surv_state[j]))
        
        @guppy
        def main() -> None:
            
            q = array(qubit() for _ in range(comptime(n_qubits)))
            
            for q_i in range(comptime(n_qubits)):
                if comptime(survival_state)[q_i] == 1:
                    x(q[q_i])
            
            if comptime(barriers):
                barrier(q)
            
            # measure
            measure_and_record_leakage(q, comptime(meas_leak))
    
        # return the compiled program (HUGR)
        return main.compile()
            
    
    def analyze_results(self, plot=True, display=True, save=True, **kwargs):
        
        n = self.n_qubits
        
        
        meas_leak = self.options['measure_leaked']
            
        # create list of SPAM probabilities for the n qubits
        SPAM_probs = []
        if meas_leak:
            leakage_rates = [{} for _ in range(n)]
        for q_i in range(n):
            probs = {'0':[], '1':[]}
            for setting in self.results:
                surv_state = setting[1]
                outcomes = self.results[setting]
                shots = sum(outcomes.values())
                exp_out_bit = surv_state[q_i]
                counts = 0
                
                if meas_leak:
                    leaked_shots = 0    
                    for b_str in outcomes:
                        if b_str[q_i] == '2':
                            leaked_shots += outcomes[b_str]
                        if b_str[q_i] == exp_out_bit:
                            counts += outcomes[b_str]
                    probs[exp_out_bit].append(counts/(shots-leaked_shots))
                    leakage_rates[q_i][setting] = leaked_shots/shots
                
                else:
                    for b_str in outcomes:
                        if b_str[q_i] == exp_out_bit:
                            counts += outcomes[b_str]
                    probs[exp_out_bit].append(counts/shots)
                    
                    
            probs = {'0':float(np.mean(probs['0'])), '1':float(np.mean(probs['1']))}
            SPAM_probs.append(probs)
        
        self.SPAM_probs = SPAM_probs
        self.avg_SPAM_probs = {'0':float(np.mean([sp['0'] for sp in SPAM_probs])),
                               '1':float(np.mean([sp['1'] for sp in SPAM_probs]))}
        
        if plot:
            self.plot_results(**kwargs)
            
        if display:
            self.display_results(**kwargs)
            
        # leakage analysis
        if self.options['measure_leaked'] == True:
            leakage_probs = {'0':[], '1':[]}
            for q_i in range(n):
                for sett in leakage_rates[q_i]:
                    exp_out = sett[1][q_i]
                    rate = leakage_rates[q_i][sett]
                    leakage_probs[exp_out].append(rate)
            self.leakage_rates = {b_str:float(np.mean(leakage_probs[b_str])) for b_str in ['0', '1']}
            tot_shots = self.shots*self.rounds
            self.leakage_stds = {}
            for b_str in ['0', '1']:
                p_eff = tot_shots*self.leakage_rates[b_str]/(tot_shots+2)
                p_std = float(np.sqrt(p_eff*(1-p_eff)/tot_shots))
                self.leakage_stds[b_str] = p_std
            L0 = self.leakage_rates['0']
            L1 = self.leakage_rates['1']
            L0_std = self.leakage_stds['0']
            L1_std = self.leakage_stds['1']
            print('Leakage Rates:')
            print(f'p(L|0) = {round(L0,6)} +/- {round(L0_std, 6)}')
            print(f'p(L|1) = {round(L1,6)} +/- {round(L1_std, 6)}')
            
        if save:
            self.save()
        
        
    def plot_results(self, **kwargs):
        
        ylim = kwargs.get('ylim', None)
        
        spam_probs = self.SPAM_probs

        y1 = [1-sp['0'] for sp in spam_probs]
        y2 = [1-sp['1'] for sp in spam_probs]
        y1mean = float(np.mean(y1))
        y2mean = float(np.mean(y2))
        x = np.array(range(self.n_qubits))
        w = 0.4 # width
        
        plt.bar(x-w/2, y1, width=w, label='P(1|0)', color='b')
        plt.bar(x+w/2, y2, width=w, label='P(0|1)', color='g')
        plt.axhline(y1mean, linestyle='--', color='b', label='mean P(1|0)')
        plt.axhline(y2mean, linestyle='--', color='g', label='mean P(0|1)')
        if self.n_qubits <= 20:
            plt.xticks(ticks=x, labels=x)
        plt.xlabel('Qubit')
        plt.ylabel('Error Probability')
        plt.ylim(ylim)
        plt.legend()
        plt.show()
        
    
    def display_results(self, **kwargs):
        
        precision = kwargs.get('precision', 4)
        
        tot_shots = self.n_qubits*self.rounds*self.shots
        e0 = 1 - self.avg_SPAM_probs['0']
        e1 = 1 - self.avg_SPAM_probs['1']
        e0_std = float(np.sqrt(e0*(1-e0)/tot_shots))
        e1_std = float(np.sqrt(e1*(1-e1)/tot_shots))
        
        print('Average SPAM Errors:')
        print(f'prob(1|0) = {round(e0, precision)} +/- {round(e0_std, precision)}')
        print(f'prob(0|1) = {round(e1, precision)} +/- {round(e1_std, precision)}')
        
        
        
        