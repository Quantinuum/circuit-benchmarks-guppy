# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 12:35:18 2025

GHZ state fidelity test

@author: Karl.Mayer
"""

import numpy as np
import matplotlib.pyplot as plt

from guppylang import guppy
from guppylang.std.builtins import array, barrier, comptime, result
from guppylang.std.quantum import measure_array, qubit, cx, h, rz, ry
from guppylang.std.angles import angle, pi

from experiment import Experiment


class GHZ_Experiment(Experiment):
    
    def __init__(self, n_qubits, **kwargs):
        super().__init__(**kwargs)
        self.protocol = 'GHZ fidelity'
        self.parameters = {'n_qubits':n_qubits}
        self.n_qubits = n_qubits
        self.setting_labels = ('circuit_index', 'meas_basis')
        
        
    def add_settings(self):
        
        meas_bases = []
        for k in range(1, self.n_qubits+1):
            meas_bases.append(0)
            meas_bases.append(k)
        
        for m, basis in enumerate(meas_bases):
            sett = (m, basis)
            self.add_setting(sett)
        
    
    def make_circuit(self, setting: tuple):
        """ 
        meas_basis: if 0, measure in Z,
        otherwise if k for 0<k<=n, measure in cos(pi*k/n)X + sin(pi*k/n)Y
        """
        
        meas_basis = setting[1]
        n_qubits = self.n_qubits
        
        depth = int(np.ceil(np.log2(n_qubits))) # for log-depth construction
        con = [] # array of control qubit indices
        tar = [] # array of target qubit indices
        ang = -meas_basis/n_qubits # angle for final rotation
    
        for i in range(depth):
            for j in range(2**i):
                if j+2**i < n_qubits:
                    con.append(j)
                    tar.append(j+2**i)
    
        @guppy
        def main() -> None:
            q = array(qubit() for _ in range(comptime(n_qubits)))
    
            # prepare GHZ state
            h(q[0])
            for i in range(comptime(n_qubits-1)):
                cx(q[comptime(con)[i]], q[comptime(tar)[i]])
            
            barrier(q)
            if comptime(meas_basis) > 0:
                for i in range(comptime(n_qubits)):
                    rz(q[i], angle(comptime(ang)))
                    ry(q[i], -pi/2)
                    
            # measure
            b_str = measure_array(q)
    
            # report measurement outcomes
            for b in b_str:
                result("c", b)
    
        # return the compiled program (HUGR)
        return main.compile()
    
    
    # Analysis Methods
    
    def analyze_results(self, plot=True, display=True, **kwargs):
        
        # estimate fidelity and std
        f, std = fidelity(self.results)
        self.fid = f    
        self.fid_std = std
        
        # display results
        if display == True:
            n = self.n_qubits
            print(f'GHZ n={n} Fidelity = {round(f,4)} +/- {round(std,4)}\n')
        
        # plot results
        if plot == True:
            self.plot_populations(**kwargs)
            self.plot_parities(**kwargs)
            
    
    def plot_populations(self, **kwargs):
        
        ylim = kwargs.get('ylim', (0,0.55))
        save_plots = kwargs.get('save_plots', False)
        
        results = self.results
        
        # read in number of qubits and shots
        n = self.n_qubits
        
        pops = {}
        for setting in results:
            if setting[1] == 0:
                outcomes = results[setting]
                pops = merge_outcomes(pops, outcomes)
        
        shots = sum(pops.values())
        if n <= 12:
            x = ['0'*n, 'everything else', '1'*n]
        elif n > 12:
            x = ['00...0', 'everything else', '11...1']
            
        y0 = pops['0'*n]/shots
        y1 = pops['1'*n]/shots
        y_else = 1-y0-y1
        
        y = [y0, y_else, y1]
        yerr = [np.sqrt(p*(1-p)/shots) for p in y]
        
        plt.bar(x,y, yerr=yerr)
        plt.ylim(ylim)
        plt.ylabel('Probability')
        plt.title(f'N = {n} GHZ populations')
        if save_plots == True:
            plt.savefig(f'GHZ_N{n}_populations.pdf', format='pdf')
        plt.show()
        
        
    def plot_parities(self, **kwargs):
        
        save_plots = kwargs.get('save_plots', False)
        
        results = self.results
    
        # read in n
        n = self.n_qubits
        
        parities = {}
        stds = {}
        thetas = [i for i in range(1,n+1)]
    
        for theta in thetas:
            for name in results:
                if name[1] == theta:
                    outcomes = results[name]
                    shots = sum(outcomes.values())
                    exp_val = exp_value(outcomes)
                    p = (exp_val + 1)/2
                    var = 4*p*(1-p)/shots
                    parities[theta] = exp_val
                    stds[theta] = np.sqrt(var)
        
        # make parity plot
        
        x = [int(i) for i in parities]
        p_plus = []
        p_minus = []
        for i in parities:
            if int(i)%2==0:
                p_plus.append(parities[i])
            elif int(i)%2==1:
                p_minus.append(parities[i])
        
        y = list(parities.values())
        yerr = list(stds.values())
        y_plus = np.mean(p_plus)
        y_minus = np.mean(p_minus)
        plt.errorbar(x, y, yerr=yerr, fmt='bo')
        plt.axhline(y_plus, color='b', linestyle='--')
        plt.axhline(y_minus, color='b', linestyle='--')
        plt.xlabel('Meas angle (in units of pi/N)')
        plt.ylabel('Parity')
        plt.title(f'N={n} GHZ Parity')
        plt.ylim((-1.1,1.1))
        if save_plots == True:
            plt.savefig(f'GHZ_N{n}_parities.pdf', format='pdf')
        plt.show()
        
        
# additional functions

# auxilliary function for computing expectation value
# product of Pauli operators
def exp_value(outcomes):
    
    exp_val = 0.0
    shots = sum(outcomes.values())
    for b_str in outcomes:
        parity = (-1)**(b_str.count('1'))
        counts = outcomes[b_str]
        exp_val += parity*counts/shots
    
    return round(exp_val,6)


def fidelity(results):
    
    # read in number of qubits and shots
    n = len(list(list(results.values())[0].keys())[0])
    
    num_pop_meas = 0
    num_par_meas = 0
    for name in results:
        meas_basis = name[1]
        if meas_basis == 0:
            num_pop_meas += 1
        elif meas_basis > 0:
            num_par_meas += 1
        
    N1 = num_pop_meas
    N2 = num_par_meas
        
    f = 0.0 # fidelity estimate
    var = 0.0 # variance of fidelity estimate
    
    for name in results:
        meas_basis = name[1]
        outcomes = results[name]
        shots = sum(outcomes.values())
            
        if meas_basis == 0:
            p = 0.0 # success prob
            if '0'*n in outcomes:
                p0 = outcomes['0'*n]/shots
                p += p0
            if '1'*n in outcomes:
                p1 = outcomes['1'*n]/shots
                p += p1
            f += (1/2)*p/N1
            var += (1/4)*((p*(1-p))/shots)/N1**2
            
        
        # for phase scan method
        elif meas_basis > 0:
            sign = (-1)**meas_basis
            # compute expectation value of spin operator
            exp_val = exp_value(outcomes)
            # update fidelity estimate and variance
            f += (1/2)*exp_val*sign/N2
            var += (1/2)*((1+exp_val)*(1-exp_val)/shots)/N2**2
    
        
    std = np.sqrt(var)
    
    return f, std


def merge_outcomes(out1, out2):
    """ combine outcomes from different circuit executions
        out1, out2 (dict)
    """
    
    outcomes = out1
    for b_str in out2:
        counts = out2[b_str]
        if b_str in outcomes:
            outcomes[b_str] += counts
        elif b_str not in outcomes:
            outcomes[b_str] = counts
    
    return outcomes
    
    
        
        
        
            

