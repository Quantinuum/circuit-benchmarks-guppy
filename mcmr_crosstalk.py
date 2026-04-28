# -*- coding: utf-8 -*-
"""
Created on Mon Oct 2025

Mid-circuit measurement and reset (MCMR) crosstalk benchmarking for Helios-1

@author: Victor Colussi
"""


import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

from guppylang import guppy
from guppylang.std.builtins import array, barrier, comptime, result
from guppylang.std.quantum import measure_array, qubit, x, y, z, h, rz, rx, ry, pi
from guppylang.std.qsystem import measure_and_reset
from guppylang.std.qsystem import measure_leaked, reset
from hugr.package import FuncDefnPointer

from Clifford_tools import apply_SQ_Clifford, apply_SQ_Clifford_inv

from experiment import Experiment
import bootstrap as bs
from leakage_measurement import measure_and_record_leakage
from analysis_tools import postselect_leakage, get_postselection_rates

from qtm_platform.ops import order_in_zones


class MCMR_Crosstalk_Experiment(Experiment):
    
    def __init__(self, focus_qubits, seq_lengths, **kwargs):
        super().__init__(**kwargs)
        self.focus_qubits = focus_qubits #focus_qubits
        self.n_qubits = 98 # 98 total qubits on Helios-1
        self.n_zone_qubits = 16 # total qubits in the gatezones.  
        self.n_ring_qubits = self.n_qubits - self.n_zone_qubits # there are 82 qubits stored in the ring for Helios-1.
        self.probe_qubits = self.get_probe_qubits() 
        self.init_states = 2 # Poles on Bloch sphere
        self.protocol = f'MCMR Crosstalk: Focus qubit q{self.focus_qubits}'
        self.parameters = {'n_qubits':self.n_qubits,
                           'seq_lengths':seq_lengths,
                           'init_states':self.init_states}
        
        self.which_qubits = np.sort( np.concatenate( (self.probe_qubits, self.focus_qubits), axis = 0) )
        self.seq_lengths = seq_lengths
        self.setting_labels = ('seq_len', 'init_state', 'surv_state')
        
    def get_probe_qubits(self):
        ''' Given a  list of focus qubits, this constructs the list of probe qubits from the remainder. 
        '''

        probe_qubits = [q for q in range(self.n_qubits)]

        for focus_qubit in self.focus_qubits:
            probe_qubits.remove(focus_qubit)
        
        assert len(probe_qubits) + len(self.focus_qubits) == self.n_qubits, "MCMR Crosstalk code valid only for up to 16 qubits, currently."

        return probe_qubits 
        


    def add_settings(self):
        
        for seq_len in self.seq_lengths:
            for init_state in range(self.init_states):
                surv_state = '' 
                for _ in range(self.n_qubits):
                    surv_state += str(np.random.choice(['0'])) 
                
                sett = (seq_len, init_state, surv_state)
                self.add_setting(sett)


    def get_setting_gate_index(self, init_state):
        if init_state==0: # |0> 
            gate_index = 0 # identity
        elif init_state==1:  # |1> 
            gate_index = 4 # X
        return gate_index

        
    
    def make_circuit(self, setting: tuple) -> FuncDefnPointer:

        seq_len = setting[0] # number of MCMR's to perform on each target qubit
        init_state = setting[1] # for each init_state we perform a different circuit
        meas_leak = self.options['measure_leaked']


        gate_index = self.get_setting_gate_index(init_state)
        
        n_qubits = self.n_qubits
        n_zone_qubits = self.n_zone_qubits
        n_ring_qubits = self.n_ring_qubits

        focus_qubits = self.focus_qubits

        @guppy  # guppy main program.  
        def main() -> None:
            
            ring_qubits: array[qubit, comptime(n_ring_qubits)] = array(qubit() for _ in range(comptime(n_ring_qubits)))
            zone_qubits: array[qubit, comptime(n_zone_qubits)] = array(qubit() for _ in range(comptime(n_zone_qubits)))    

            for q_i in range(comptime(n_ring_qubits)): # put all ring qubits in a Pauli eigenstate
                if comptime(gate_index) > 0:
                    apply_SQ_Clifford(ring_qubits[q_i], comptime(gate_index))
            
            barrier(ring_qubits) # put into the ring 

            for q_i in range(comptime(n_zone_qubits)): # put all zone qubits in a Pauli eigenstate
                if comptime(gate_index) > 0:
                    apply_SQ_Clifford(zone_qubits[q_i], comptime(gate_index))

               
            for i in range(comptime(seq_len)):
                for j in comptime(focus_qubits): # only MCMR
                    measure_and_reset(zone_qubits[j])
            

            for q_i in range(comptime(n_zone_qubits)): # put all zone qubits back into |0> state
                if comptime(gate_index) > 0:
                    apply_SQ_Clifford_inv(zone_qubits[q_i], comptime(gate_index))
            

            # measure
            measure_and_record_leakage(zone_qubits, comptime(meas_leak))

            for q_i in range(comptime(n_ring_qubits)): # put all qubits back into |0> state
                if comptime(gate_index) > 0:
                    apply_SQ_Clifford_inv(ring_qubits[q_i], comptime(gate_index))
    
            measure_and_record_leakage(ring_qubits, comptime(meas_leak))
    
        # return the compiled program (HUGR)
        return main.compile()


    def analyze_results(self, **kwargs):
        
        self.marginal_results = marginalize_hists(self.n_qubits, self.probe_qubits, self.results)

        # get rates of postselection (# of non-leaked shots over the total number of shots)
        self.postselection_marginal_results = [postselect_leakage(mar_re) for mar_re in self.marginal_results]
        self.postselection_rates = []
        self.postselection_rates_stds = []
        for mar_re in self.marginal_results: 
            ps_rates, ps_stds = get_postselection_rates(mar_re, self.setting_labels)
            self.postselection_rates.append(ps_rates) #it's only the size of the probe qubits
            self.postselection_rates_stds.append(ps_stds) 


        # get the circuit-resolved results
        self.success_probs = []
        self.success_stds = []
        self.avg_success_probs = []
        self.postselect_probs = []
        self.postselect_stds = []
        self.avg_postselect_probs = []
        for j, hists in enumerate(self.marginal_results): # loops over each qubit. 
            succ_probs_j, succ_stds_j, postselect_probs_j, postselect_stds_j = get_success_probs(hists, postselect = True)  
            self.success_probs.append(succ_probs_j)
            self.success_stds.append(succ_stds_j)
            self.postselect_probs.append(postselect_probs_j)
            self.postselect_stds.append(postselect_stds_j)

        self.estimate_error_channels()
            
        
        self.plot_results()
        self.display_results() 



    def plot_results(self, **kwargs):
 
        scale = 1e-4
        shift = 0.07

        qtm_hex = [
            '#E1F6F2', '#A5E5D7', '#6AD3BE', '#30A08E', '#1D605B',
            '#E6E0EC', '#B3A2C7', '#8064A2', '#604A7B', '#403152',
            '#FCDFE4', '#F59EAF', '#E75D72', '#BB4658', '#7C3349',
            '#FFEBDD', '#FFC29A', '#FF9A56', '#E5803C', '#92542A',
            '#c3e1ee', '#96cae1', '#69B3D4', '#548faa', '#4a7d94',
            '#fff2b3', '#ffea81', "#FED402", '#cbaa02', '#b29401',
            '#F2F2F2', '#CACACA', '#7F7F7F', '#2B2B2B', '#000000',
        ]

        custom_colors = ['#6AD3BE', '#69B3D4', '#FF9A56', '#8064A2', '#2B2B2B', '#FED402', '#7F7F7F'] # Red, Yellow, Blue

        self.measured_qubits = {i: [] for i in range(5)} # key is error channel
        self.measured_qubits_std = {i: [] for i in range(5)}
        self.global_inf = 0.0

        for i in range(5):
            j1 = 0
            for j in range(98):
                if j in self.probe_qubits:
                    self.measured_qubits[i].append(self.error_channels[i][j1]/scale) # lth experiment, ith error channel
                    self.measured_qubits_std[i].append(self.error_channels_stds[i][j1]/scale)
                    j1 += 1
                else:
                    self.measured_qubits[i].append(0)
                    self.measured_qubits_std[i].append(0)

        fig, ax = plt.subplots(1,1, figsize=(18, 6))
        handles = []
        self.global_inf = np.mean([(2*self.measured_qubits[0][j] + 2*self.measured_qubits[1][j] + 6*self.measured_qubits[4][j])/6 for j in range(98) if j not in self.focus_qubits])
        for i in range(4): # the error channels
            handles.append(ax.bar(
                np.arange(17) + i*0.9/4,
                [self.measured_qubits[i][j] for j in range(16)] 
                    + [np.mean(self.measured_qubits[i][16:])],
                yerr=[self.measured_qubits_std[i][j] for j in range(16)] 
                    + [np.sqrt(sum([self.measured_qubits_std[i][k]**2 for k in range(16,98)]))/82],
                width=0.9/4,
                align='edge',
                color=custom_colors[i],
                zorder=2
            ))
        handles.append(ax.bar( # the average fidelity bars
        np.arange(0, 17) + 0.45, # the infidelity 
        (
            [(2*self.measured_qubits[0][j] + 2*self.measured_qubits[1][j] + 6*self.measured_qubits[4][j])/6 for j in range(0,16)] 
            + [np.mean([(2*self.measured_qubits[0][j] + 2*self.measured_qubits[1][j] + 6*self.measured_qubits[4][j])/6 for j in range(16,98)])]
        ),
        yerr=(
            [np.sqrt((2*self.measured_qubits_std[0][j])**2 +(2*self.measured_qubits_std[1][j])**2 
                            + (6*self.measured_qubits_std[4][j])**2)/6 for j in range(0,16)] 
            + [np.sqrt(sum(((2*self.measured_qubits_std[0][j])**2 +(2*self.measured_qubits_std[1][j])**2 
                            + (6*self.measured_qubits_std[4][j])**2)/36 for j in range(16,98)))/82]
        ),
        width=0.9,
        align='center',
        edgecolor=custom_colors[4],
        facecolor='white',
        zorder=1,
        linewidth=2,
        alpha=0.5,
        error_kw=dict(ecolor=custom_colors[4], capthick=5)

        ))
        ax.axhline( # the average zone infidelity (horizontal line)
        self.global_inf,
        color=custom_colors[4],
        linestyle='--',
        alpha=0.5
        )
        ax.set_ylabel('Error ($\\times 10^{-4}$)', fontsize=12)
        ax.grid(visible=True, axis="y", linestyle="-", linewidth=0.5, alpha=0.7,zorder = 0)
        ax.set_xticks(np.arange(0.5, 17.5, 1))
        ax.set_xticklabels(['' for i in range(17)])
        ax.set_xlim(0,17)

        ax.legend(handles, ['Bitflip $p(1|0)$','Bitflip $p(0|1)$','Leakage $p(L|0)$','Leakage $p(L|1)$', 'Avg. Infidelity'])
        lb = 0
        ax.set_ylim(lb, 7)#7)
        ax.set_xlabel('Qubit index', fontsize=12)
        ax.set_xticks(np.arange(0.5, 17.5, 1))
        ax.set_xticklabels([str(i) for i in range(16)] + ['ring'])
        fig.suptitle(f'Helios-1 MCMR Crosstalk Results: Target Qubit(s) = {self.focus_qubits}')
        fig.tight_layout()




    def display_results(self):

        prec = 6
        num_mcmrs = len(self.focus_qubits)
        global_std = 0.0
        stds = [np.sqrt( (2*self.measured_qubits_std[0][j])**2 + (2*self.measured_qubits_std[1][j])**2 
                    + (6*self.measured_qubits_std[4][j])**2 )/6 for j in range(self.n_qubits) if j not in self.focus_qubits]
        global_std = np.sqrt(np.sum([std**2 for std in stds]))/len(stds)
        print("Global per MCMR Crosstalk Infidelity (1e-4):  " + f"{round(self.global_inf,prec)/num_mcmrs} +/- {round(global_std,prec)/num_mcmrs}" )

        # global mean and std for each error channel.
        error_channel_label = ["p(1|0)", "p(0|1)","p(L|0)","p(L|1)"]
        global_mean_ec = np.zeros(4)
        global_stds_ec = np.zeros(4)
        num_spectators = len([i for i in range(self.n_qubits) if i not in self.focus_qubits])
        global_mean_ec = [np.mean([self.measured_qubits[j][i] for i in range(98) if i not in self.focus_qubits]) for j in range(4)]
        stds_sq = [ np.sum([self.measured_qubits_std[j][i]**2 for i in range(98) if i not in self.focus_qubits]) for j in range(4)]
        for j in range(4):
            global_stds_ec[j] = np.sqrt(np.sum(stds_sq[j]))/num_spectators
            print(f"Global per MCMR Crosstalk Error Channel {error_channel_label[j]} (1e-4):  " + f"{round(global_mean_ec[j],prec)/num_mcmrs} +/- {round(global_stds_ec[j],prec)/num_mcmrs}")


    def estimate_error_channels(self, **kwargs):
        '''Input:  marginalized histogram for an single qubit.
        Output: process fidelity, state fidelity, depolarizing parameter, computational population 
        ''' 

        n_probe = len(self.probe_qubits) # number of probe qubits

        ### Error channels ###
        # First, output per MCMR success probability for individual circuits
        dict_probs = {L:[] for L in self.seq_lengths}
        dict_stds = {L:[] for L in self.seq_lengths}
        ps_dict_probs = {L:[] for L in self.seq_lengths}
        ps_dict_stds = {L:[] for L in self.seq_lengths}

        fid_init_state = np.zeros([n_probe,self.init_states]) # "F_i" for each qubit (qubit, init_state)
        std_init_state = np.zeros([n_probe,self.init_states])
        ps_fid_init_state = np.zeros([n_probe,self.init_states])
        ps_std_init_state = np.zeros([n_probe,self.init_states])

        for qi in range(n_probe):

            hists_probs = self.success_probs[qi]
            hists_stds = self.success_stds[qi]
            ps_hists_probs = self.postselect_probs[qi]
            ps_hists_stds = self.postselect_stds[qi]
      
            for init_state in range(self.init_states):
                for L in self.seq_lengths:
                    dict_probs[L] = hists_probs[L][init_state] 
                    dict_stds[L] = hists_stds[L][init_state] 
                    ps_dict_probs[L] = ps_hists_probs[L][init_state]
                    ps_dict_stds[L] = ps_hists_stds[L][init_state]
                fid_init_state[qi, init_state], std_init_state[qi, init_state] = estimate_fidelity_std(dict_probs, dict_stds) 
                ps_fid_init_state[qi, init_state], ps_std_init_state[qi, init_state] = estimate_fidelity_std(ps_dict_probs, ps_dict_stds) 


        # Next, average and std over all qubits for fixed init_state.
        E01 = np.zeros(n_probe); E01_std = np.zeros(n_probe)
        E10 = np.zeros(n_probe); E10_std = np.zeros(n_probe)
        EL0 = np.zeros(n_probe); EL0_std = np.zeros(n_probe)
        EL1 = np.zeros(n_probe); EL1_std = np.zeros(n_probe)
        E_Leak = np.zeros(n_probe); E_Leak_std = np.zeros(n_probe)


        # error channels for individual qubits. 
        for qi in range(n_probe):
            E10[qi] = ps_fid_init_state[qi,0] - fid_init_state[qi,0]
            E01[qi] = ps_fid_init_state[qi,1] - fid_init_state[qi,1]
            if E10[qi] < 0: E10[qi] = 0 # enforce bounds on probabilities
            if E01[qi] < 0: E01[qi] = 0 # enforce bounds on probabilities
            EL0[qi] = 1.0 -  ps_fid_init_state[qi,0] 
            EL1[qi] = 1.0 -  ps_fid_init_state[qi,1]  
            E_Leak[qi] = (EL0[qi] + EL1[qi])/2 # average leakage rate.


            E10_std[qi] = np.sqrt( ps_std_init_state[qi,0]**2 + std_init_state[qi,0]**2 )
            E01_std[qi] = np.sqrt( ps_std_init_state[qi,1]**2 + std_init_state[qi,1]**2 ) 
            EL0_std[qi] = ps_std_init_state[qi,0] 
            EL1_std[qi] = ps_std_init_state[qi,1]  
            E_Leak_std[qi] =  np.sqrt( EL0_std[qi]**2 + EL1_std[qi]**2 ) / 2
            
        self.error_channels = [E10, E01, EL0, EL1, E_Leak]
        self.error_channels_stds = [E10_std, E01_std, EL0_std, EL1_std, E_Leak_std]
        self.avg_error_channels = [np.mean(error) for error in self.error_channels]

        self.avg_std_error_channels = []
        for error in self.error_channels_stds:
            avg_error_std = np.sqrt( sum([error[i]**2 for i in range(n_probe)]) ) / n_probe
            self.avg_std_error_channels.append(avg_error_std)


                
def marginalize_hists(n_qubits, probe_qubits, hists):
    """ return list of hists of same length as number of qubits """
    
    
    mar_hists = []
    for q in probe_qubits:# range(n_qubits):
        hists_q = {}
        for name in hists:
            L, init_state, exp_out = name[0], name[1], name[2][q]
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
            hists_q[(L, init_state, exp_out)] = mar_out
        mar_hists.append(hists_q)
    
    return mar_hists            


def get_success_probs(hists: dict, postselect = False):
    """ compute dictionary of surv probs for 1 qubit """
    
    # read in list of sequence lengths
    seq_len = list(set([sett[0] for sett in list(hists.keys())]))
    seq_len.sort()
    
    success_probs = {L:[] for L in seq_len}
    success_std = {L:[] for L in seq_len}

    postselect_success_probs = {L:[] for L in seq_len}
    postselect_success_std = {L:[] for L in seq_len}

    for sett in hists:
        L = sett[0]
        exp_out = sett[2] # the survival state for the qubit
        outcomes = hists[sett] # dictionary of results for each setting
        shots = sum(outcomes.values()) # sum up shot numbers for each setting
        
        prob = outcomes[exp_out]/shots # divide number of times we return the survival state
        p = outcomes[exp_out]/(shots + 2)
        std = float(np.sqrt(p*(1-p)/shots)) # bernoulli trial variance

        success_probs[L].append(prob)
        success_std[L].append(std)

        if postselect:
            if '2' in outcomes:
                postselect_prob = 1.0 - (outcomes['2']) / shots
                ps = (shots - outcomes['2']) / (shots + 2)
            else:
                postselect_prob = 1.0
                ps = (shots) / (shots + 2)
            postselect_std = float(np.sqrt(ps*(1-ps)/shots))

            postselect_success_probs[L].append(postselect_prob)
            postselect_success_std[L].append(postselect_std)
        
        
    if postselect:
        return success_probs, success_std, postselect_success_probs, postselect_success_std
    else:    
        return success_probs

def estimate_fidelity_std(avg_success_probs, std_success_probs):

    def fit_func(L, A, f):
        return A - L * f 

    x = [L for L in avg_success_probs]
    x.sort()

    y = [avg_success_probs[L] for L in x]
    yerr = [std_success_probs[L] for L in x]

    # perform best fit
    popt, pcov = curve_fit(fit_func, x, y, p0=[1.0, 0.1], bounds=([0,0], [1,1]), sigma=yerr, absolute_sigma = True)

    avg_fidelity = 1 - popt[1]
    fid_std = np.sqrt(pcov[1][1])

    return avg_fidelity, fid_std


    
def estimate_fidelity(avg_success_probs):
    
    def fit_func(L, A, f):
        return A - L * f 
    
    x = [L for L in avg_success_probs]
    x.sort()
        
    y = [avg_success_probs[L] for L in x]

    # perform best fit
    popt, _ = curve_fit(fit_func, x, y) #
    
    avg_fidelity = 1 - popt[1]
    
    
    return avg_fidelity











