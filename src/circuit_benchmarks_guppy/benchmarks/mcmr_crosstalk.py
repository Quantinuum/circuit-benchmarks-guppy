# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22025

Mid-circuit measurement and reset (MCMR) crosstalk benchmarking for Helios-1

@author: Victor Colusi
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
# from guppylang.std.qsystem.functional import measure, reset

from hugr.package import FuncDefnPointer

from circuit_benchmarks_guppy.benchmarks.experiment import Experiment
from circuit_benchmarks_guppy.tools import bootstrap as bs
from circuit_benchmarks_guppy.tools.leakage_measurement import measure_and_record_leakage
from circuit_benchmarks_guppy.tools.analysis import postselect_leakage, get_postselection_rates
# from qtm_platform.ops import order_in_zones, sleep ##V:  This needs to be updated once we get the documentation for this functionality ###



class MCMR_Crosstalk_Experiment(Experiment):
    
    def __init__(self, focus_qubits, seq_lengths, **kwargs):
        super().__init__(**kwargs)
        self.focus_qubits = focus_qubits #focus_qubits
        self.n_qubits = 98 # 98 total qubits on Helios-1
        self.n_zone_qubits = 16 # total qubits in the gatezones.  max that can be used in order function.
        self.n_ring_qubits = self.n_qubits - self.n_zone_qubits # there are 82 qubits stored in the ring for Helios-1.
        self.probe_qubits = self.get_probe_qubits() 
        self.init_states = 6 # six different cardinal directions on Bloch sphere
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
        elif init_state==2: # |+x>
            gate_index = 1 # H
        elif init_state==3: # |-x>
            gate_index = 9 # HX
        elif init_state==4: # |+y>
            gate_index = 12 # SH
        elif init_state==5: # |-y>
            gate_index = 15 # Sdg H
        return gate_index

        
    
    def make_circuit(self, setting: tuple) -> FuncDefnPointer:

        seq_len = setting[0] # number of MCMR's to perform on each target qubit
        init_state = setting[1] # for each init_state we perform a different circuit

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

            order_in_zones(zone_qubits) # order zone qubits into gatezones

            for q_i in range(comptime(n_zone_qubits)): # put all zone qubits in a Pauli eigenstate
                if comptime(gate_index) > 0:
                    apply_SQ_Clifford(zone_qubits[q_i], comptime(gate_index))

               
            for i in range(comptime(seq_len)):
                for j in comptime(focus_qubits): # only MCMR
                    measure_and_reset(zone_qubits[j])
                order_in_zones(zone_qubits)
            

            for q_i in range(comptime(n_zone_qubits)): # put all zone qubits back into |0> state
                if comptime(gate_index) > 0:
                    apply_SQ_Clifford_inv(zone_qubits[q_i], comptime(gate_index))
            
            order_in_zones(zone_qubits)

            # measure
            measure_and_record_leakage(zone_qubits, True)

            for q_i in range(comptime(n_ring_qubits)): # put all qubits back into |0> state
                if comptime(gate_index) > 0:
                    apply_SQ_Clifford_inv(ring_qubits[q_i], comptime(gate_index))
    
            measure_and_record_leakage(ring_qubits, True)
    
        # return the compiled program (HUGR)
        return main.compile()


    # Currently the postselected results are not bootstrapped.  The std calculation is not totally complete here...
    def analyze_results(self, error_bars=True, plot=True, display=True, **kwargs):
        
        self.marginal_results = marginalize_hists(self.n_qubits, self.probe_qubits, self.results)

        # get rates of postselection (# of non-leaked shots over the total number of shots)
        self.postselection_marginal_results = [postselect_leakage(mar_re) for mar_re in self.marginal_results]
        self.postselection_rates = []
        self.postselection_rates_stds = []
        for mar_re in self.marginal_results: 
            ps_rates, ps_stds = get_postselection_rates(mar_re, self.setting_labels)
            self.postselection_rates.append(ps_rates) #it's only the size of the probe qubits
            self.postselection_rates_stds.append(ps_stds) 


        # get the non-postselected results
        self.success_probs = []
        self.avg_success_probs = []
        self.postselect_probs = []
        self.avg_postselect_probs = []
        for j, hists in enumerate(self.marginal_results): # loops over each qubit. 
             # VEC:  Replace this with a single function
            succ_probs_j, postselect_probs_j = get_success_probs(hists, postselect = True)
            avg_succ_probs_j = get_avg_success_probs(succ_probs_j)
            avg_postselect_probs_j = get_avg_success_probs(postselect_probs_j) # checked that this is the same as Karl's functions
            self.success_probs.append(succ_probs_j)
            self.avg_success_probs.append(avg_succ_probs_j)
            self.postselect_probs.append(postselect_probs_j)
            self.avg_postselect_probs.append(avg_postselect_probs_j)


        # to construct the correct fidelity, we need both postselected and total results.  
        self.fid_avg = [estimate_fidelity(avg_succ_probs) for avg_succ_probs in self.avg_success_probs] # all channels
        self.mean_fid_avg = float(np.mean(self.fid_avg)) # averaged over all the qubits

        self.postselect_fid_avg = [estimate_fidelity(avg_succ_probs) for avg_succ_probs in self.avg_postselect_probs] # all channels
        self.postselect_mean_fid_avg = float(np.mean(self.postselect_fid_avg)) # averaged over all the qubits
        # 

        # compute error bars
        # we need equivalent for postselected results
        if error_bars == True:
            self.error_data = [compute_error_bars(hists) for hists in self.marginal_results]
            self.fid_avg_std = [data['avg_fid_std'] for data in self.error_data]
            self.mean_fid_avg_std = float(np.sqrt(sum([s**2 for s in self.fid_avg_std]))/len(self.fid_avg_std))
            
            
        
        self.plot_results(error_bars=error_bars, **kwargs)
        self.display_results(error_bars=error_bars, **kwargs) # this gives the correct infidelities resolved into comp and leak channels.
            
        self.plot_postselection_rates(display=display, **kwargs)
        
        self.plot_error_channels(**kwargs)
            
            
        # if display == True:


    def plot_results(self, error_bars=True, **kwargs):
        ylim = kwargs.get('ylim', None)
        title = kwargs.get('title', f'{self.protocol} \n Decays')
        probs_array = self.avg_success_probs
        
  
        
        def fit_func(L, A, f):
            return A - L * f
        
        
        # Create a colormap
        cmap = cm.turbo

        # Normalize color range from 0 to num_lines-1
        cnorm = mcolors.Normalize(vmin=0, vmax=self.n_qubits-1)
        
        
        x = self.seq_lengths
        xfit = np.linspace(x[0], x[-1], 100)

        fig = plt.figure(figsize=(12, 8))

        for j, avg_succ_probs in enumerate(probs_array):
            
            ind_probe = self.probe_qubits[j]
            co = cmap(cnorm(j))
        
            y = [avg_succ_probs[L] for L in x]
            if error_bars == False:
                yerr = None
            else:
                yerr = [self.error_data[j]['success_probs_stds'][L] for L in x]
        
            # perform best fit
            popt, _ = curve_fit(fit_func, x, y)
            yfit = fit_func(xfit, *popt)
            plt.errorbar(x, y, yerr=yerr, fmt='*', color=co, label=f'q{ind_probe}')
            plt.plot(xfit, yfit, '-', color=co)
        
        plt.title(title)
        plt.ylabel('Success Probability')
        plt.xlabel('Sequence Length')
        plt.xticks(ticks=x, labels=x)
        plt.ylim(ylim)
        plt.legend()
        plt.show()

    def plot_postselection_rates(self, display=True, **kwargs):
        
        ylim = kwargs.get('ylim2', None)
        title = kwargs.get('title2', f'{self.protocol} \n Leakage Postselection Rates')
        
        def fit_func(L, A, f):
            return A - L * f
        
        # Create a colormap
        cmap = cm.turbo

        # Normalize color range from 0 to num_lines-1
        cnorm = mcolors.Normalize(vmin=0, vmax=self.n_qubits-1)
        
        x = self.seq_lengths
        xfit = np.linspace(x[0], x[-1], 100)
        leakage_rates = []
        leakage_stds = []

        fig = plt.figure(figsize=(12, 8))
        
        for j, ps_rates in enumerate(self.postselection_rates):

            ind_probe = self.probe_qubits[j]

            y = [ps_rates[L] for L in x]
            yerr = [self.postselection_rates_stds[j][L] for L in x]
        
            # perform best fit
            popt, pcov = curve_fit(fit_func, x, y, p0=[0.4, 0.9], bounds=([0,0], [1,1]), sigma=yerr, absolute_sigma = True)
            leakage_rates.append(1-popt[1])
            leakage_stds.append(float(np.sqrt(pcov[1][1])))
            yfit = fit_func(xfit, *popt)
            plt.errorbar(x, y, yerr=yerr, fmt='o', color=cmap(cnorm(j)), label=f'q{ind_probe}')
            plt.plot(xfit, yfit, '-', color=cmap(cnorm(j)))
        
        plt.title(title)
        plt.ylabel('Postselection Rate')
        plt.xlabel('Sequence Length')
        plt.xticks(ticks=x, labels=x)
        plt.ylim(ylim)
        if self.n_qubits <= 16:
            plt.legend()
        plt.show()
        
        self.leakage_rates = leakage_rates
        self.leakage_rates_stds = leakage_stds
        self.mean_leakage_rate = float(np.mean(leakage_rates))
        self.mean_leakage_std = float(np.sqrt(sum([s**2 for s in leakage_stds]))/len(leakage_stds))
        
        prec = kwargs.get('precision', 6)
        verbose = kwargs.get('verbose', True)

        if display:
            leak_rate = self.mean_leakage_rate
            leak_std = self.mean_leakage_std
            if verbose:
                print('Average leakage rates\n' + '-'*30)
                for i, rate in enumerate(leakage_rates):
                    q = self.probe_qubits[i]
                    message = f'qubit {q}: {round(1-rate, prec)}'
                    message += f' +/- {round(leakage_stds[i], prec)}'
                    print(message)
            print('-'*30)
            print(f'Qubit average leakage rate: {round(1 - leak_rate, 6)} +/- {round(leak_std, 6)}')
            print( f'\n\n\n')

    def plot_error_channels(self, **kwargs):
        
        self.estimate_error_channels() # calculate the error channels
        
        barWidth = 0.25
        fig = plt.subplots(figsize =(12, 8)) 

        labels = ['Bitflip 0->1','Bitflip 1->0','Leakage {0,1}->L','Dephasing']
        br_1 = self.probe_qubits
        br_2 = [x + barWidth for x in self.probe_qubits]
        br_3 = [x + 2*barWidth for x in self.probe_qubits]
        br_4 = [x + 3*barWidth for x in self.probe_qubits]
        br_h1 = self.error_channels[0]
        br_h2 = self.error_channels[1]
        br_h3 = self.error_channels[2]
        br_h4 = self.error_channels[3]

        plt.bar(br_1, br_h1, color ='orange', width = barWidth, 
                edgecolor ='none', label =labels[0]) 
        plt.bar(br_2, br_h2, color ='blue', width = barWidth, 
                edgecolor ='none', label =labels[1])
        plt.bar(br_3, br_h3, color ='red', width = barWidth, 
                edgecolor ='none', label =labels[2])
        plt.bar(br_4, br_h4, color ='green', width = barWidth, 
                edgecolor ='none', label = labels[3])
        plt.title(f'MCMR Crosstalk \n focus qubits = q{self.focus_qubits}, machine = Helios-1')


        plt.xlabel('Probe Qubit', fontsize = 12) 
        plt.ylabel('Per MCMR Probability', fontsize = 12) 
        plt.xticks(range(self.n_qubits))

        plt.legend()
        plt.yscale('log')
        plt.show()

        prec = kwargs.get('precision', 6)
        state_fidelity_check = (self.avg_error_channels[0] + self.avg_error_channels[1] + 4*self.avg_error_channels[2] + 4 * self.avg_error_channels[3])/6
        process_fidelity_check = (self.avg_error_channels[0] + self.avg_error_channels[1] + 2*self.avg_error_channels[2] + 4 * self.avg_error_channels[3])/4

        print('Average error rates\n' + '-'*50)
        for channel in range(4):
            message = f'{labels[channel]}: '
            message += f'{round(self.avg_error_channels[channel], prec)}'
            message += f' +/- {round(self.std_error_channels[channel], prec)}'
            print(message)
        print('-'*50)
        print( f'\n')                         
        print('Error channel estimates of fidelities\n' + '-'*50)
        print(f'Average state infidelity:  {round(state_fidelity_check,prec)}')
        print(f'Average process infidelity:  {round(process_fidelity_check, prec)}' )
        print('-'*50)
        print( f'\n\n\n')                         



    def display_results(self, error_bars=True, **kwargs):
        
        prec = kwargs.get('precision', 6)
        verbose = kwargs.get('verbose', True)

        self.estimate_leakage_errors() # calculate the process infidelity

        avg_process_infidelity = self.avg_leakage_errors[2]
        std_process_infidelity = self.std_leakage_errors[2]

        if verbose:
            print('Average State Infidelities\n' + '-'*30)
            for i, f_avg in enumerate(self.fid_avg):
                q = self.probe_qubits[i]
                message = f'qubit {q}: {round(1-f_avg, prec)}'
                if error_bars == True:
                    f_std = self.error_data[i]['avg_fid_std']
                    message += f' +/- {round(f_std, prec)}'
                print(message)
        avg_message = 'Qubit Average State: '
        mean_infid = 1-self.mean_fid_avg
        avg_message += f'{round(mean_infid,prec)}'
        if error_bars == True:
            mean_fid_avg_std = self.mean_fid_avg_std
            avg_message += f' +/- {round(mean_fid_avg_std, prec)}'
        print('-'*30)
        print(avg_message)
        avg_message = 'Qubit Average Process (no bootstrapping): '
        avg_message += f'{round(avg_process_infidelity, prec)}'
        avg_message += f' +/- {round(std_process_infidelity, prec)}'
        print(avg_message)



    def estimate_leakage_errors(self, **kwargs):

        n_probe = len(self.probe_qubits) # number of probe qubits
         
        ### Fidelities. ###  
        # Averaged over all init_states.
        computational_population = self.postselect_fid_avg # t

        leakage_rate = np.zeros(n_probe)
        depolarizing_parameter = np.zeros(n_probe)
        process_fid = np.zeros(n_probe)
        state_fid = np.zeros(n_probe)
        for i in range(n_probe):
            leakage_rate[i] = 1.0 - self.postselect_fid_avg[i] # \tau = 1 - t
            depolarizing_parameter[i] =  2 * self.fid_avg[i] - computational_population[i]# r 
            process_fid[i] = ( computational_population[i] + 3 * depolarizing_parameter[i] ) / 4
            state_fid[i] = ( computational_population[i] + depolarizing_parameter[i] ) / 2
        
        process_infid = 1 - process_fid
        state_infid = 1 - state_fid
        # computational_error = computational_population - depolarizing_parameter 

        self.leakage_errors = [leakage_rate, 1 - depolarizing_parameter, process_infid, state_infid]
        self.avg_leakage_errors = [np.mean(error) for error in self.leakage_errors]
        self.std_leakage_errors = [np.std(error)/np.sqrt(n_probe) for error in self.leakage_errors] # we report the standard error in the mean

    def estimate_error_channels(self, **kwargs):
        '''Input:  marginalized histogram for an single qubit.
        Output: process fidelity, state fidelity, depolarizing parameter, computational population 
        ''' 

        n_probe = len(self.probe_qubits) # number of probe qubits

        ### Error channels ###
        # First, output per MCMR success probability for individual circuits
        dict_foo = {L:[] for L in self.seq_lengths}

        fid_init_state = np.zeros([n_probe,self.init_states]) # "F_i" for each qubit (qubit, init_state)
        for qi, hists in enumerate(self.success_probs):
      
            for init_state in range(self.init_states):
                for L in self.seq_lengths:
                    dict_foo[L] = hists[L][init_state] #jth qubit, L depth, ith init state.  ## Checked
                fid_init_state[qi, init_state] = estimate_fidelity(dict_foo) # all channels ## Checked
        
        # VEC:  duplicated code. write a helper function below...
        ps_fid_init_state = np.zeros([n_probe,self.init_states]) # Postselection rate for each qubit (qubit, init_state)
        for qi, hists in enumerate(self.postselect_probs):
      
            for init_state in range(self.init_states):
                for L in self.seq_lengths:
                    dict_foo[L] = hists[L][init_state] #jth qubit, L depth, ith init state.  ## Checked
                ps_fid_init_state[qi, init_state] = estimate_fidelity(dict_foo) # all channels ## Checked

        # Next, average and std over all qubits for fixed init_state.  VEC:  Duplicated code, write helper function
        E01 = np.zeros(n_probe)
        E10 = np.zeros(n_probe)
        EL0 = np.zeros(n_probe)
        EL1 = np.zeros(n_probe)
        E_Leak = np.zeros(n_probe)
        pZ = np.zeros(n_probe)

        # print(ps_fid_init_state)
        # print(fid_init_state)

        # error channels for individual qubits
        for qi in range(n_probe):
            E10[qi] = ps_fid_init_state[qi,0] - fid_init_state[qi,0]
            E01[qi] = ps_fid_init_state[qi,1] - fid_init_state[qi,1]
            E_Leak[qi] = sum(1.0 -  ps_fid_init_state[qi,:] )/6 # Checked against self.mean_leakage_rate
            pZ[qi] = ( sum( 1 - fid_init_state[qi, 2:self.init_states]) - 2 * E_Leak[qi] ) / 4
            
        self.error_channels = [E01,E10,E_Leak,pZ]
        self.avg_error_channels = [np.mean(error) for error in self.error_channels]
        self.std_error_channels = [np.std(error)/np.sqrt(n_probe) for error in self.error_channels] # we report the standard error in the mean.

        #
        # for completeness 
        for qi in range(n_probe):
            EL0[qi] = 1.0 -  ps_fid_init_state[qi,0] 
            EL1[qi] = 1.0 -  ps_fid_init_state[qi,1]  
        self.EL0 = EL0
        self.EL1 = EL1
        self.EL0_avg = np.mean(EL0)
        self.EL1_avg = np.mean(EL1)

        
  

        
        



    

# def get_error_channels(self):

                
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
    postselect_success_probs = {L:[] for L in seq_len}

    for sett in hists:
        L = sett[0]
        exp_out = sett[2] # the survival state for the qubit
        outcomes = hists[sett] # dictionary of results for each setting
        shots = sum(outcomes.values()) # sum up shot numbers for each setting
        
        if exp_out in outcomes:
            prob = outcomes[exp_out]/shots # divide number of times we return the survival state
        else:
            prob = 0.0

        if postselect:
            if '2' in outcomes:
                postselect_prob = 1.0 - (outcomes['2']) / shots
            else:
                postselect_prob = 1.0
            postselect_success_probs[L].append(postselect_prob)
        success_probs[L].append(prob)
        
    if postselect:
        return success_probs, postselect_success_probs
    else:    
        return success_probs


def get_avg_success_probs(success_probs: dict):
    
    avg_success_probs = {}
    for L in success_probs:
        avg_success_probs[L] = float(np.mean(success_probs[L]))
    
    return avg_success_probs
        
    
def estimate_fidelity(avg_success_probs):
    
    def fit_func(L, A, f):
        return A - L * f 
    
    x = [L for L in avg_success_probs]
    x.sort()
        
    y = [avg_success_probs[L] for L in x]

    # perform best fit
    popt, _ = curve_fit(fit_func, x, y) #, p0=[0.4, 0.9], bounds=([0,0], [0.5,1]))
    
    avg_fidelity = 1 - popt[1]
    
    
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
        probs_stds[L] = float(np.std([b_p[L] for b_p in boot_avg_succ_probs]))
    
    avg_fid_std = float(np.std([f for f in boot_avg_fids]))
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
            init_states = len(circ_list)
            resamp_circs = np.random.choice(init_states, size=init_states)
            for init_state, init_state2 in enumerate(resamp_circs):
                circ = circ_list[init_state2]
                name_resamp = (L, init_state, circ[2])
                outcomes = hists[circ]
                hists_resamp[name_resamp] = outcomes
        
        # do parametric resample
        boot_hists.append(bs.resample_hists(hists_resamp))
    
    return boot_hists










