# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:01:55 2022

RB for arbitrary angle TQ gates

@author: Karl Mayer
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import expm
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from pytket.circuit import Unitary2qBox, OpType, Circuit
from pytket.passes import DecomposeBoxes, AutoRebase

from guppylang import guppy
from guppylang.std.builtins import array, result, owned
from guppylang.std.quantum import qubit, measure_array
from guppylang.std.qsystem import measure_leaked
from qtm_platform.ops import order_in_zones

import bootstrap as bs
from experiment import Experiment
import analysis_tools as at


# rebase required for Nexus submission
rebase = AutoRebase({OpType.ZZPhase, OpType.PhasedX, OpType.Rz})


n = guppy.nat_var("n")


# define Paulis
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
ZZ = np.kron(Z,Z)


class ArbRB_Experiment(Experiment):
    
    def __init__(self, qubits, seq_len, seq_reps, ZZ_angle=[1/2], **kwargs):
        super().__init__()
        self.protocol = 'TQ arbZZ rb'
        self.parameters = {'qubits':qubits,
                           'seq_len':seq_len,
                           'seq_reps':seq_reps}
        self.qubits = qubits
        self.seq_len = seq_len
        self.seq_reps = seq_reps
        self.ZZ_angle = ZZ_angle
        self.machine = kwargs.get('machine', None)
        self.filename = kwargs.get('filename', None)

        
    # add circuits
    def add_circuits(self):        
        self.surv_state = {}
        for angle in self.ZZ_angle:
            for L in self.seq_len:
                for s in range(self.seq_reps):
                    exp_out = np.random.choice(['00', '01', '10', '11'])
                    # circ = self.make_circuit(L, angle, exp_out)
                    # rebase.apply(circ)
                    setting = (angle, L, s, exp_out)
                    self.add_setting(setting)
                    self.surv_state[setting] = exp_out
                    # self.add_circuit(circ)
        
        
    # circuit building method
    def make_circuit(self, sett):
        
        angle, L, s, exp_out = sett

        # convert to list of qubit pairs if only one qubit pair
        qubits = self.qubits
        if type(qubits[0]) == int:
            qubits = [qubits]
            
        # number of qubits needed
        n = max([max(q_pair) for q_pair in qubits]) + 1

        # initialize unitary
        U = np.identity(4)
        
        # initialize circuit
        # circ = oc.init_circuit_with_qubit_order(qubits, self.machine)
        circ = Circuit(n, n)
        # circuit sequence
        for i in range(L):
            
            # SQ gates
            SU2s = rand_SU2s(2)
            
            # update U
            th0, phi0, angle0 = SU2s[0][0], SU2s[0][1], SU2s[0][2]
            th1, phi1, angle1 = SU2s[1][0], SU2s[1][1], SU2s[1][2]
            U0 = expm(-1j*(np.sin(th0)*np.cos(phi0)*X +
                           np.sin(th0)*np.sin(phi0)*Y +
                           np.cos(th0)*Z)*angle0/2)
            U1 = expm(-1j*(np.sin(th1)*np.cos(phi1)*X +
                           np.sin(th1)*np.sin(phi1)*Y +
                           np.cos(th1)*Z)*angle1/2)
            U = np.kron(U1, U0) @ U
            
            # appy SQ gates
            for q_pair in qubits:
                q0, q1 = q_pair[0], q_pair[1]
                SU2_pytket(circ, SU2s[0], q0)
                SU2_pytket(circ, SU2s[1], q1)
            
            # TQ gates
            for q_pair in qubits:
                q0, q1 = q_pair[0], q_pair[1]
                circ.ZZPhase(angle, q0, q1)
                
            # barrier
            circ.add_barrier(list(range(n)))
            
            # update U
            U = expm(-1j*ZZ*np.pi*angle/2) @ U
            
        # apply inverse SU(4)
        Udg = np.conj(U.T)
        U_inv_box = Unitary2qBox(Udg)
        for q_pair in qubits:
            q0, q1 = q_pair[0], q_pair[1]
            circ.add_unitary2qbox(U_inv_box, q1, q0)
        
        circ.add_barrier(list(range(n)))

        # apply final random Pauli
        for q_pair in qubits:
            for q in [0,1]:
                if exp_out[1-q] == '1':
                    circ.X(q_pair[q])
        
        circ.add_barrier(list(range(n)))
        # final order command
        # oc.add_order_command(circ, machine=self.machine)
        
        # # measure
        # circ.add_barrier(range(n))
        # c_bit = 0
        # for q_pair in qubits:
        #     for q in q_pair:
        #         circ.Measure(q,c_bit)
        #         c_bit += 1
        
        DecomposeBoxes().apply(circ)
        rebase.apply(circ)

        return pytket_to_guppy(circ)
        
        
    def analyze_results(self, error_bars=True, plot=True, display=True, **kwargs):
        
        #fit_method = kwargs.get('fit_method', 1)
        #detect_leakage = kwargs.get('detect_leakage', self.options['detect_leakage'])
        import circ_benchmarks.tools.analysis_tools as at
        results = {}
        for sett in self.results:
            angle = sett[0]
            outcomes = self.results[sett]
            if angle not in results:
                results[angle] = {sett:outcomes}
            elif angle in results:
                results[angle][sett] = outcomes
                
        
        #if detect_leakage == False:
        self.marginal_results = {
            angle: [
                postselect_leakage(marg) 
                for marg in at.marginalize_hists(self.qubits, results[angle])
            ] 
            for angle in results
        }
        unpostselected_marginal_results = {
            angle: [
                marg
                for marg in at.marginalize_hists(self.qubits, results[angle])
            ] 
            for angle in results
        }
        # #elif detect_leakage == True:
        # if type(self.qubits[0]) == int:
        #     qubits = [self.qubits]
        # else:
        #     qubits = self.qubits
        # # self.marginal_results = [{setting: results[setting][i] for setting in results} for i in range(len(qubits))]
        # self.postselection_rate = results_to_ps_rate(results, self.seq_len, self.shots)
        # self.avg_postselection_rate = avg_ps_rate(self.postselection_rate)
        self.postselection_rates = {angle: [] for angle in results}
        self.postselection_rates_stds = {angle: [] for angle in results}
        self.leakage_rates = {}
        self.leakage_rates_stds = {}
        self.mean_leakage_rate = {}
        self.mean_leakage_std = {}
        for angle in results:
            for mar_re in unpostselected_marginal_results[angle]:
                ps_rates, ps_stds = get_postselection_rates(mar_re)
                self.postselection_rates[angle].append(ps_rates)
                self.postselection_rates_stds[angle].append(ps_stds)
            leakage_rates, leakage_stds = estimate_leakage_rates(self.postselection_rates[angle],
                                                                    self.postselection_rates_stds[angle],
                                                                    self.seq_len)
            self.leakage_rates[angle] = leakage_rates
            self.leakage_rates_stds[angle] = leakage_stds
            self.mean_leakage_rate[angle] = float(np.mean(leakage_rates))
            self.mean_leakage_std[angle] = float(np.sqrt(sum([s**2 for s in leakage_stds]))/len(leakage_stds))
        
        
        self.success_probs = {angle:[at.get_success_probs(hists)
                              for hists in self.marginal_results[angle]]
                              for angle in self.marginal_results}
        self.avg_success_probs = {angle:[at.get_avg_success_probs(hists)
                              for hists in self.marginal_results[angle]]
                                  for angle in self.marginal_results}
        self.fid_avg = {angle:[at.estimate_fidelity(avg_succ_probs) for avg_succ_probs
                             in self.avg_success_probs[angle]]
                        for angle in self.avg_success_probs}
        self.mean_fid_avg = {angle:np.mean(self.fid_avg[angle]) + self.mean_leakage_rate[angle] for angle in self.fid_avg} 
        
        # compute error bars
        if error_bars == True:
            self.error_data = {angle:[compute_error_bars(hists)
                               for hists in self.marginal_results[angle]]
                                    for angle in self.marginal_results}
            self.fid_avg_std = {angle:[self.error_data[angle][i]['avg_fid_std'] for i in range(len(self.error_data[angle]))] # + self.leakage_rates_stds[angle][i]
                                for angle in self.error_data}
            self.mean_fid_avg_std = {angle:np.sqrt(sum([s**2 for s in self.fid_avg_std[angle]]))/len(self.fid_avg_std[angle])
                                     for angle in self.fid_avg_std}
        
        # make plots
        if plot == True:
            self.plot_results(error_bars=error_bars, **kwargs)
            self.plot_postselection_rates()
            
        # display results
        if display == True:
            self.display_results(error_bars=error_bars)
            
        # estimate leakage errors
        #if detect_leakage == True:
            #self.estimate_leakage_rate(plot=plot, display=display)
    
    
    def plot_results(self, error_bars=True, **kwargs):
        
        seq_len = self.seq_len
        protocol = self.protocol
        labels = self.qubits
        
        for angle in self.avg_success_probs:
            
            title = kwargs.get('title', protocol + f' theta = {angle}')
            
            avg_success_probs = self.avg_success_probs[angle]
            
            if error_bars:
                avg_success_stds = [{L:error_data['success_probs_stds'][L] for L in seq_len if L in error_data['success_probs_stds']}
                                    for error_data in self.error_data[angle]]
            else:
                avg_success_stds = None
            
            at.plot_TQ_decays(seq_len, avg_success_probs, avg_success_stds,
                              title=title, labels=labels,
                              **kwargs)
            
        
        # plot error versus angle
        if len(self.ZZ_angle) > 1:
            if error_bars == True:
                plot_error_versus_angle(self.fid_avg, self.fid_avg_std)
            elif error_bars == False:
                plot_error_versus_angle(self.fid_avg, None)

    def plot_postselection_rates(self, display=True, **kwargs):
        
        for angle in self.leakage_rates:
            ylim = kwargs.get('ylim2', None)
            title = kwargs.get('title2', f'{self.protocol} Leakage Postselection Rates f = {angle}')
            
            # define fit function
            def fit_func(L, a, f):
                return a*f**L
            
            # Create a colormap
            cmap = cm.turbo

            # Normalize color range from 0 to num_lines-1
            cnorm = mcolors.Normalize(vmin=0, vmax=len(self.qubits)-1)
            
            x = self.seq_len
            xfit = np.linspace(x[0], x[-1], 100)
            
            for j, ps_rates in enumerate(self.postselection_rates[angle]):
            
                y = [ps_rates[L] for L in x]
                yerr = [self.postselection_rates_stds[angle][j][L] for L in x]
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

        # # plot error versus angle
        # if len(self.ZZ_angle) > 1:
        #     plot_leakage_versus_angle(self.fid_avg, self.fid_avg_std)

            
    
    def display_results(self, error_bars=True):
        
        for angle in self.fid_avg:
            print(f'\nTheta = {round(angle,4)} Average Fidelities\n' + '-'*34)
            for j, f_avg in enumerate(self.fid_avg[angle]):
                q_pair = self.qubits[j]
                message = f'qubits {q_pair}: {round(f_avg, 5)}'
                if error_bars == True:
                    f_std = self.error_data[angle][j]['avg_fid_std']
                    message += f' +/- {round(f_std, 5)}'
                print(message)
            avg_message = '-'*34 + '\nZone average:  '
            mean_fid_avg = self.mean_fid_avg[angle]
            avg_message += f'{round(mean_fid_avg,5)}'
            if error_bars == True:
                mean_fid_avg_std = self.mean_fid_avg_std[angle]
                avg_message += f' +/- {round(mean_fid_avg_std, 5)}'
            print(avg_message)
        
                    
                    
### Analysis functions


def compute_error_bars(hists):
    
    
    boot_hists = bootstrap(hists)
    boot_avg_succ_probs = [at.get_avg_success_probs(b_h) for b_h in boot_hists]
    boot_avg_fids = [at.estimate_fidelity(avg_succ_prob)
                     for avg_succ_prob in boot_avg_succ_probs]
    
    
    # read in seq_len and list of Paulis
    seq_len = list(boot_avg_succ_probs[1].keys())
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
            for s, s2 in enumerate(resamp_circs):
                circ = circ_list[s2]
                name_resamp = (L, s, circ[-1])
                outcomes = hists[circ]
                hists_resamp[name_resamp] = outcomes
        
        # do parametric resample
        boot_hists.append(bs.resample_hists(hists_resamp))
    
    return boot_hists


def plot_error_versus_angle(fid_avg, fid_avg_std=None, display=True, savefig=False):
    
    x = list(fid_avg.keys())
    x.sort()
    
    y = [1-np.mean(fid_avg[angle]) for angle in x]
    if fid_avg_std:
        yerr = [np.sqrt(sum([s**2 for s in fid_avg_std[angle]]))/len(fid_avg_std) for angle in x]
    else:
        yerr=None
    
    def fit_func(x, a, b):
        return a*x+b
    
    popt, pcov = curve_fit(fit_func, x, y)
    perr = np.sqrt(np.diag(pcov))
    xfit = np.linspace(x[0],x[-1],100)
    yfit = fit_func(xfit, *popt)

    plt.errorbar(x, y, yerr=yerr, fmt='o')
    plt.plot(xfit, yfit, '--', color='k')
    plt.xlabel('ZZ_angle (units of pi)')
    plt.ylabel('Average infidelity')
    plt.title('Arb-angle TQ benchmarking')
    plt.xticks(ticks=x, labels=x)
    
    if savefig == True:
        plt.savefig('arbrb_error_vs_angle.svg', format='svg')
    plt.show()
    
    if display == True:
        print('fit params (linear model): a*x+b\n')
        print(f'a =  {round(popt[0],5)} +/- {round(perr[0],5)}')
        print(f'b =  {round(popt[1],5)} +/- {round(perr[1],5)}')
        
        

def plot_leakage_versus_angle(exp_list, display=True, save=False,
                            **kwargs):
    
    
    title = kwargs.get('title', None)
    xlim = kwargs.get('xlim', None)
    ylim = kwargs.get('ylim', None)
    
    x = [exp.options['ZZ_angle'] for exp in exp_list]
    y = [np.mean(exp.leakage_rate) for exp in exp_list]
    yerr = [np.sqrt(sum([s**2 for s in exp.leakage_rate_std]))/len(exp.leakage_rate_std) for exp in exp_list]

    def fit_func(x, a, b):
        return a*x+b
    
    popt, pcov = curve_fit(fit_func, x, y)
    perr = np.sqrt(np.diag(pcov))
    xfit = np.linspace(x[0],x[-1],100)
    yfit = fit_func(xfit, *popt)

    plt.errorbar(x, y, yerr=yerr, fmt='o')
    plt.plot(xfit, yfit, '--', color='k')
    plt.xlabel('ZZ_angle (units of pi)')
    plt.ylabel('Average Leakage Rate')
    plt.title('arb angle TQ leakage')
    plt.xticks(ticks=x, labels=x)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)
    if save == True:
        plt.savefig('arbrb_leakage_vs_angle.svg', format='svg')
    plt.show()
    
    if display == True:
        print('fit params (linear model): a*x+b\n')
        print(f'a =  {round(popt[0],5)} +/- {round(perr[0],5)}')
        print(f'b =  {round(popt[1],5)} +/- {round(perr[1],5)}')

    
# compute postselection rates

def results_to_ps_rate(results, seq_len, shots):
    
    postselection_rate = {L:[] for L in seq_len}
    for setting in results:
        L = setting[0]
        rates = []
        for outcomes in results[setting]:
            ps_shots = sum(outcomes.values())
            rates.append(ps_shots/shots)
        postselection_rate[L].append(rates)
        
    return postselection_rate


def avg_ps_rate(ps_rate):
    
    avg_postselection_rate = {L:[] for L in ps_rate}
    for L in ps_rate:
        for i in range(len(ps_rate[L][0])):
            avg_rate_i = np.mean([rates[i] for rates in ps_rate[L]])
            avg_postselection_rate[L].append(avg_rate_i)
        
    return avg_postselection_rate


def estimate_leakage_rates(post_rates, post_stds, seq_lengths):
    
    leakage_rates = []
    leakage_stds = []
    
    # define fit function
    def fit_func(L, a, f):
        return a*f**L
    
    for j, ps_rates in enumerate(post_rates):
        
        y = [ps_rates[L] for L in seq_lengths if L in ps_rates]
        yerr = [post_stds[j][L] for L in seq_lengths if L in post_stds[j]]
        x = [L for L in seq_lengths if L in ps_rates]
        # perform best fit
        popt, pcov = curve_fit(fit_func, x, y, p0=[0.9, 0.9], bounds=([0,0], [1,1]), sigma=yerr)
        leakage_rates.append((1-popt[1]))
        leakage_stds.append(float(2*np.sqrt(pcov[1][1]))/3)
    
    return leakage_rates, leakage_stds


def postselect_leakage(results: dict) -> dict:
    """ returns results dict containing no leakage '2' outcomes """
    
    ps_results = {}
    for sett in results:
        outcomes = results[sett]
        ps_outcomes = {}
        for b_str in outcomes:
            if '2' not in b_str:
                ps_outcomes[b_str] = outcomes[b_str]
                
        if len(ps_outcomes) > 0:
            ps_results[sett] = ps_outcomes
    
    return ps_results


def get_postselection_rates(results: dict) -> dict:
    """ returns dictionary of postselection rates for each sequence length """
    
    total_shots = {}
    ps_shots = {}
    for sett in results:
        L = sett[1]
        if L not in total_shots:
            total_shots[L] = 0
        if L not in ps_shots:
            ps_shots[L] = 0
            
        outcomes = results[sett]
        for b_str in outcomes:
            counts = outcomes[b_str]
            total_shots[L] += counts
            if '2' not in b_str:
                ps_shots[L] += counts
    
    ps_rates = {L:ps_shots[L]/total_shots[L] for L in ps_shots}
    ps_stds = {}
    for L in ps_rates:
        p = ps_shots[L]/(total_shots[L]+2) # rule of 2
        ps_stds[L] = float(np.sqrt(p*(1-p)/total_shots[L]))
    
    return ps_rates, ps_stds


def rand_SU2s(n):
    """ represent an SU2 as tup: (theta, phi, angle) """
    
    SU2_list = []
    #r_theta = np.pi*np.random.rand(n)
    r_theta = np.array([2*np.arcsin(np.sqrt(r)) for r in np.random.rand(n)])
    r_phi = 2*np.pi*np.random.rand(n)
    r_angle = 2*np.pi*np.random.rand(n)
    
    for q in range(n):
        SU2 = (r_theta[q], r_phi[q], r_angle[q])
        SU2_list.append(SU2)
    
    return SU2_list


def SU2_pytket(circ, SU2, q):
    """
    circ (pytket Circuit object)
    SU2 (tup) : (theta, phi, angle)
    q (int) : qubit
    
    """
    
    theta = SU2[0]
    phi = SU2[1]
    angle = SU2[2]
    
    circ.Rz(-phi/np.pi, q)
    circ.Ry(-theta/np.pi, q)
    circ.Rz(angle/np.pi, q)
    circ.Ry(theta/np.pi, q)
    circ.Rz(phi/np.pi, q) 



def pytket_to_guppy(circ):
    func = guppy.load_pytket("circuit", circ)
    @guppy.comptime
    def main() -> None:
        q = array(qubit() for _ in range(circ.n_qubits))
        order_in_zones(q)
        func(q)
        order_in_zones(q)
        measure_and_record_leakage(q, True)

    return main.compile()
 

@guppy
def measure_and_record_leakage(q: array[qubit, n] @ owned, meas_leak: bool) -> None:

    if meas_leak:
        meas_leaked_array = array(measure_leaked(q_i) for q_i in q)
        for m in meas_leaked_array:
            if m.is_leaked():
                m.discard()
                result("c", 2)
            else:
                result("c", m.to_result().unwrap())
    else:
        b_str = measure_array(q)
        for b in b_str:
            result("c", b)