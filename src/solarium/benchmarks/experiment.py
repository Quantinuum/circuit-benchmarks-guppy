#!/usr/bin/env python

##############################################################################
#
# QUANTINUUM LLC CONFIDENTIAL & PROPRIETARY.
# This work and all information and expression are the property of
# Quantinuum LLC, are Quantinuum LLC Confidential & Proprietary,
# contain trade secrets and may not, in whole or in part, be licensed,
# used, duplicated, disclosed, or reproduced for any purpose without prior
# written permission of Quantinuum LLC.
#
# In the event of publication, the following notice shall apply:
# (c) 2022 Quantinuum LLC. All Rights Reserved.
#
##############################################################################

''' general benchmarking Experiment class object  '''

#from typing import Optional, Union


from copy import deepcopy
from collections import Counter
import datetime
import numpy as np
import pickle

from hugr.qsystem.result import QsysResult
from selene_core import Utility

from solarium.tools.dfl_parser import get_selene_output, parse_output
from solarium.tools.imports import import_optional


# if qnexus:
#     qnexus.config.domain = "qa.myqos.com"





class Experiment():
    
    def __init__(self, **kwargs): 
        self.protocol = None # string specifying protocol name
        self.parameters = None # dict describing experiment parameters
        self.options = {'measure_leaked':False}
        self.options['order_in_zones'] = False
        self.circuits = [] # list of circuit objects (HUGRs)
        self.settings = [] # list of circuit settings
        self.results = None # dict of (key=setting:value=result object) pairs
        self.analysis = None # Analysis class object
        self.filename = kwargs.get('filename', None)
        
        
        
    #### Methods ####
        
    def save(self, filename=None):      
        # save as pickle file to local directory
        
        if filename != None:
            self.filename = filename
            
        elif self.filename == None:
            self.filename = 'experiment.p'
        
        pickle.dump(self, open(self.filename, 'wb'))
        print(f'Experiment saved! {self.filename}')
        
    
    # static method for loading    
    @staticmethod
    def load(filename):
        # load experiment from file
        exp = pickle.load(open(filename, 'rb'))
        
        return exp
        
    
    def copy(self):
        # copy experiment object
        return deepcopy(self)
        
    
    def add_circuit(self, circuit):
        # append circuit object
        self.circuits.append(circuit)
        
    
    def add_setting(self, setting):
        # append setting
        self.settings.append(setting)
        
    
    def get_dfl(self, setting, simulator):
        
        n_qubits = self.n_qubits
        hugr = self.make_circuit(setting)
        
        optimiser_output = get_selene_output(hugr, simulator, n_qubits)
        
        return parse_output(optimiser_output)
    
    
    def get_hqc(self, setting, shots, simulator):
        """ uses Selene event hook to estimate HQC cost of circuit """
        selene_sim = import_optional("selene_sim", errors="raise")
        selene_anduril = import_optional("selene_anduril", errors="raise")
        
        hugr = self.make_circuit(setting)
        runner = selene_sim.build(hugr)
    
        event_hook = selene_sim.event_hooks.CircuitExtractor()
        runtime = selene_anduril.AndurilRuntimePlugin()
    
        event_hook = selene_sim.event_hooks.MultiEventHook([
            selene_sim.event_hooks.CircuitExtractor(),
            selene_sim.event_hooks.MetricStore()
        ])

        qsys_result = QsysResult(runner.run_shots(
            simulator,
            event_hook=event_hook,
            runtime=runtime,
            n_qubits=self.n_qubits,
            n_shots=1
        ))

        resources = event_hook.event_hooks[1].shots[0].get("user_program")
        N_1q = resources['rxy_count']
        N_2q = resources['rzz_count']
        N_m = resources['measure_request_count']
        hqc = 5 + shots*(N_1q + 10*N_2q + 5*N_m)/5000
        
        return hqc
    
    
    def to_program_refs(self, shuffle=False):
        """ returns list of qnexus program references """
        qnexus = import_optional("qnexus", errors="raise")
        
        n_prog = len(self.settings)
        
        if shuffle:
            submit_order = [int(j) for j in np.random.choice(n_prog, size=n_prog, replace=False)]
        else:
            submit_order = [int(j) for j in range(n_prog)]
        
        self.submit_order = submit_order
        program_refs = []
        for j in submit_order:
            sett = self.settings[j]
            prog = self.make_circuit(sett)
            prog_ref = qnexus.hugr.upload(hugr_package=prog,
                                          name=f"{self.protocol} circuit {j}")
            program_refs.append(prog_ref)
        
        return program_refs
    
    
    def submit(self, shots, backend_config, shuffle=False, save=True, **kwargs):
        """ returns qnexus ExecuteJobRef """
        qnexus = import_optional("qnexus", errors="raise")
        
        self.shots = shots
        
        program_refs = self.to_program_refs(shuffle=shuffle)
        
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        execute_job_ref = qnexus.start_execute_job(
            programs=program_refs,
            n_shots=[shots for _ in range(len(program_refs))],
            backend_config=backend_config,
            name=f"{self.protocol} job" + timestamp,
        )
        
        # add Nexus UUID as attribute
        self.nexus_id = execute_job_ref.id
        
        # save experiment
        if save == True:
            self.save(filename=self.filename)
        
        return execute_job_ref
    
    
    def retrieve(self, execute_job_ref, save=True):
        qnexus = import_optional("qnexus", errors="raise")

        job_results_ref = qnexus.jobs.results(execute_job_ref, allow_incomplete=True)
        
        results = {}
        for i, j in enumerate(self.submit_order):
            if i < len(job_results_ref):
                sett = self.settings[j]
                job_results = job_results_ref[i].download_result()
                outcomes = dict(Counter("".join(f"{e[1]}" for e in shot.entries) for shot in job_results.results))
                results[sett] = outcomes
            
        # reorder results
        self.results = {}
        for sett in self.settings:
            if sett in results:
                self.results[sett] = results[sett]
    
        if save == True:
            self.save()
                

    def sim(self, shots, error_model, simulator, extensions: list[Utility] | None = None, eldarion: bool = False, verbose=True):
        """ simulate experiment using selene_sim simulator
            simulator: Stim() or Quest()
            shots: int or dict of shots for each setting label
        """

        if not extensions:
            extensions = []

        selene_sim = import_optional("selene_sim", errors="raise")
        selene_anduril = import_optional("selene_anduril", errors="warn")
        
        protocol = self.protocol
        n_qubits = self.n_qubits
        
        self.shots = shots
        self.results = {}
        print('Simulating ...')
        for j, sett in enumerate(self.settings):
            if isinstance(shots, int):
                n_shots = shots
            elif isinstance(shots, dict):
                n_shots = shots[sett]
            setting = self.settings[j]
            prog = self.make_circuit(setting)
            runner = selene_sim.build(prog, f'{protocol} circuit {j}', eldarion=eldarion, utilities=extensions)
            
            if selene_anduril:
                shot_results = QsysResult(runner.run_shots(simulator,
                                        n_qubits=n_qubits,
                                        n_shots=n_shots,
                                        error_model=error_model,
                                        runtime=selene_anduril.AndurilRuntimePlugin()))
            else:
                shot_results = QsysResult(runner.run_shots(simulator,
                                        n_qubits=n_qubits,
                                        n_shots=n_shots,
                                        error_model=error_model)
                                         )
                
            outcomes = dict(Counter("".join(f"{e[1]}" for e in shot.entries) for shot in shot_results.results))
            self.results[sett] = outcomes
            if verbose:
                print(f'{j+1}/{len(self.settings)} circuits complete')
                
    
    def check_for_results(self):
        
        try:
            results = self.results
        except Exception:
            print('Error: Experiment has no results')
        
        return results
        
    
    def analyze_results(self):
        # uses Analysis object to analyze results, if non-empty
        #analysis = self.analysis
        pass
    
    

    
    
    