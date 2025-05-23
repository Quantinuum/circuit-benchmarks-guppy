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

import datetime
import pickle
from copy import deepcopy

from collections import Counter
from hugr.qsystem.result import QsysResult
from selene_sim import build

import qnexus
from qnexus.config import CONFIG

CONFIG.domain = "qa.myqos.com"


class Experiment():
    
    def __init__(self, **kwargs): 
        self.protocol = None # string specifying protocol name
        self.parameters = None # dict describing experiment parameters
        self.options = {'detect_leakage':False}
        self.options['order_qubits'] = False
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
    
    
    def to_program_refs(self):
        """ returns list of qnexus program references """
        
        program_refs = []
        for j, sett in enumerate(self.settings):
            prog = self.make_circuit(sett)
            prog_ref = qnexus.hugr.upload(hugr_package=prog.to_executable_package().package,
                                          name=f"{self.protocol} circuit {j}")
            program_refs.append(prog_ref)
        
        return program_refs
    
    
    def submit(self, shots, backend_config, save=True, **kwargs):
        """ returns qnexus ExecuteJobRef """
        
        program_refs = self.to_program_refs()
        
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        execute_job_ref = qnexus.start_execute_job(
            programs=program_refs,
            n_shots=[shots for _ in range(len(program_refs))],
            backend_config=backend_config,
            name=f"{self.protocol} job" + timestamp,
        )
        
        # save experiment
        if save == True:
            self.save(filename=self.filename)
        
        return execute_job_ref
    
    
    def retrieve(self, execute_job_ref, save=True):
        
        self.results = {}
        for j, sett in enumerate(self.settings):
            job_results = qnexus.jobs.results(execute_job_ref)[j].download_result()
            outcomes = dict(Counter("".join(f"{e[1]}" for e in shot.entries) for shot in job_results.results))
            self.results[sett] = outcomes
    
        if save == True:
            self.save()
                

    def sim(self, shots, error_model, simulator, verbose=True):
        """ simulate experiment using selene_sim simulator
            simulator: Stim() or Quest()
        """
        
        protocol = self.protocol
        n_qubits = self.n_qubits
        
        self.shots = shots
        self.results = {}
        print('Simulating ...')
        for j, sett in enumerate(self.settings):
            setting = self.settings[j]
            prog = self.make_circuit(setting)
            runner = build(prog, f'{protocol} circuit {j}')
            shot_results = QsysResult(runner.run_shots(simulator,
                                    n_qubits=n_qubits,
                                    n_shots=shots,
                                    error_model=error_model))
            outcomes = dict(Counter("".join(f"{e[1]}" for e in shot.entries) for shot in shot_results.results))
            self.results[sett] = outcomes
            if verbose:
                print(f'{j+1}/{len(self.settings)} circuits complete')
                
    
    def check_for_results(self):
        
        try:
            results = self.results
        except:
            print('Error: Experiment has no results')
        
        return results
        
    
    def analyze_results(self):
        # uses Analysis object to analyze results, if non-empty
        #analysis = self.analysis
        pass
    
    

    
    
    