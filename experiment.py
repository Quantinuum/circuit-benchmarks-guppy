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

import pickle
from copy import deepcopy

from collections import Counter
from guppylang.qsys_result import QsysResult
from selene_sim import build
    

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
    
    
    def from_batch(self, batch):
        # create an Experiment from a qjobs Batch object
        pass
    
    
    #def submit(self, shots, machine, shuffle=True, save=True, filename=None,
    #           **kwargs):
        
                  
        # create batch
    #    if self.batch == None:
    #        self.batch = self.to_batch(shots, machine, **kwargs)
        
        # submit batch
    #    self.batch.submit(shuffle=shuffle)
        
    #    # save experiment
    #    if save == True:
    #        self.save(filename=filename)
                

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
    
    

    
    
    