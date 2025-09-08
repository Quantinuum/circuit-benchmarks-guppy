# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 14:44:16 2025

DFL parser

@author: Karl.Mayer
"""

from typing import List, Any, Tuple, Dict
import msgpack
import re


from selene_sim import build
from selene_anduril import AndurilRuntimePlugin
from hugr.qsystem.result import QsysResult
from selene_sim.event_hooks import (
   CircuitExtractor, 
    MultiEventHook, 
    MetricStore
)

try:
    from selene_eldarion import register_eldarion, QtmPlatformPlugin
    register_eldarion()
except:
    pass


def get_selene_output(hugr, simulator, n_qubits):
    
    helios_runtime = AndurilRuntimePlugin()
    runner = build(hugr, eldarion=True, utilities=[QtmPlatformPlugin()])

    event_hook = MultiEventHook(
       event_hooks=[
          CircuitExtractor(), 
          MetricStore()
       ]
    )


    _ = QsysResult(runner.run_shots(
        simulator, 
        n_qubits=n_qubits,
        runtime=helios_runtime,
        n_shots=1,
        event_hook=event_hook,
    ))

    optimiser_output = event_hook.event_hooks[0].shots[0].get_optimiser_output()

    return optimiser_output


def keyword_to_value(kw: str) -> str:
    match kw:
        # storage leg
        case "Down":
            return "0"
        case "Up":
            return "1"
        # crystal config        
        case "BY":
            return "0"
        case "YB":
            return "1"
        # intrazone transport
        case "CC":
            return "0"
        case "LR":
            return "1"
        case "LC":
            return "2"
        case "CR":
            return "3"
        case _:
            return kw
        
def parse_output(optimizer_output: Dict[str, Any]) -> str:
    qmc_output = ""
    for operation in optimizer_output:
        if operation['op'] == 'CustomOperation':
            output = _parse_custom_operation(operation['tag'], operation['data'])
            name, ops = output.popitem()
            if name == "Transport":                
                if not isinstance(ops, str):
                    func, args = ops.popitem()                    
                    if func == 'ULDLRightAlign':
                        func_name = "uldl_right_align"
                    else:
                        func_name = "_".join(re.split('(?<=.)(?=[A-Z])', func)).lower()
                    if isinstance(args, list):
                        args = [keyword_to_value(item) for item in args]
                        function_args = ', '.join(map(str, args))
                        qmc_output += f"{func_name}({function_args})\n"
                    else:
                        if args == 'Up': args = 'U_leg'
                        if args == 'Down': args = 'D_leg'
                        if func == "Rot":
                            function_args = str(''.join(str(args)))
                        else:
                            function_args = ', '.join(str(args))
                        qmc_output += f"{func_name}({function_args})\n"
            else:
                zones, index = _get_gate_args(ops)
                new_indexes = [idx+1 for idx in index]
                match name:
                    # ADD CL COMMENTS FOR TESTING TRANSPORT ONLY
                    case "Reset":
                        qmc_output += f"{name.lower()}({new_indexes})\n"
                    case "Rxy":
                        angles0 = _get_args(ops[1], index)
                        angles1 = _get_args(ops[2], index)
                        qmc_output += f"{name.lower()}({new_indexes}, {angles0}, {angles1})\n"
                    case "Measure":
                        bit_index = _get_args(ops[1], index)
                        qmc_output += f"{name.lower()}({new_indexes}, {bit_index}, creg)\n"
                    case "ProtectedMeasureSetup":                        
                        qmc_output += f"{name.lower()}({new_indexes})\n"
                    case "ProtectedMeasureCleanup":
                        qmc_output += f"{name.lower()}()\n"
                    case "Rzz":
                        angles0 = _get_args(ops[2], index)
                        qmc_output += f"rzz({new_indexes}, {angles0})\n"
    return qmc_output


def parse_output2(optimizer_output: Dict[str, Any]) -> str:
    qmc_output = ""
    for operation in optimizer_output:
        if operation['op'] == 'CustomOperation':
            output = _parse_custom_operation(operation['tag'], operation['data'])
            name, ops = output.popitem()
            if name == "Transport":                
                if not isinstance(ops, str):
                    func, args = ops.popitem()
                    if func == 'ULDLRightAlign':
                        func_name = "uldl_right_align"
                    else:
                        func_name = "_".join(re.split('(?<=.)(?=[A-Z])', func)).lower()
                    if isinstance(args, list):
                        args = ['U_leg' if item == 'Up' else item for item in args]
                        args = ['D_leg' if item == 'Down' else item for item in args]
                        if func == "RotEnter":
                            args = [args[0], args[2], args[3], args[1]]
                        function_args = ', '.join(map(str, args))
                        qmc_output += f"{func_name}({function_args})\n"
                    else:
                        if args == 'Up': args = 'U_leg'
                        if args == 'Down': args = 'D_leg'
                        if func == "Rot":
                            function_args = str(''.join(str(args)))                                                
                        else:
                            function_args = ', '.join(str(args))
                        qmc_output += f"{func_name}({function_args})\n"
            else:
                zones, index = _get_gate_args(ops)
                new_indexes = [idx+1 for idx in index]
                match name:

                    # ADD CL COMMENTS FOR TESTING TRANSPORT ONLY
                    case "Reset":
                        qmc_output += f"{name.lower()}({new_indexes})\n"
                    case "Rxy":
                        angles0 = _get_args(ops[1], index)
                        angles1 = _get_args(ops[2], index)
                        qmc_output += f"{name.lower()}({new_indexes}, {angles0}, {angles1})\n"
                    case "Measure":
                        bit_index = _get_args(ops[1], index)
                        qmc_output += f"{name.lower()}({new_indexes}, {bit_index}, creg=creg)\n"
                    case "ProtectedMeasureSetup":                        
                        qmc_output += f"{name.lower()}({new_indexes})\n"
                    case "ProtectedMeasureCleanup":
                        qmc_output += f"{name.lower()}()\n"
                    #case "Rzz24":
                    case "Rzz":
                        rzz_indexes = [idx*2 for idx in new_indexes]
                        angles0 = _get_args(ops[2], index)
                        qmc_output += f"rzz({rzz_indexes}, {angles0})\n"    
    return qmc_output


def _get_gate_args(arguments: List[Any]) -> Tuple[List[int], List[int]]:    
    if not arguments or len(arguments) == 0:
        return [], []
    
    zones_data = arguments[0]
    
    # Handle case where zones_data is a list of integers (your current case)
    if isinstance(zones_data, list) and all(isinstance(x, (int, float)) for x in zones_data):
        # Get non-zero values and their indices
        zones = []
        indices = []
        for index, zone in enumerate(zones_data):
            if zone != 0:  # Assuming 0 means unused
                zones.append(int(zone))
                indices.append(index)
        return zones, indices
    
    # Handle case where zones_data contains dictionaries (original logic)
    elif isinstance(zones_data, list):
        try:
            zones, index = zip(*[
                (element.popitem()[1], int(index)) 
                for index, element in enumerate(zones_data) 
                if element is not None and isinstance(element, dict)
            ])
            return list(zones), list(index)
        except Exception:
            return [], []
    
    return [], []

def _get_args(ops: List[float], index: List[int]):
    args = []
    for i in index:
        args += [ops[i]]
    return args

def _parse_custom_operation(tag: int, data: bytes) -> dict:
    return msgpack.unpackb(data)

