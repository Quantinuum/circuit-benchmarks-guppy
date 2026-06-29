# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 09:42:09 2025

funtion for measuring qubit array using the leakage heralded measurement

@author: Karl.Mayer
"""

from guppylang import guppy
from guppylang.std.builtins import array, owned, result
from guppylang.std.quantum import measure_array, qubit
from guppylang.std.qsystem import measure_leaked


n = guppy.nat_var("n")

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
            
            
