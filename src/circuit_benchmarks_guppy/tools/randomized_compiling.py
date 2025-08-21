# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:55:56 2025

Module with Guppy functions for randomized compiling
with run-time randomness

@author: Karl.Mayer
"""

from guppylang import guppy
from guppylang.std.quantum import qubit, cz, z, x, y
from guppylang.std.qsystem import zz_phase
from guppylang.std.angles import angle


from guppylang.std.qsystem.random import RNG

@guppy
def rand_comp_cz(q0: qubit, q1: qubit, rng: RNG) -> None:

    randval = rng.random_int_bounded(16)
    
    if randval == 1:
        x(q0)
    elif randval == 2:
        y(q0)
    elif randval == 3:
        z(q0)
    elif randval == 4:
        x(q1)
    elif randval == 5:
        x(q1)
        x(q0)
    elif randval == 6:
        x(q1)
        y(q0)
    elif randval == 7:
        x(q1)
        z(q0)
    elif randval == 8:
        y(q1)
    elif randval == 9:
        y(q1)
        x(q0)
    elif randval == 10:
        y(q1)
        y(q0)
    elif randval == 11:
        y(q1)
        z(q0)
    elif randval == 12:
        z(q1)
    elif randval == 13:
        z(q1)
        x(q0)
    elif randval == 14:
        z(q1)
        y(q0)
    elif randval == 15:
        z(q1)
        z(q0)
    
    cz(q0, q1)

    if randval == 1:
        x(q0)
        z(q1)
    elif randval == 2:
        y(q0)
        z(q1)
    elif randval == 3:
        z(q0)
    elif randval == 4:
        x(q1)
        z(q0)
    elif randval == 5:
        y(q1)
        y(q0)
    elif randval == 6:
        y(q1)
        x(q0)
    elif randval == 7:
        x(q1)
    elif randval == 8:
        y(q1)
        z(q0)
    elif randval == 9:
        x(q1)
        y(q0)
    elif randval == 10:
        x(q1)
        x(q0)
    elif randval == 11:
        y(q1)
    elif randval == 12:
        z(q1)
    elif randval == 13:
        x(q0)
    elif randval == 14:
        y(q0)
    elif randval == 15:
        z(q1)
        z(q0)
        
        
@guppy
def rand_comp_rzz(q0: qubit, q1: qubit, rng: RNG) -> None:

    randval = rng.random_int_bounded(16)
    
    if randval == 1:
        x(q0)
    elif randval == 2:
        y(q0)
    elif randval == 3:
        z(q0)
    elif randval == 4:
        x(q1)
    elif randval == 5:
        x(q1)
        x(q0)
    elif randval == 6:
        x(q1)
        y(q0)
    elif randval == 7:
        x(q1)
        z(q0)
    elif randval == 8:
        y(q1)
    elif randval == 9:
        y(q1)
        x(q0)
    elif randval == 10:
        y(q1)
        y(q0)
    elif randval == 11:
        y(q1)
        z(q0)
    elif randval == 12:
        z(q1)
    elif randval == 13:
        z(q1)
        x(q0)
    elif randval == 14:
        z(q1)
        y(q0)
    elif randval == 15:
        z(q1)
        z(q0)
    
    zz_phase(q0, q1, angle(0.5))

    if randval == 1:
        y(q0)
        z(q1)
    elif randval == 2:
        x(q0)
        z(q1)
    elif randval == 3:
        z(q0)
    elif randval == 4:
        z(q0)
        y(q1)
    elif randval == 5:
        x(q1)
        x(q0)
    elif randval == 6:
        x(q1)
        y(q0)
    elif randval == 7:
        y(q1)
    elif randval == 8:
        z(q0)
        x(q1)
    elif randval == 9:
        y(q1)
        x(q0)
    elif randval == 10:
        y(q1)
        y(q0)
    elif randval == 11:
        x(q1)
    elif randval == 12:
        z(q1)
    elif randval == 13:
        y(q0)
    elif randval == 14:
        x(q0)
    elif randval == 15:
        z(q1)
        z(q0)    
    
