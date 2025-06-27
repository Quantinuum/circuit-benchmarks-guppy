
from types import *
from typing import *

from guppylang import guppy
from guppylang.std.quantum import x
from guppylang.std.angles import pi
from guppylang.std.qsystem.random import RNG
from guppylang.std.qsystem.utils import get_current_shot
from hugr.package import FuncDefnPointer

from eris.lib.iceberg import (
    Iceberg,
    iceberg_module,
    logical_xx_phase,
    logical_yy_phase,
    logical_zz_phase,
    logical_rx,
    logical_rz,
    logical_x,
    logical_y,
    logical_z,
    logical_global_x_rotation,
    logical_global_y_rotation,
    logical_global_z_rotation,
    logical_global_x_rotation_except,
    logical_global_y_rotation_except,
    logical_global_z_rotation_except
)

n = guppy.nat_var("n")

@guppy
def random_pair_gate(src: Iceberg[n], n_logical_qubits: int,  q0: int, q1: int, axis: int, sign: int) -> None:
    ''' 
    Randomly pair each physical qubit in Iceberg code block 
    q = (0-n_logical_qubits): correspond to logical qubit
    q = n_logical_qubit: top
    q = n_logical_qubit + 1: bottom

    '''
    # TQ gates
    if q0 < n_logical_qubits and q1 < n_logical_qubits:
        if axis == 0:
            logical_xx_phase(src, q0, q1, (-1)**sign*0.5*pi)
        elif axis == 1:
            logical_yy_phase(src, q0, q1, (-1)**sign*0.5*pi)
        elif axis == 2:
            logical_zz_phase(src, q0, q1, (-1)**sign*0.5*pi)

    # q1 top qubit
    elif q0 < n_logical_qubits and q1 == n_logical_qubits:
        if axis == 0:
            logical_rx(src, q0, (-1)**sign*0.5*pi)
        elif axis == 1:
            logical_global_x_rotation_except(src, q0, (-1)**sign*0.5*pi) 
        elif axis == 2:
            logical_global_y_rotation_except(src, q0, (-1)**sign*0.5*pi)
        elif axis == 2:
            logical_global_z_rotation_except(src, q0, (-1)**sign*0.5*pi)

     # q0 top qubit
    elif q0 == n_logical_qubits + 1 and q1 < n_logical_qubits:
        if axis == 0:
            logical_rx(src, q1, (-1)**sign*0.5*pi)
        elif axis == 1:
            logical_global_y_rotation_except(src, q1, (-1)**sign*0.5*pi)
        elif axis == 2:
            logical_global_x_rotation_except(src, q1, (-1)**sign*0.5*pi)

    # bottom qubit
    elif q0 < n_logical_qubits and q1 == n_logical_qubits + 2:
        if axis == 0:
            logical_global_x_rotation_except(src, q0, (-1)**sign*0.5*pi)
        elif axis == 1:
            logical_global_y_rotation_except(src, q0, (-1)**sign*0.5*pi)
        elif axis == 2:
            logical_global_z_rotation_except(src, q0, (-1)**sign*0.5*pi)
    elif q0 == n_logical_qubits + 2 and q1 < n_logical_qubits:
        if axis == 0:
            logical_global_x_rotation_except(src, q1, (-1)**sign*0.5*pi)
        elif axis == 1:
            logical_global_y_rotation_except(src, q1, (-1)**sign*0.5*pi)
        elif axis == 2:
            logical_global_z_rotation_except(src, q1, (-1)**sign*0.5*pi)

    # global gates
    if q0 > n_logical_qubits and q1 > n_logical_qubits:
        if axis == 0:
            logical_global_x_rotation(src, (-1)**sign*0.5*pi)
        elif axis == 1:
            logical_global_y_rotation(src, (-1)**sign*0.5*pi)
        elif axis == 2:
            logical_global_z_rotation(src, (-1)**sign*0.5*pi)
            

# def random_pauli(src : Iceberg[n], rng):
    # pass




 # applies a Pauli logical X gate to the idx-th logical qubit of the iceberg code
@guppy(iceberg_module)
@no_type_check
def logical_syndrome_x(ice : Iceberg[n]) -> None:
    for i in range(n):
        x(ice.block[i])
    x(ice.top)
    x(ice.bottom)


@guppy
def rand_comp_rzz(src: Iceberg[n], q0: int, q1: int, rng: RNG) -> None:

    randval = rng.random_int_bounded(16)
    
    if randval == 1:
        logical_x(src, q0)
    elif randval == 2:
        logical_y(src, q0)
    elif randval == 3:
        logical_z(src, q0)
    elif randval == 4:
        logical_x(src, q1)
    elif randval == 5:
        logical_x(src, q1)
        logical_x(src, q0)
    elif randval == 6:
        logical_x(src, q1)
        logical_y(src, q0)
    elif randval == 7:
        logical_x(src, q1)
        logical_z(src, q0)
    elif randval == 8:
        logical_y(src, q1)
    elif randval == 9:
        logical_y(src, q1)
        logical_x(src, q0)
    elif randval == 10:
        logical_y(src, q1)
        logical_y(src, q0)
    elif randval == 11:
        logical_y(src, q1)
        logical_z(src, q0)
    elif randval == 12:
        logical_z(src, q1)
    elif randval == 13:
        logical_z(src, q1)
        logical_x(src, q0)
    elif randval == 14:
        logical_z(src, q1)
        logical_y(src, q0)
    else:
        logical_z(src, q1)
        logical_z(src, q0)
    
    logical_zz_phase(src, q0, q1, 0.5*pi)

    if randval == 1:
        logical_y(src, q0)
        logical_z(src, q1)
    elif randval == 2:
        logical_x(src, q0)
        logical_z(src, q1)
    elif randval == 3:
        logical_z(src, q0)
    elif randval == 4:
        logical_z(src, q0)
        logical_y(src, q1)
    elif randval == 5:
        logical_x(src, q1)
        logical_x(src, q0)
    elif randval == 6:
        logical_x(src, q1)
        logical_y(src, q0)
    elif randval == 7:
        logical_y(src, q1)
    elif randval == 8:
        logical_z(src, q0)
        logical_x(src, q1)
    elif randval == 9:
        logical_y(src, q1)
        logical_x(src, q0)
    elif randval == 10:
        logical_y(src, q1)
        logical_y(src, q0)
    elif randval == 11:
        logical_x(src, q1)
    elif randval == 12:
        logical_z(src, q1)
    elif randval == 13:
        logical_y(src, q0)
    elif randval == 14:
        logical_x(src, q0)
    else:
        logical_z(src, q1)
        logical_z(src, q0)


@guppy
def clifford_gates_1Q(src: Iceberg[n], cliff_ind: int, qubit0: int) -> None:

    if cliff_ind == 0:
        pass
    elif cliff_ind == 1:
        logical_rx(src, qubit0, -0.5*pi)
    elif cliff_ind == 2:
        logical_x(src, qubit0)
    elif cliff_ind == 3:
        logical_rx(src, qubit0, 0.5*pi)
    elif cliff_ind == 4:
        logical_y(src, qubit0)
    elif cliff_ind == 5:
        logical_rz(src, qubit0, -0.5*pi)
    elif cliff_ind == 6:
        logical_z(src, qubit0)
    elif cliff_ind == 7:
        logical_rz(src, qubit0, 0.5*pi)
    elif cliff_ind == 8:
        logical_rx(src, qubit0, 0.5*pi)
        logical_z(src, qubit0)
    elif cliff_ind == 9:
        logical_rx(src, qubit0, -0.5*pi)
        logical_rz(src, qubit0, -0.5*pi)
    elif cliff_ind == 10:
        logical_rx(src, qubit0, -0.5*pi)
        logical_z(src, qubit0)
    elif cliff_ind == 11:
        logical_rx(src, qubit0, -0.5*pi)
        logical_rz(src, qubit0, 0.5*pi)
    elif cliff_ind == 12:
        logical_x(src, qubit0)
        logical_rz(src, qubit0, -0.5*pi)
    elif cliff_ind == 13:
        logical_x(src, qubit0)
        logical_rz(src, qubit0, 0.5*pi)
    elif cliff_ind == 14:
        logical_rx(src, qubit0, 0.5*pi)
        logical_rz(src, qubit0, -0.5*pi)
    elif cliff_ind == 15:
        logical_rx(src, qubit0, 0.5*pi)
        logical_rz(src, qubit0, 0.5*pi)
    elif cliff_ind == 16:
        logical_rz(src, qubit0, -0.5*pi)
        logical_rx(src, qubit0, -0.5*pi)
    elif cliff_ind == 17:
        logical_rz(src, qubit0, -0.5*pi)
        logical_rx(src, qubit0, 0.5*pi)
    elif cliff_ind == 18:
        logical_rz(src, qubit0, 0.5*pi)
        logical_rx(src, qubit0, -0.5*pi)
    elif cliff_ind == 19:
        logical_rz(src, qubit0, 0.5*pi)
        logical_rx(src, qubit0, 0.5*pi)
    elif cliff_ind == 20:
        logical_rz(src, qubit0, -0.5*pi)
        logical_rx(src, qubit0, -0.5*pi)
        logical_rz(src, qubit0, -0.5*pi)
    elif cliff_ind == 21:
        logical_rz(src, qubit0, -0.5*pi)
        logical_rx(src, qubit0, 0.5*pi)
        logical_rz(src, qubit0, 0.5*pi)
    elif cliff_ind == 22:
        logical_rz(src, qubit0, -0.5*pi)
        logical_rx(src, qubit0, 0.5*pi)
        logical_rz(src, qubit0, -0.5*pi)
    elif cliff_ind == 23:
        logical_rz(src, qubit0, -0.5*pi)
        logical_rx(src, qubit0, -0.5*pi)
        logical_rz(src, qubit0, 0.5*pi)


def postselect_results(results, nqubits):

    new_results = {}
    for setting, counts in results.items():
        new_results[setting] = {key: val for key, val in counts.items() if len(key) == nqubits}

    return new_results