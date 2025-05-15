# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:43:37 2020

@author: Karl Mayer
"""

import numpy as np

from guppylang import guppy
from guppylang.std.builtins import py
from guppylang.std.quantum import qubit, h, z, x, y, s, sdg
    

@guppy
def apply_SQ_Clifford(q: qubit, gate_index:int) -> None:

    if gate_index == 1:
        h(q)
    elif gate_index == 2:
        s(q)
    elif gate_index == 3:
        sdg(q)
    elif gate_index == 4:
        x(q)
    elif gate_index == 5:
        y(q)
    elif gate_index == 6:
        z(q)
    elif gate_index == 7: # (-Y, -Z, X)
        s(q)
        h(q)
    elif gate_index == 8: # (Y, Z, X)
        sdg(q)
        h(q)
    elif gate_index == 9: # (Z, Y, -X)
        x(q)
        h(q)
    elif gate_index == 10: # (-Z, -Y, -X)
        y(q)
        h(q)
    elif gate_index == 11: # (-Z, Y, X)
        z(q)
        h(q)
    elif gate_index == 12: # (Z, X, Y)
        h(q)
        s(q)
    elif gate_index == 13: # (Y, X, -Z)
        x(q)
        s(q)
    elif gate_index == 14: # (-Y, -X, -Z)
        y(q)
        s(q)
    elif gate_index == 15: # (Z, -X, -Y)
        h(q)
        sdg(q)
    elif gate_index == 16:
        h(q)
        s(q)
        h(q)
    elif gate_index == 17: # (-Y, Z,-X)
        x(q)
        s(q)
        h(q)
    elif gate_index == 18: # (Y,-Z,-X)
        y(q)
        s(q)
        h(q)
    elif gate_index == 19:
        h(q)
        sdg(q)
        h(q)
    elif gate_index == 20: # (-X, Z, Y)
        sdg(q)
        h(q)
        s(q)
    elif gate_index == 21: # (-Z, X, -Y)
        y(q)
        h(q)
        s(q)
    elif gate_index == 22: # (-Z, -X, Y)
        z(q)
        h(q)
        s(q)
    elif gate_index == 23: # (-X, -Z, -Y)
        s(q)
        h(q)
        sdg(q)
        
        
@guppy
def apply_SQ_Clifford_inv(q: qubit, gate_index:int) -> None:

    inv_list = py([0, 1, 3, 2, 4, 5, 6, 15, 12, 11, 10, 9, 8, 13, 14, 7, 19, 22, 21, 16, 20, 18, 17, 23])
    inv_index = inv_list[gate_index]
    apply_SQ_Clifford(q, inv_index)


gate_convert = {0:0, 1:4, 2:5, 3:6, 4:16, 5:19, 6:20, 7:23, 8:13, 9:2, 10:3, 11:14,
                12:8, 13:18, 14:17, 15:7, 16:12, 17:15, 18:21, 19:22, 20:9, 21:1, 22:11, 23:10}


def Clifford_list(indices):
    
    # notation: ('+X', '+Y', '+Z') means X -> +X, Y -> +Y, Z -> +Z
    
    Clifford_dict = {}
    Clifford_dict[0] = ('+X', '+Y', '+Z') # I
    Clifford_dict[1] = ('+Z', '-Y', '+X') # H
    Clifford_dict[2] = ('+Y', '-X', '+Z') # S
    Clifford_dict[3] = ('-Y', '+X', '+Z') # Sdg
    Clifford_dict[4] = ('+X', '-Y', '-Z') # X
    Clifford_dict[5] = ('-X', '+Y', '-Z') # Y
    Clifford_dict[6] = ('-X', '-Y', '+Z') # Z
    Clifford_dict[7] = ('-Y', '-Z', '+X') # Ry(pi/2) Sdg
    Clifford_dict[8] = ('+Y', '+Z', '+X') # Ry(pi/2) S
    Clifford_dict[9] = ('+Z', '+Y', '-X') # Ry(-pi/2)
    Clifford_dict[10] = ('-Z', '-Y', '-X') # Z Ry(pi/2)
    Clifford_dict[11] = ('-Z', '+Y', '+X') # Ry(pi/2)
    Clifford_dict[12] = ('+Z', '+X', '+Y') # Sdg Ry(-pi/2)
    Clifford_dict[13] = ('+Y', '+X', '-Z') # Y S
    Clifford_dict[14] = ('-Y', '-X', '-Z') # Y Sdg
    Clifford_dict[15] = ('+Z', '-X', '-Y') # S Ry(-pi/2)
    Clifford_dict[16] = ('+X', '+Z', '-Y') # Rx(pi/2)
    Clifford_dict[17] = ('-Y', '+Z', '-X') # Ry(-pi/2) Sdg
    Clifford_dict[18] = ('+Y', '-Z', '-X') # Ry(-pi/2) S
    Clifford_dict[19] = ('+X', '-Z', '+Y') # Rx(-pi/2)
    Clifford_dict[20] = ('-X', '+Z', '+Y') # Rx(-pi/2) Z
    Clifford_dict[21] = ('-Z', '+X', '-Y') # Sdg Ry(pi/2)
    Clifford_dict[22] = ('-Z', '-X', '+Y') # S Ry(pi/2)
    Clifford_dict[23] = ('-X', '-Z', '-Y') # Rx(pi/2) Z
    
    C = [Clifford_dict[x] for x in indices]
    
    return C


def inverse_Clifford(C):
    """ C is of form (+X, +Y, +Z) """
    
    D = {}
    D[('+X', '+Y', '+Z')] = ('+X', '+Y', '+Z')
    D[('+X', '-Y', '-Z')] = ('+X', '-Y', '-Z')
    D[('-X', '+Y', '-Z')] = ('-X', '+Y', '-Z')
    D[('-X', '-Y', '+Z')] = ('-X', '-Y', '+Z')
    D[('+X', '+Z', '-Y')] = ('+X', '-Z', '+Y')
    D[('+X', '-Z', '+Y')] = ('+X', '+Z', '-Y')
    D[('-X', '+Z', '+Y')] = ('-X', '+Z', '+Y')
    D[('-X', '-Z', '-Y')] = ('-X', '-Z', '-Y')
    D[('+Y', '+X', '-Z')] = ('+Y', '+X', '-Z')
    D[('+Y', '-X', '+Z')] = ('-Y', '+X', '+Z')
    D[('-Y', '+X', '+Z')] = ('+Y', '-X', '+Z')
    D[('-Y', '-X', '-Z')] = ('-Y', '-X', '-Z')
    D[('+Y', '+Z', '+X')] = ('+Z', '+X', '+Y')
    D[('+Y', '-Z', '-X')] = ('-Z' ,'+X', '-Y')
    D[('-Y', '+Z', '-X')] = ('-Z', '-X', '+Y')  
    D[('-Y', '-Z', '+X')] = ('+Z', '-X', '-Y')
    D[('+Z', '+X', '+Y')] = ('+Y', '+Z', '+X')
    D[('+Z', '-X', '-Y')] = ('-Y', '-Z', '+X')
    D[('-Z', '+X', '-Y')] = ('+Y', '-Z', '-X')
    D[('-Z', '-X', '+Y')] = ('-Y', '+Z', '-X')
    D[('+Z', '+Y', '-X')] = ('-Z', '+Y', '+X')
    D[('+Z', '-Y', '+X')] = ('+Z', '-Y', '+X')
    D[('-Z', '+Y', '+X')] = ('+Z', '+Y', '-X')
    D[('-Z', '-Y', '-X')] = ('-Z', '-Y', '-X')
    
    return D[C]
    
    
    

def initialize_stabilizers(n):
    """ make representation of |0...0>,
        stabilizers ZI...I, ...,I...IZ  """
    
    # repersent stabilizers as strs, i.e., IZ = '+IZ'
    
    stabilizers = []
    for i in range(n):
        stab = '+' # single stabilizer
        for j in range(n):
            if j != i:
                stab += 'I'
            elif j == i:
                stab += 'Z'
        stabilizers.append(stab)
    
    return stabilizers

def update_stab_SQ(C, stab):
    """   C (list): ex: H \tensor S = [('+Z', '-Y', '+X'), ('+Y', '-X', '+Z')] 
        stab (str): ex: '-YZ'                           """
    
    # read number of qubits
    n = len(stab) - 1
    
    new_stab = ''
    sign = stab[0]
    for j in range(n): # stab index
        q = j # qubit index
        if stab[j+1] == 'I':
            new_stab += 'I'
            new_sign = '+'
        if stab[j+1] == 'X':
            new_stab += C[q][0][1]
            new_sign = C[q][0][0]
        if stab[j+1] == 'Y':
            new_stab += C[q][1][1]
            new_sign = C[q][1][0]
        if stab[j+1] == 'Z':
            new_stab += C[q][2][1]
            new_sign = C[q][2][0]
        if new_sign == '-':
            if sign == '+':
                sign = '-'
            elif sign == '-':
                sign = '+'
    new_stab = sign + new_stab
    
    return new_stab


def update_stab_ZZ(stab, qubit_pairs):
    
    U_ZZ_dict = {'II':'+II', 'IX':'+ZY', 'IY':'-ZX', 'IZ':'+IZ', 'XI':'+YZ', 'XX':'+XX', 'XY':'+XY', 'XZ':'+YI', 'YI':'-XZ', 'YX':'+YX', 'YY':'+YY', 'YZ':'-XI', 'ZI':'+ZI', 'ZX':'+IY', 'ZY':'-IX', 'ZZ':'+ZZ'}
    
    # read number of qubits
    #n = len(stab) - 1
    for q_pair in qubit_pairs:
        j0 = q_pair[0]+1
        j1 = q_pair[1]+1
        key = stab[j0] + stab[j1]
        val = U_ZZ_dict[key]
        sign = val[0]
        if sign == '-':
            if stab[0] == '+':
                stab = '-' + stab[1:]
            elif stab[0] == '-':
                stab = '+' + stab[1:]
        # update stabilizer
        stab = stab[:j0] + val[1] + stab[j0+1:j1] + val[2] + stab[j1+1:]
    
    return stab



# copy of invert_stabilizers function, for testing
def invert_stabilizers2(stabilizers):
    
    # auxilliary function
    def min_weight_stab(stabilizers, reduced_stabs):
        
        n = len(stabilizers)
        min_weight = n
        unreduced_stabs = list(set(range(n))-set(reduced_stabs))
        min_i = unreduced_stabs[0]
        for i in unreduced_stabs:
            weight = n - stabilizers[i].count('I')
            if weight < min_weight:
                min_weight = weight
                min_i = i
        
        return min_i
    
    # auxilliary function
    def reduce_stab(i, reduced_qubits, stabilizers):
        # find qubit index that will be converted to Z
        n = len(stabilizers)
        stab = stabilizers[i]
        for q in range(n):
            if q not in reduced_qubits and stab[n-q] != 'I':
                break

        # map other qubits to 'Z' and apply U_ZZ gate
        for q2 in range(n):
            if q2 != q and stab[n-q2] != 'I':
                # map q to 'X' if 'Z'
                if stab[n-q] == 'Z':
                    C = [('+X', '+Y', '+Z') for i in range(n)]
                    C[q] = ('+Z', '-Y', '+X')
                    stabilizers = [update_stab_SQ(C, st) for st in stabilizers]
                    stab = stabilizers[i]
                # map q2 to 'Z'
                if stab[n-q2] == 'X':
                    C = [('+X', '+Y', '+Z') for i in range(n)]
                    C[q2] = ('+Z', '-Y', '+X')
                    stabilizers = [update_stab_SQ(C, st) for st in stabilizers]
                    stab = stabilizers[i]
                if stab[n-q2] == 'Y':
                    C = [('+X', '+Y', '+Z') for i in range(n)]
                    C[q2] = ('-X', '+Z', '+Y')
                    stabilizers = [update_stab_SQ(C, st) for st in stabilizers]
                    stab = stabilizers[i]
                stabilizers = [update_stab_ZZ(st, [list(np.sort([q,q2]))]) for st in stabilizers]
                stab = stabilizers[i]
        reduced_qubits.append(q)
        
        # map qubit q to Z
        if stab[n-q] == 'X':
            C = [('+X', '+Y', '+Z') for i in range(n)]
            C[q] = ('+Z', '-Y', '+X')
            stabilizers = [update_stab_SQ(C, st) for st in stabilizers]
        if stab[n-q] == 'Y':
            C = [('+X', '+Y', '+Z') for i in range(n)]
            C[q] = ('-X', '+Z', '+Y')
            stabilizers = [update_stab_SQ(C, st) for st in stabilizers]
        
        return stabilizers, reduced_qubits
                
    # main body of function
    n = len(stabilizers)
    
    reduced_stabs = []
    reduced_qubits = []
    while len(reduced_stabs) < n:
        # pick unreduced stabilizer with minumum weight
        i = min_weight_stab(stabilizers, reduced_stabs)
        # reduce stabilizer
        stabilizers, reduced_qubits = reduce_stab(i, reduced_qubits, stabilizers)
        reduced_stabs.append(i)
    
    return stabilizers



