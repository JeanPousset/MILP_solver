from .basis import Basis
from .slp import SLP_Model
from collections import deque
from scipy import sparse
import numpy as np
import sys
import math

def delete_last_lines(n=1):
    """Removes the n last lines in the terminal"""
    for _ in range(n):
        sys.stdout.write('\033[F\033[K')


def branch(self, base: Basis, slp: SLP_Model) -> list[tuple[Basis,SLP_Model]]:
    """Performs the branch operation of the B&B algorithms over a restricted version of the relaxed initial SLP.
    Args:
        base (Basis): Optimal basis found for the SLP problem.
        slp (SLP_Model): one of the SLP sub-problem currently handled by the B&B algorithm.
    Returns:
        (list[tuple[Basis,SLP_Model]]): Dual basis and SLP models for the two new sub-problem created (lower and upper restrictions).
    """
    # finds the most fractionnal variables x_f
    x_int_orig = base.x[:self.n][self.int_vars] * slp.col_scales[:self.n][self.int_vars] + self.x_l[self.int_vars]
    f_idx = np.argmax(np.abs(x_int_orig - np.round(x_int_orig)))
    f = np.arange(self.n)[self.int_vars][f_idx]
    # buils restrictions constraints (x_f >= threshold + 1, -x_f >= -threshold)
    threshold = math.floor(x_int_orig[f])
    constraint = np.zeros((slp.n),dtype='d')
    constraint[f] = 1.0
    sp_consrtaint = sparse.csc_matrix(constraint, dtype='d')
    base_low, slp_low = slp.restraint(base, -sp_consrtaint, (-threshold) / slp.col_scales[f])
    base_up, slp_up = slp.restraint(base, sp_consrtaint, (threshold+1) / slp.col_scales[f])
    return [base_low, slp_low], [base_up, slp_up]


def b_and_d(self, max_nb_nodes=100, verbosity=-1) -> Basis:
    """Branch & Bound algorithm to solve MILP problems.
    Args:
        max_nb_nodes (int, optional): Maximum number of Linear Problem to solve for safety. Default: 100.
        verbosity (int, optional): Verbosity level for logs. Default: -1.
    Returns:
        (np.ndarray): Optimal integer-feasible solution.
        (float): Optimal objective value. 
    """

    # --- Initialisation ---

    sign = -1.0 if self.flag_max else 1.0   # used in case of maximisation
    z_up = np.inf
    x_up = None
    base_relaxed, slp_relaxed = self.solve_relaxation()
    x0, z0 = self.getResult(base_relaxed,slp_relaxed.col_scales)
    nb_nodes = 1
    nb_prompt = 0
    # Check if relaxed solution is integer
    if self.check_integrity(x0, slp_relaxed):
        z_up = sign * z0
        x_up = x0.copy()
    
    [node_inf, node_sup] = self.branch(base_relaxed, slp_relaxed)
    Q = deque()
    Q.append(node_sup)
    Q.append(node_inf)

    # --- Walk throught restriction tree
    while len(Q)>0 and nb_nodes < max_nb_nodes:

        # logs
        if verbosity < 1:
            delete_last_lines(nb_prompt)
            nb_prompt = 0
        if verbosity >= 0:
            print(f"[B&B]: node {nb_nodes}, z_+ = {z_up}, |Q| = {len(Q)}")
            nb_prompt += 1

        [base_k, slp_k] = Q.pop()
        nb_nodes += 1

        # --- Solves current node
        try:
            base_k = slp_k.dual_simplex(base_k)
        except RuntimeError:
            # Problem is not feasible -> node pruned
            continue
        # retrieves real solution without scaling and offset
        xk, zk = self.getResult(base_k, slp_k.col_scales)
        sign_zk = sign * zk


        # --- Checks if current node can lead to a better integer-feasible solution
        if sign_zk >= z_up:
            # node pruned [its descendants can not gives a better solution than the one that gaved z_up]
            if verbosity >= 0:
                print(f"\t bad objetive value -> pruned")
                nb_prompt += 1
            continue 

        # --- Checks if relaxed solution satisfies integrity constraints
        if self.check_integrity(xk, slp_k):
            if sign_zk < z_up: 
                # better integer solution found
                z_up = sign_zk
                x_up = xk.copy()
                if verbosity >= 0:
                    print(f"\t better objective value found: z_+ = {z_up*sign}")
                    nb_prompt += 1
                continue
        
        # --- Otherwise : branch to create two new subproblems
        [node_inf, node_sup] = self.branch(base_k, slp_k)
        Q.append(node_sup)
        Q.append(node_inf)

    # clean last logs
    if verbosity < 1:
            delete_last_lines(nb_prompt)
            nb_prompt = 0

    if nb_nodes == max_nb_nodes:
        raise Warning("[B&B]: maximum number of nodes reached, solution is not optimal !")

    return x_up, z_up * sign
