import numpy as np
from .basis import Basis
from .param import *

class SLP_Model:
    """Represents a Linear Programming problem in its standard formluation (SLP).
    Attributes:
        A (np.ndarray): Contraints matrix.
        b (np.ndarray): 2nd member of the constraints.
        c (np.ndarray): Vector of the objective function.
        n (int): Number of variables.
        m (int): Number of constraints.
        offset (float): objective value offset that araise when converting a general linear problem into a standard (SLP) formulation.
    """
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    n: int
    m: int
    offset: float

    def __init__(self):
        """Instanciates an empty SLP model."""
        self.A = np.array((0,0),dtype='d')
        self.b = np.array([],dtype='d')
        self.c = np.array([],dtype='d')
        self.n = -1
        self.m = -1
        self.offset = 0.

    def modelPhaseI(self):
        """Creates a SLP model for the phase I / initialization of the primal simplex from the current (self) SLP model.
        Returns:
            (SLP_Model): Phase I SLP problem for the self SLP model.
            (SolutionBase): Feasible basis for the phase I SLP problem.
        """
        n = len(self.c)
        m = len(self.b)
        D = np.zeros((m,m),dtype='d')
        for i in range(len(self.b)):
            D[i,i] = 1. if self.b[i] >= 0. else -1.

        # Build phase I SLP model
        slp_I = SLP_Model()
        slp_I.A = np.hstack([self.A,D])
        slp_I.b = self.b
        slp_I.c = np.concatenate([np.zeros((n),dtype='d'),np.ones((m),dtype='d')])
        slp_I.n = n+m
        slp_I.m = m
        slp_I.offset = self.offset

        # Buid initial feasible basis
        baseI = Basis(n+m,m)
        baseI.x = np.concatenate([np.zeros((n),dtype='d'),np.abs(self.b)])
        baseI.B = np.arange(n, n+m, dtype=int)
        baseI.N = np.arange(n, dtype=int)
        baseI.invA_B = D.copy()
        return slp_I, baseI


    def primalSimplex(self, base0: Basis, it_max = 1000):
        """Solve the given SLP problem starting from base0 basis.
        Args:
            base0 (Basis): Feasible basis for the first iteration.
            it_max (int, optional): Maximum number of iterations. Degault: 1000.
        Returns:
            (Basis): Optimal basis.
        """
        base = base0 # only a reference
        base.invA_B = np.linalg.inv(self.A[:,base.B])
        it = 0
        while it < it_max:

            # Step 1: reducted cost
            y = (self.c[base.B].T @ base.invA_B).T
            r = self.c[base.N] - (self.A[:,base.N].T @ y)

            # Step 2: optimality check
            if np.all(r >= -DIGITAL_0):
                # print(f"--> End of simplex in {it} iterations, z = {base.x.dot(self.c)}, r = {r}")
                return base

            # Step 3: Descent direction (incomming variable j)
            j = np.min(np.where(r<0)[0])
            d = -base.invA_B @ self.A[:,base.N[j]]

            # Step 4: check if problem is bounded
            neg_mask = d < -DIGITAL_0
            if not np.any(neg_mask):
                raise ValueError(f"[primalSimplex]: problem is not bounded !")

            # Step 5: maximal step computation
            ratios = np.full_like(d, np.inf, dtype=float)
            ratios[neg_mask] = -base.x[base.B[neg_mask]] / d[neg_mask]
            α = np.min(ratios)

            # Step 6: outcomming variable l
            argmin_ratios = np.where(np.abs(ratios-α) < DIGITAL_0)[0]
            argmin_var_indices = [base.B[i] for i in argmin_ratios]
            l = argmin_ratios[np.argmin(argmin_var_indices)] # Bland rule (minimum variable index)

            # Step 7: update new basis
            base.x[base.B] += α*d
            base.x[base.N[j]] += α
            base.x[base.B[l]] = 0. # hard reset to avoid null imprecisions
            incomming_var = base.N[j]
            outcommig_var = base.B[l]
            base.B[l] = incomming_var
            base.N[j] = outcommig_var
            base.invA_B = np.linalg.inv(self.A[:,base.B])

            # Logs
            it += 1
            # print(f"\t - it[{it}]: z = {base.x.dot(self.c)}")

        print(f"--> [primalSimplex]: non-convegence after {it_max} iterations !")
        return base
    
    def __str__(self):
        slp_str = f"(SLP) formulation ({self.n},{self.m}):\n"
        slp_str += f" -> c = {self.c}\n"
        slp_str += f" -> offset = {self.offset}\n"
        slp_str += f" -> A = {self.A}\n"
        slp_str += f" -> b = {self.b}\n"
        return slp_str


