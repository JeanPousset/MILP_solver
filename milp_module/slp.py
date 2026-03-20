from __future__ import annotations
import numpy as np
from .basis import Basis
from scipy import sparse
from scipy.sparse.linalg import splu
from ._primal_simplex import primal_simplex
from ._dual_simplex import dual_simplex

class SLP_Model:
    """Represents a Linear Programming problem in its standard formluation (SLP).
    Attributes:
        A (np.ndarray): Contraints matrix.
        b (np.ndarray): 2nd member of the constraints.
        c (np.ndarray): Vector of the objective function.
        n (int): Number of variables.
        m (int): Number of constraints.
        offset (float): Objective value offset that araise when converting a general linear problem into a standard (SLP) formulation.
        col_scales (np.ndarray): Column scales.
    """
    A: sparse.csc_matrix
    b: np.ndarray
    c: np.ndarray
    n: int
    m: int
    offset: float
    col_scales: np.ndarray


    # --- Simplex methods ---
    primal_simplex = primal_simplex
    dual_simplex = dual_simplex

    def __init__(self):
        """Instanciates an empty SLP model."""
        self.A = sparse.csc_matrix((0,0),dtype='d')
        self.b = np.array([],dtype='d')
        self.c = np.array([],dtype='d')
        self.n = -1
        self.m = -1
        self.offset = 0.
        self.col_scales = 0.

    def scale_model(self):
        """Scales the constraint matrix and second members b."""
        # 1. Row Scaling
        row_maxes = np.abs(self.A).max(axis=1).toarray().flatten()
        row_maxes[row_maxes < 1e-10] = 1.0
        S_inv = sparse.diags(1.0 / row_maxes)
        self.A = S_inv @ self.A
        self.b = S_inv @ self.b

        # 2. Column Scaling
        col_maxes = np.abs(self.A).max(axis=0).toarray().flatten()
        col_maxes[col_maxes < 1e-10] = 1.0
        self.col_scales = col_maxes 
        C_inv = sparse.diags(1.0 / col_maxes)
        self.A = self.A @ C_inv
        self.c = self.c / col_maxes

    def modelPhaseI(self):
        """Creates a SLP model for the phase I / initialization of the primal simplex from the current (self) SLP model.
        Returns:
            (SLP_Model): Phase I SLP problem for the self SLP model.
            (SolutionBase): Feasible basis for the phase I SLP problem.
        """
        n = len(self.c)
        m = len(self.b)
        diag_values = [1.0 if self.b[i] >= 0 else -1.0 for i in range(m)]
        D = sparse.diags(diag_values, format='csc') # CSC conversion

        # Build phase I SLP model
        slp_I = SLP_Model()
        slp_I.A = sparse.hstack([self.A,D], format='csc', dtype='d')
        slp_I.b = self.b
        slp_I.c = np.concatenate([np.zeros((n),dtype='d'),np.ones((m),dtype='d')])
        slp_I.n = n+m
        slp_I.m = m
        slp_I.offset = self.offset
        # slp_I.scale_model() # avoid rescale

        # Buid initial feasible basis
        baseI = Basis(n+m,m)
        baseI.x = np.concatenate([np.zeros((n),dtype='d'),np.abs(self.b)])
        baseI.B = np.arange(n, n+m, dtype=int)
        baseI.N = np.arange(n, dtype=int)
        baseI.lu_solver = splu(D)
        return slp_I, baseI
    

    def restrain(self, base: Basis, a: sparse.csc_matrix, b: float) -> tuple[Basis, SLP_Model]:
        """Restraints the problem with the given constraint a*x >= b.
        Args:
            base (Basis): The optimal basis assiociated to the initial (self) SLP problem.
            a (sparse.csc_matrix): Constraint coefficients that are stored in a flat horizontal CSC matrix (row vector).
            b (float): Constraint lower bound.
        return:
            (Basis): dual basis for the restrained SLP problem.
            (SLP_Model): restrained SLP problem.
        """
        # SLP restriction
        slp_r = SLP_Model()
        slp_r.n = self.n + 1
        slp_r.m = self.m + 1
        slp_r.c = np.append(self.c,0.)
        slp_r.offset = self.offset
        slp_r.col_scales = np.append(self.col_scales,1.0) # we don't update column scalling yet
        
        # scale constraints
        new_a = sparse.hstack([a,sparse.csc_matrix([-1.0])], format='csc', dtype='d')
        scale = np.abs(new_a).max()
        A_col = sparse.hstack([self.A,np.zeros((self.m,1), dtype='d')], format='csc', dtype='d')
        slp_r.A = sparse.vstack([A_col,new_a/scale], format='csc', dtype='d')
        slp_r.b = np.append(self.b,b/scale) 
        
        # formulation of a dual basis
        base_r = Basis(base.n+1,base.m+1)
        base_r.B = np.append(base.B,self.n) # B' = B u {n}
        base_r.N = base.N.copy() # N' = N
        base_r.update_lu(slp_r.A)
        base_r.x = np.zeros(base_r.n)
        base_r.x[base_r.B] = base_r.lu_solver.solve(slp_r.b)
        base_r.y = np.append(base.y, 0.) # <=> base.lu_solver.solve(self.c[base.B], trans='T')


        return base_r, slp_r


    def __str__(self):
        slp_str = f"(SLP) formulation ({self.n},{self.m}):\n"
        slp_str += f" -> c = {self.c}\n"
        slp_str += f" -> offset = {self.offset}\n"
        slp_str += f" -> A = {np.asarray(self.A)}\n"
        slp_str += f" -> b = {self.b}\n"
        return slp_str
