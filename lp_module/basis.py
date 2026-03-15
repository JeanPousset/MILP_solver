import numpy as np
from .param import *
from scipy.sparse.linalg import splu
import scipy.sparse.linalg as spla

class Basis:
    """Stores data of a simplex basis."""
    n:     int
    m:     int
    B:     np.ndarray
    N:     np.ndarray
    x:     np.ndarray
    lu_solver: spla.SuperLU

    def __init__(self, n: int, m: int):
        """Instanciates an empty simplex basis of size (n,m).
        Args:
            n (int): simplex dimension (number of variables).
            m (int): basis size (number of constraints).
        """
        assert n >= m, f"[Basis]: the number of constraints (m={m}) must not exceed the number of constraints (here n={n})"
        self.n = n
        self.m = m
        self.B = np.arange(m, dtype=int)
        self.N = np.arange(m, n, dtype=int)
        self.x = np.zeros((n),dtype='d')
        self.lu_solver = None

    def update_lu(self, A_matrix):
        """Updates LU factorization from the constraint matrix A of the problem.
        Args:
            A_matrix (sparse.csc_matrix): Constraint matrix of the SLP formlulation of the problem.
        """
        self.lu_solver = splu(A_matrix[:, self.B].tocsc())

    def extract_baseII(self, slp, A_phaseI):
        """Extracts a feasible basis (baseII) from the optimal basis resulting of phase I of the simplex initialization.
        Args:
            slp (SLP_Model): Standard formulation of the inital problem.
            A_phaseI (np.ndarray): Constraint matrix of the phase I SLP formulation.
        Returns:
            (Basis): a feasible basis for the initial SLP.
        """

        n_orig = slp.n
        m_orig = slp.m
        # Check that initialization variables z are nul
        if any(abs(self.x[n_orig:]) > TOL_FEAS):
            raise ValueError(f"[solve]: problem is not feasible !")
        
        # Remove initialization variables z from baseI
        B_list = list(self.B)
        N_list = list(self.N)
        N_real = [j for j in N_list if j < n_orig] # variables in N that belongs to the initial problem
        for i, var_index in enumerate(B_list):
            if var_index >= n_orig: # i.e. artificial variable is inside the basis
                found_pivot = False
                for j_idx, var_N in enumerate(N_real):
                    # check if pivot is digitaly feasible 
                    d_col = self.lu_solver.solve(A_phaseI[:, var_N].toarray())
                    if abs(d_col[i]) > TOL_PIVOT:
                        B_list[i] = var_N
                        N_real.pop(j_idx)
                        found_pivot = True
                        self.update_lu(A_phaseI)
                        break
                if not found_pivot:
                    print(f"[extract_baseII]: Warning : pivot not found for switching initial variable that is still in basis.")

        # builds new basis    
        baseII = Basis(n_orig, m_orig)
        baseII.B = np.array(B_list, dtype=int)
        baseII.N = np.setdiff1d(range(n_orig),baseII.B)
        baseII.update_lu(slp.A)
        baseII.x = np.zeros(n_orig)
        baseII.x[baseII.B] = baseII.lu_solver.solve(slp.b)
        return baseII

    def __str__(self):
        """Gives strings to display when calling `print(:Basis)`."""
        res = f"Basis({self.n},{self.m})\n"
        res += f"in-basis: {self.B},\t out-basis: {self.N}\n"
        res += f"x = {self.x}"
        return res
