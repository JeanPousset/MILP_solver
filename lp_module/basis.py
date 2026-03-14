import numpy as np
from .param import *

class Basis:
    """Stores data of a simplex basis."""
    n:     int
    m:     int
    B:     np.ndarray
    N:     np.ndarray
    invA_B: np.ndarray
    x:     np.ndarray

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
        self.invA_B = np.zeros((m,m),dtype='d')
        self.x = np.zeros((n),dtype='d')

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
        if any(abs(self.x[n_orig:]) > DIGITAL_0):
            raise ValueError(f"[solve]: problem is not feasible !")
        
        # Remove initialization variables z from baseI
        B_list = list(self.B)
        N_list = list(self.N)
        N_real = [j for j in N_list if j < n_orig] # variables in N that belongs to the initial problem
        for i, var_index in enumerate(B_list):
            if var_index >= n_orig: # i.e. artificial variable is inside the basis
                print("Found residual initialization variable in B")
                found_pivot = False
                for j_idx, var_N in enumerate(N_real):
                    # check if pivot is digitaly feasible 
                    d_col = self.invA_B @ A_phaseI[:, var_N]
                    if abs(d_col[i]) > DIGITAL_0:
                        B_list[i] = var_N
                        N_real.pop(j_idx)
                        found_pivot = True
                        self.invA_B = np.linalg.inv(A_phaseI[:, B_list])
                        break
                if not found_pivot:
                    print(f"[extract_baseII]: Warning : pivot not found for switching initial variable that is still in basis.")

        # builds new basis    
        baseII = Basis(n_orig, m_orig)
        baseII.B = np.array(B_list, dtype=int)
        baseII.N = np.setdiff1d(range(n_orig),baseII.B)
        baseII.invA_B = np.linalg.inv(slp.A[:, baseII.B])
        baseII.x = np.zeros(n_orig)
        baseII.x[baseII.B] = baseII.invA_B @ slp.b
        return baseII

    def __str__(self):
        """Gives strings to display when calling `print(:Basis)`."""
        res = f"Basis({self.n},{self.m})\n"
        res += f"in-basis: {self.B},\t out-basis: {self.N}\n"
        res += f"x = {self.x}"
        return res
