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
        self.y = np.zeros((m),dtype='d')
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
        current_B = list(self.B)
        N_real_candidates = [j for j in range(n_orig) if j not in current_B]
        rows_with_art = [i for i, var_idx in enumerate(current_B) if var_idx >= n_orig]
        rows_to_remove = []

        for i in rows_with_art:
            found_pivot = False
            for j_cand in N_real_candidates:
                try:
                    column_j = A_phaseI[:, j_cand].toarray().ravel()
                    d_col = self.lu_solver.solve(column_j)
                    # Test whether the new base is OK
                    if abs(d_col[i]) > TOL_PIVOT_II:
                        current_B[i] = j_cand
                        N_real_candidates.remove(j_cand)
                        found_pivot = True
                        self.B = np.array(current_B)
                        self.update_lu(A_phaseI)
                        break
                except:
                    continue
            if not found_pivot:
                # it means that row i is redundant.
                print(f"Redundancy detected at row {i}")
                rows_to_remove.append(i)

        # builds smaller SLP model in case of redundancies
        if rows_to_remove:
            valid_rows = [r for r in range(len(current_B)) if r not in rows_to_remove]
            slp.A = slp.A[valid_rows, :]
            slp.b = slp.b[valid_rows]
            slp.m = len(valid_rows)
            current_B = [current_B[r] for r in valid_rows]
        
        # final baseII build
        baseII = Basis(slp.n, slp.m)
        baseII.B = np.array(current_B, dtype=int)
        baseII.N = np.array([j for j in range(slp.n) if j not in baseII.B], dtype=int)

        baseII.update_lu(slp.A)
        baseII.x = np.zeros(slp.n)
        baseII.x[baseII.B] = baseII.lu_solver.solve(slp.b)
       
        return baseII


    def __str__(self):
        """Gives strings to display when calling `print(:Basis)`."""
        res = f"Basis({self.n},{self.m})\n"
        res += f"in-basis: {self.B},\t out-basis: {self.N}\n"
        res += f"x = {self.x}"
        return res
