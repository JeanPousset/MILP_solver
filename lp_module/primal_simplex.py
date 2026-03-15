import numpy as np
from .basis import Basis
from .param import *
from scipy import sparse
from scipy.sparse.linalg import splu

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
    A: sparse.csc_matrix
    b: np.ndarray
    c: np.ndarray
    n: int
    m: int
    offset: float

    def __init__(self):
        """Instanciates an empty SLP model."""
        self.A = sparse.csc_matrix((0,0),dtype='d')
        self.b = np.array([],dtype='d')
        self.c = np.array([],dtype='d')
        self.n = -1
        self.m = -1
        self.offset = 0.

    def scale_model(self):
        """Scales the constraint matrix and second members b."""
        # lists maximum coefficient of each constraint
        row_maxes = np.array(self.A.max(axis=1).toarray()).flatten()        
        row_maxes[row_maxes == 0] = 1.0 # avoids divisions by 0
        S_inv = sparse.diags(1.0 / row_maxes)  # scale diagonal matrix
        self.A = S_inv @ self.A
        self.b = S_inv @ self.b

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
        slp_I.A = sparse.hstack([self.A,D], format='csc')
        slp_I.b = self.b
        slp_I.c = np.concatenate([np.zeros((n),dtype='d'),np.ones((m),dtype='d')])
        slp_I.n = n+m
        slp_I.m = m
        slp_I.offset = self.offset
        slp_I.scale_model()

        # Buid initial feasible basis
        baseI = Basis(n+m,m)
        baseI.x = np.concatenate([np.zeros((n),dtype='d'),np.abs(self.b)])
        baseI.B = np.arange(n, n+m, dtype=int)
        baseI.N = np.arange(n, dtype=int)
        baseI.lu_solver = splu(D)
        return slp_I, baseI


    def primalSimplex(self, base0: Basis, it_max = 20000, verbosity=-1):
        """Solve the given SLP problem starting from base0 basis.
        Args:
            base0 (Basis): Feasible basis for the first iteration.
            it_max (int, optional): Maximum number of iterations. Default: 1000.
            vebosity (int, optional): Verbosity level for logs. Default: -1.
        Returns:
            (Basis): Optimal basis.
        """
        base = base0 # only a reference
        base.update_lu(self.A)

        it = 0
        while it < it_max:

            # Step 1: reducted cost
            y = base.lu_solver.solve(self.c[base.B], trans='T')
            r = self.c[base.N] - (self.A[:,base.N].T @ y)

            # Step 2: optimality check
            candidates = np.where(r < -TOL_REL_COST)[0]
            if len(candidates) == 0:
                # print(f"--> End of simplex in {it} iterations, z = {base.x.dot(self.c)}, r = {r}")
                return base
            
            # --- Looking for a stable pivot
            found_stable_pivot = False
            sorted_candidate_indices = np.argsort(base.N[candidates]) # Bland rule
            unbounded_count = 0
            for idx in sorted_candidate_indices:
                j_rel = candidates[idx]

                # Step 3: Descent direction (incomming variable j)
                column_j = self.A[:, base.N[j_rel]].toarray().flatten()
                d_try = base.lu_solver.solve(-column_j)

                # Step 4: check if problem is bounded
                neg_mask = d_try < -TOL_PIVOT
                if not np.any(neg_mask):
                    unbounded_count += 1
                    continue # we try the next candidate
                    # raise ValueError(f"[primalSimplex]: problem is not bounded !")

                # Step 5: maximal step computation
                ratios = np.full_like(d_try, np.inf, dtype=float)
                ratios[neg_mask] = -base.x[base.B[neg_mask]] / d_try[neg_mask]
                α = np.min(ratios)

                # Step 6: outcomming variable l
                argmin_ratios = np.where(np.abs(ratios-α) < TOL_PIVOT)[0]
                argmin_var_indices = [base.B[i] for i in argmin_ratios]
                l_try = argmin_ratios[np.argmin(argmin_var_indices)] # Bland rule (minimum variable index)

                # --- Critical test : whether pivot is big enough ?

                # 1st defense: pivot magnitude
                if abs(d_try[l_try]) < TOL_PIVOT:
                    continue

                # 2nd defense: factorization test before committing
                B_next = base.B.copy()
                B_next[l_try] = base.N[j_rel]

                try:
                    test_lu = splu(self.A[:, B_next].tocsc())
                    # if we reach the following, pivot is stable
                    j, l, d = j_rel, l_try, d_try
                    found_stable_pivot = True
                    break
                except RuntimeError:
                    if verbosity >= 0:
                        print(f"Factorization failed for variable {base.N[j_rel]}, trying next candidate...")
                        continue # Try next candidate

            if not found_stable_pivot:
                if unbounded_count == len(candidates):
                    raise ValueError("[primalSimplex]: Problem is unbounded")
                else:
                    raise RuntimeError("[primalSimplex]: No stable pivot found (Degeneracy)")

            # Step 7: update new basis
            incomming_var = base.N[j]
            outcommig_var = base.B[l]
            base.B[l] = incomming_var
            base.N[j] = outcommig_var
            base.update_lu(self.A)
            base.x = np.zeros(self.n) 
            base.x[base.B] = base.lu_solver.solve(self.b)
            base.x[base.x < TOL_FEAS] = 0.0 # corrections

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


