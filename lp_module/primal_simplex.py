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
        col_scales (np.ndarray): column scales.

    """
    A: sparse.csc_matrix
    b: np.ndarray
    c: np.ndarray
    n: int
    m: int
    offset: float
    col_scales: np.ndarray

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
        slp_I.scale_model()

        # Buid initial feasible basis
        baseI = Basis(n+m,m)
        baseI.x = np.concatenate([np.zeros((n),dtype='d'),np.abs(self.b)])
        baseI.B = np.arange(n, n+m, dtype=int)
        baseI.N = np.arange(n, dtype=int)
        baseI.lu_solver = splu(D)
        return slp_I, baseI


    def primalSimplex(self, base0: Basis, pertub = 1e-9, it_max = 20000, verbosity=-1):
        """Solve the given SLP problem starting from base0 basis.
        Args:
            base0 (Basis): Feasible basis for the first iteration.
            perturb (float): perturbation factor to apply on 2nd member.
            it_max (int, optional): Maximum number of iterations. Default: 1000.
            vebosity (int, optional): Verbosity level for logs. Default: -1.
        Returns:
            (Basis): Optimal basis.
        """
        base = base0 # only a reference
        z_history = []
        max_history = 15 # we check the last max_history values
        stalling = False
        it = 0

        ### --- 0: Pertubations
        b_orig = self.b.copy() # Sauvegarde
        perturbation = np.array([pertub * (1.01**i) for i in range(self.m)]) # unique noise per row
        self.b += perturbation

        base.update_lu(self.A)
        base.x = np.zeros(self.n, dtype='d')
        base.x[base.B] = base.lu_solver.solve(self.b)


        while it < it_max:
            
            z = base.x[base.B].dot(self.c[base.B])

            # Step 1: reducted cost
            base.y = base.lu_solver.solve(self.c[base.B], trans='T') # dual vector
            r = self.c[base.N] - (self.A[:,base.N].T @ base.y)

            # Step 2: optimality check
            candidates = np.where(r < -TOL_REL_COST)[0]
            if len(candidates) == 0:
                # print(f"--> End of simplex in {it} iterations, z = {base.x.dot(self.c)}, r = {r}")
                break
            
            # --- Looking for a stable pivot
            found_stable_pivot = False
            unbounded_count = 0

            # special pivot choice in case of cycling
            if stalling:
                base.update_lu(self.A) 
                stalling = False
     

            indices_to_test = np.argsort(r[candidates]) # not bland rule

            for idx in indices_to_test:
                j_rel = candidates[idx]

                # Step 3: Descent direction (incomming variable j)
                column_j = np.asarray(self.A[:, base.N[j_rel]].todense()).reshape(-1)
                d_try = base.lu_solver.solve(-column_j)

                # Step 4: check if problem is bounded
                neg_mask = d_try < -TOL_PIVOT
                if not np.any(neg_mask):
                    unbounded_count += 1
                    continue # we try the next candidate
                    # raise ValueError(f"[primalSimplex]: problem is not bounded !")

                # Step 5: maximal step computation
                ratios = np.full_like(d_try, np.inf, dtype='d')
                ratios[neg_mask] = -base.x[base.B[neg_mask]] / d_try[neg_mask]
                α = np.min(ratios)

                # Step 6: outcomming variable l
                argmin_ratios = np.where(np.abs(ratios-α) < TOL_RATIO)[0]
                argmin_var_indices = [base.B[i] for i in argmin_ratios]
                l_try = argmin_ratios[np.argmin(argmin_var_indices)] # Bland rule (minimum variable index)

                # --- Critical test : whether pivot is big enough ?

                # 1st defense: pivot magnitude
                if abs(d_try[l_try]) < TOL_PIVOT:
                    continue

                # Step 7: update new basis
                incomming_var = base.N[j_rel]
                outcommig_var = base.B[l_try]
                old_B = base.B.copy()
                old_N = base.N.copy()
                base.B[l_try] = incomming_var
                base.N[j_rel] = outcommig_var

                try:
                    base.update_lu(self.A)
                    x_test = base.lu_solver.solve(self.b)
                    denom = max(1.0, np.linalg.norm(self.b, ord=np.inf))
                    residual = np.linalg.norm(self.A[:,base.B] @ x_test - self.b, ord=np.inf) / denom
                    if residual > 1e-7:
                        raise RuntimeError("Instable basis: residual is too high")
                    # if everything is ok, we keep going
                    base.x = np.zeros(self.n) 
                    base.x[base.B] = x_test
                    base.x[np.abs(base.x) < TOL_FEAS] = 0.0 # digital zero corrections
                    found_stable_pivot = True
                    break
                
                except RuntimeError:
                    # cancel (choosen pivot is a disaster)
                    base.B = old_B
                    base.N = old_N
                    base.update_lu(self.A)
                    # removes this candidates
                    found_stable_pivot = False
                    if verbosity >= 0:
                        print(f"[Warning] Pivot {j_rel} rejected after calculus (x instability).", end="")
                    continue
            
            if not found_stable_pivot:
                if unbounded_count == len(candidates):
                    base.update_lu(self.A) # LU clean-up
    
                    j_best = candidates[np.argmin(r[candidates])]
                    column_j = np.asarray(self.A[:, base.N[j_best]].todense()).reshape(-1)
                    d_check = base.lu_solver.solve(-column_j)
                    
                    if not np.any(d_check < -1e-15): 
                        raise ValueError("[primalSimplex]: Problem is mathematically unbounded")
                    else:
                        continue
                # --- Last chance for stability ---
                if not stalling: 
                    if verbosity >= 0:
                        print("\n[Warning] No stable pivot. Refactorizing and retrying...")
                    base.update_lu(self.A)
                    stalling = True # We try a last stalling iteration
                    continue 
                else:
                    raise RuntimeError("[primalSimplex]: Critical numerical failure (Singular Basis)")


            # Step 8: cycling check
            z = np.dot(base.x[base.B],self.c[base.B])
            z_history.append(round(z,9))
            if len(z_history) > max_history:
                 z_history.pop(0)
            stalling = len(z_history) == max_history and z_history.count(z_history[-1]) > 2
            if stalling and len(z_history) > 1 and abs(z_history[-1] - z_history[-2]) > TOL_REL_COST:
                # reset stalling
                z_history = []
                stalling = False

            # Logs
            it += 1
            if verbosity >= 0:
                print(f"\rprimalSimex ({self.m},{self.n}) - it[{it}]: z = {z}",end="",flush=True)

        if it == it_max:
            print(f"--> [primalSimplex]: non-convegence after {it_max} iterations !")

        self.b = b_orig
        # --- Final clean
        try:
            base.update_lu(self.A)
            x_final = base.lu_solver.solve(self.b)
            base.x = np.zeros(self.n)
            base.x[base.B] = x_final
            base.x[np.abs(base.x) < TOL_FEAS] = 0.0
        except:
            print("[Warning] Fail to clean final vector x.")

        return base
    

    def dual_simplex(self, base0: Basis, it_max=10000, verbosity=-1):
        """Solve the given SLP problem with the simplex method starting from base0 dual basis.
        Args:
            base0 (Basis): A dual basis of SLP for initialization.
            it_max (int, optional): Maximum number of iterations. Default: 10000.
            vebosity (int, optional): Verbosity level for logs. Default: -1.
        Returns:
            (Basis): the primal-dual optimal basis
        """
        base = base0
        y = base.lu_solver.solve(self.c[base.B], trans='T')
        r = self.c[base.N] - (self.A[:,base.N].T @ y)
        it = 0
        while it < it_max:
            
            # --- 1: Convergence check
            if np.all(base.x[base.B] >= 0):
                return base
            
            # --- 2: Pivot choice (with Bland rule)
            x_min = np.min(base.x[base.B])
            candidates_in_B = np.where(abs(base.x[base.B]-x_min) < TOL_RATIO)
            p = np.min([base.B[candidates_in_B]])

            # --- 3: Descent step α
            e_p = np.zeros((self.m), dtype='d')
            e_p[p] = 1.0
            y = base.lu_solver.solve(e_p, trans='T')
            α = self.A[:, base.N].T @ y

            # --- 4: Entry variable (with Bland rule)
            J = np.where(α <= TOL_FEAS)[0]
            if len(J) == 0:
                raise RuntimeError("[dual_simplex]: problem is not bounded")
            Θ = np.max(r/α)
            ratio = np.full((self.n-self.m),-np.inf, dtype='d')
            ratio[J] = r[J]/α[J]
            candidates_in_N = np.where(np.abs(ratio-Θ) < TOL_RATIO)[0]
            q = np.min(base.N[candidates_in_N])

            # --- 5: Base update
            incomming_var = base.N[q]
            outcommig_var = base.B[p]
            base.B[q] = incomming_var
            base.N[p] = outcommig_var
            base.update_lu(self.A)
            base.x = np.zeros(self.n) 
            base.x[base.B] = base.lu_solver.solve(self.b)
            base.x[np.abs(base.x) < TOL_FEAS] = 0.0 # digital zero corrections
            base.y = base.lu_solver.solve(self.c[base.B], trans='T')
            r = self.c[base.N] - (self.A[:,base.N].T @ y)
            z = np.dot(base.x[base.B], self.c[base.B])

            # --- Logs
            it += 1
            if verbosity >= 0:
                print(f"\rdual_simplex ({self.m},{self.n}) - it[{it}]: z = {z}",end="",flush=True)

    
    def __str__(self):
        slp_str = f"(SLP) formulation ({self.n},{self.m}):\n"
        slp_str += f" -> c = {self.c}\n"
        slp_str += f" -> offset = {self.offset}\n"
        slp_str += f" -> A = {self.A}\n"
        slp_str += f" -> b = {self.b}\n"


