import numpy as np
from .basis import Basis
from .param import TOL_FEAS, TOL_RATIO_DUAL

def dual_simplex(self, base0: Basis, it_max=10000, verbosity=-1) -> Basis:
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
                break # basis is optimal
            
            # --- 2: Pivot choice (with Bland rule)
            x_min = np.min(base.x[base.B])
            candidates_in_B = np.where(abs(base.x[base.B]-x_min) < TOL_FEAS)[0]
            p = candidates_in_B[np.argmin(base.B[candidates_in_B])]

            # --- 3: Descent step α
            e_p = np.zeros((self.m), dtype='d')
            e_p[p] = 1.0
            y_p = base.lu_solver.solve(e_p, trans='T')
            α = y_p @ self.A[:, base.N] 

            # --- 4: Entry variable (with Bland rule)
            J = np.where(α <= -TOL_FEAS)[0]
            if len(J) == 0:
                raise RuntimeError("[dual_simplex]: problem is not feasible")
            
            ratio = np.full((self.n-self.m),-np.inf, dtype='d')
            ratio[J] = r[J]/α[J]
            Θ = np.max(ratio)
            candidates_in_N = np.where(np.abs(ratio-Θ) < TOL_RATIO_DUAL)[0]
            q = candidates_in_N[np.argmin(base.N[candidates_in_N])]


            # --- 5: Base update
            incomming_var = base.N[q]
            outcommig_var = base.B[p]
            base.B[p] = incomming_var
            base.N[q] = outcommig_var
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
        
        if it == it_max:
            raise Warning(f"[dual_simplex]: did not converge after {it} iterations.")
        
        return base
