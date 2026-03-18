import numpy as np
from .basis import Basis
from .param import TOL_RED_COST, TOL_PIVOT, TOL_RATIO_PRIMAL, TOL_FEAS

def primal_simplex(self, base0: Basis, pertub = 1e-9, it_max = 20000, verbosity=-1):
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
            candidates = np.where(r < -TOL_RED_COST)[0]
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
                argmin_ratios = np.where(np.abs(ratios-α) < TOL_RATIO_PRIMAL)[0]
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
            if stalling and len(z_history) > 1 and abs(z_history[-1] - z_history[-2]) > TOL_RED_COST:
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
    