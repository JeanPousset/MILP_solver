from .lp import LinearProblem
from .param import TOL_INT
from .basis import Basis
from .slp import SLP_Model
from ._branch_and_bound import b_and_d, branch
import numpy as np


class MILP_Problem(LinearProblem):
    """Represents a general Mixed Integer Linear Programming (MILP) problem.
    Attributes:
        int_vars (np.ndarray): boolean array that indicates whether the variable is discrete (True) or continuous (False).
    """
    int_vars: np.ndarray


    # outsider functions:
    branch = branch
    branch_and_bound = b_and_d


    def __init__(self):
        super().__init__()
        self.int_vars = np.array([], dtype=bool)

    def set_variable_bounds(self, x_l: np.ndarray, x_u: np.ndarray, int_flags: np.ndarray):
        super().set_variable_bounds(x_l, x_u)
        # assert np.all(x_u[int_flags] < np.inf), f"[set_variable_bounds]: integer variables must be upper-bounded."
        self.int_vars = int_flags

    def check_integrity(self, x: np.ndarray, slp: SLP_Model):
        """Returns true only if all variables that must be integers are (apart from TOL_INT) integers.
        Args: 
            x (np.ndarray): solution vectors to check.
        Returns
            (bool): whether the given solution x satisfies integrity constraints.
        """
        x_int_orig = x[:self.n][self.int_vars] * slp.col_scales[:self.n][self.int_vars] + self.x_l[self.int_vars]
        return np.all(np.abs(np.rint(x_int_orig)-x_int_orig) < TOL_INT)


    def solve_relaxation(self) -> tuple[Basis, SLP_Model]:
        opti_base, slp = super().solve()
        return opti_base, slp
    
    def solve(self, max_nb_nodes=100, verbosity=1):
        """Solve the MILP problem with the B&B algorithm.
        Args:
            max_nb_nodes (int, optional): Maximum number of Linear Problem to solve for safety. Default: 100.
            verbosity (int, optional): Verbosity level for logs. Default: -1.
        Returns:
            (np.ndarray): Optimal integer-feasible solution.
            (float): Optimal objective value. 
        """
        if verbosity >= 1:
            print(self)
        x_opti, z_opti = self.branch_and_bound(max_nb_nodes=max_nb_nodes, verbosity=verbosity)

        if verbosity >= 0:
            print(f"MILS successfully solved with B&B : \t • z = {z_opti}\n\t • x = {x_opti}")
        return x_opti, z_opti