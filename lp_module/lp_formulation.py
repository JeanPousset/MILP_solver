# %% Classes
import numpy as np
from .primal_simplex import SLP_Model
from .param import *
from .basis import Basis
from scipy import sparse
import highspy  # for the MPS reader


class Constraint:
    """Defines a linear constraint of the form b_l <= a*x <= b_u, and the symbol to print it

    Attributes:
        a (np.ndarray): Constraint coefficients (for the linear combinaison of x).
        symbol (str):   Operator that must be either an equality `==` or an inequality `<=` / `>=`, only used for displays/logs.
        b_l (float):    2nd member lower bound for the constraint.
        b_u (float):    2nd member upper bound for the constraint.
    """
    a:         np.ndarray
    symbol:    str
    b_l:       float
    b_u:       float

    def __init__(self, a: np.ndarray, symbol: str, b_l=-np.inf, b_u=np.inf):
        """ Initializes an instance of Constraint.

        Args:
            a (np.ndarray): Vector of coefficients constraints.
            symbol (str):   Operator that must be either an equality `==` or an inequality `<=` / `>=`.
            b_l (float, optional):    2nd member lower bound for the constraint. Default: -np.inf
            b_u (float, optional):    2nd member upper bound for the constraint. Default: np.inf
        """
        assert (b_l != -np.inf or b_u != np.inf), f"[Constraint]: constraint expression must be at least upper or lower bounded (currently given : -inf <= a*x <= inf)."
        assert b_l <= b_u, f"[Constraint]: given lower bound ({b_l}) is greater than given upper bound ({b_u})."
        assert any(symbol == s for s in ["<=",">=","=="]), f"[Constraint]: `symbol` argument must be either '<=' or '>=' (given : '{symbol}')"
        if symbol == "==":
            assert b_l == b_u, f"[Constraint]: symbol equal is given but lower bound ({b_l}) varies from upper bound ({b_u})."
        self.a = a
        self.symbol = symbol
        self.b_l = b_l
        self.b_u = b_u

    def __str__(self):
        """Returns string that displays the full constraint."""
        cstr_str = " ⦿ "

        if self.symbol == "==":
            cstr_str += "     "
        elif self.symbol == "<=":
            cstr_str += f"{self.b_l} {self.symbol} "
        else:
            cstr_str += f"{self.b_u} {self.symbol} "

        for i in range(len(self.a)-1):
            cstr_str += f" {self.a[i]}•X_{i} +"
        cstr_str += f" {self.a[-1]}•X_{len(self.a)-1}"

        if self.symbol == "==":
            cstr_str += f" {self.symbol} {self.b_l}"
        elif self.symbol == "<=":
            cstr_str += f" {self.symbol} {self.b_u}"
        else:
            cstr_str += f" {self.symbol} {self.b_l}"
        return cstr_str


class LinearProblem:
    """Represents a general Linear Programming (LP).

    Attributes:
        n (int): Number of variables (problem dimension).
        m (int): Number of constraints.
        constraints (List[Constraints]): List of linear constraints.
        x_l (np.ndarray): Variable lower bounds.
        x_l (np.ndarray): Variable upper bounds.
        c (np.ndarray): Vector of the objective function.
        flag_max (bool): Boolean value that is true if the given objective is a 'Max' instead of a 'Min'.
    """
    n: int
    m: int
    constraints: list[Constraint]
    x_l: np.ndarray
    x_u: np.ndarray
    c: np.ndarray
    flag_max: bool

    def __init__(self):
        """Instanciates an empty LP problem."""
        self.n = -1
        self.m = -1
        self.constraints = list()
        self.flag_max = False
        self.c = np.array([],dtype='d')

    def set_objective(self, extremum: str, c: np.ndarray):
        """Set the problem objective.
        Args:
            extremum (str): Character string to indicate wheter it is a 'Min' or 'Max' objective.
            c (np.ndarray): Vector of the objective function.
        Raises:
            ValueError: If a string different that 'Min' or 'Max' is given for parameter `extremum`.
        """
        assert (self.m==-1 or self.n == len(c)), f"[set_objective]: can not set new objective if one already exists with different dimension ({self.n} != {len(c)})"
        self.c = c
        self.n = len(c)
        if extremum == "Min":
            self.flag_max = False
        elif extremum == "Max":
            self.flag_max = True
        else:
            raise ValueError(f"[set_objective]: `extremum` argument must be either 'Min' or 'Max' (given : '{extremum}') ")
        
    def set_variable_bounds(self, x_l: np.ndarray, x_u: np.ndarray):
        """Set the variables bounds such that x_l[i] <= x_i <= x_u[i], for i in [0,n-1].
        Args:
            x_l (np.ndarray): Variable lower bounds.
            x_u (np.ndarray): Variable upper bounds.
        """
        assert not np.any(x_l == -np.inf), f"[set_variables_bounds]: variable lower bound must be finite (can not be equal to -np.inf)."
        assert len(x_l) == len(x_u), f"[set_variables_bounds]: given bound vectors don't have the same lenght (|x_l|={len(x_l)} != |x_u| = {len(x_u)})"
        assert len(x_l) == len(self.c), f"[set_variables_bounds]: bound vectors length does not match the number of variable lenght (|x_u|=|x_l|={len(x_u)} != n={self.n})"
        self.x_l = x_l
        self.x_u = x_u


    def set_constraints(self, constraints: list[Constraint]):
        """Set the given constraints for the LP problem.
        Args:
            constraints (list[Constraints]): List of linear constraints.
        """
        self.m = 0
        for cstr in constraints:
            assert len(cstr.a) == self.n, f"[set_constraints]: constraint `a` length ({len(cstr.a)}) does not match number of variables ({self.n})"
            self.constraints.append(cstr)
            self.m += 1

    
    @classmethod
    def from_mps(cls, mps_file: str):
        """Load a MPS file an create the associated linear problem instance.
        Args:
            mps_file (str): MPS file path.
        Returns:
            (LinearProblem): the linear problem stored in the MPS file.
        """

        # --- loads mps file
        h = highspy.Highs();
        status = h.readModel(mps_file)
        if status != highspy.HighsStatus.kOk:
            raise ImportError(f"[from_mps]: Can't read MPS file : {mps_file}")
        highs_model = h.getLp()
        lp = cls()

        # --- objective 
        c = np.array(highs_model.col_cost_, dtype='d')
        lp.set_objective("Min", c)

        # --- variables bounds (extremum values are replaced by +/- np.inf)
        x_l = np.array(highs_model.col_lower_)
        x_u = np.array(highs_model.col_upper_)
        x_l = np.where(x_l < -1e20, -np.inf, x_l)   # in this case, an error will araise when calling `lp.set_variables_bound``
        x_u = np.where(x_u > 1e20, np.inf, x_u)
        lp.set_variable_bounds(x_l,x_u)

        # --- constraints bounds  (extremum values are replaced by +/- np.inf)
        b_l = np.array(highs_model.row_lower_)
        b_u = np.array(highs_model.row_upper_)
        b_l = np.where(b_l < -1e20, -np.inf, b_l)
        b_u = np.where(b_u > 1e20, np.inf, b_u)

        # --- reads constraints matrix from CSS (Compressed Column Storage) format
        num_vars = len(highs_model.col_cost_)
        num_cons = len(highs_model.row_lower_)
        A_dense = np.zeros((num_cons, num_vars))
        for i_col in range(num_vars):
            start = highs_model.a_matrix_.start_[i_col]
            end = highs_model.a_matrix_.start_[i_col + 1]
            for i_el in range(start, end):
                row = highs_model.a_matrix_.index_[i_el]
                val = highs_model.a_matrix_.value_[i_el]
                A_dense[row, i_col] = val

        # --- creates constraints
        constraints_list = []
        for i in range(num_cons):
            row_coeffs = A_dense[i, :]
            if abs(b_l[i] - b_u[i]) < 1e-10:
                constraints_list.append(Constraint(row_coeffs, "==", b_l[i], b_u[i]))
            else:
                constraints_list.append(Constraint(row_coeffs, "<=", b_l[i], b_u[i]))
        lp.set_constraints(constraints_list)
        
        return lp
  

    def to_SLP(self):
        """Converts a general linear programming problem into a standard linear programming (SLP) formulation."""
        
        # --- transforms constraints b_l <= Ax <= b_u into A'*x >= b'
        deviation_flag = [] # number of deviation variables to add when switching to the SLP formulation
        nb_deviation = 0
        A = []
        b = []
        for j, cstr in enumerate(self.constraints):
            if cstr.symbol == "==":
                A.append(cstr.a)
                b.append(cstr.b_l)
                deviation_flag.append(False)
            else:
                if cstr.b_l != -np.inf: # Ax >= b_l
                    A.append(cstr.a)
                    b.append(cstr.b_l)
                    deviation_flag.append(True)
                if cstr.b_u != np.inf: # -Ax >= -bu
                    A.append(-cstr.a)
                    b.append(-cstr.b_u)
                    deviation_flag.append(True)

        # --- shift constraints bounds to cancel variables lower bounds
        A_tmp = np.array(A)
        shift_b = A_tmp @ self.x_l
        for j in range(len(b)):
            b[j] -= shift_b[j]
        
        # --- adds variables bounds (x_l <= x <= x_u)
        for i in range(self.n):
            if self.x_u[i] != np.inf: # -x >= -x_u
                e_i = np.zeros((self.n), dtype='d')
                e_i[i] = 1.0
                A.append(-e_i)
                b.append(self.x_l[i]-self.x_u[i])
                deviation_flag.append(True)

        # --- concatenates contraint matrix and deviation matrix
        nb_constraints = len(deviation_flag)
        deviation_flag = np.array(deviation_flag, dtype=bool)
        nb_deviation = np.sum(deviation_flag)
        row_indices = np.where(deviation_flag)[0]
        col_indices = np.arange(nb_deviation)
        data = np.full(nb_deviation, -1.0)
        dev_matrix_sparse = sparse.csc_matrix((data, (row_indices, col_indices)), shape=(nb_constraints, nb_deviation))
        slp = SLP_Model()
        slp.A = sparse.hstack([sparse.csc_matrix(A),dev_matrix_sparse], format='csc')

        # --- dimensions & 2nd members
        slp.n = self.n + nb_deviation
        slp.m = nb_constraints
        slp.b = np.array(b)
        slp.scale_model()

        # --- objective function
        new_c = -self.c if self.flag_max else self.c
        slp.c = np.concatenate([new_c,np.zeros((nb_deviation),dtype='d')])
        slp.offset = np.dot(self.c, self.x_l)

        return slp


    def solve(self, verbosity=-1):
        """Solves the LP problem with the primal simplex method.
        Args:
            verbosity (int, optional): whether logs will be printed.
        Returns:
            (Basis): optimal basis found.
        """

        # Display
        if verbosity > 0:
            print("----------------------")
        if verbosity >= 0:
            print(f"Resolution of problem of size (n,m) = ({self.n},{self.m}):\n")
        if verbosity > 0:
            print(self)

        # Phase I
        slp = self.to_SLP()
        slpI, baseI = slp.modelPhaseI()
        baseII_tmp = slpI.primalSimplex(baseI, verbosity=verbosity)

        # Phase II
        baseII = baseII_tmp.extract_baseII(slp, slpI.A)
        optiBasis = slp.primalSimplex(baseII, verbosity=verbosity)

        # Results
        z_slp = optiBasis.x.dot(slp.c)
        z = -z_slp if self.flag_max else z_slp
        if verbosity >= 0:
            print(f"Basis that gives optimal value z = {slp.offset+z} = {z} + {slp.offset} (offset)")
            print(optiBasis)
        if verbosity > 0:
            print("----------------------")
        
        return optiBasis

    
    def getResult(self, base: Basis):
            """Returns the solution and its associated objective value from the given basis.
            Args:
                base (Basis): basis of the solution (in the SLP formulation).
            Returns:
                (np.ndarray): Optimal solution for the initial problem formulation
                (float): Associated optimal value (without the offset of the SLP formulation).
            """
            x_res = base.x[0:self.n] + self.x_l
            z_res = np.dot(x_res,self.c)
            return x_res, z_res


    def __str__(self):
        """Displays the general LP problem."""
        # Objective
        pb_str = "Max " if self.flag_max else "Min "
        for i in range(len(self.c)-1):
            pb_str += f" {self.c[i]}•X_{i} +"
        pb_str += f" {self.c[-1]}•X_{len(self.c)-1}"
        # Constraints
        pb_str +=("\nsubject to :")
        for cstr in self.constraints:
            pb_str += f"\n{cstr}"
        # Variables bounds
        for i in range(self.n):
            pb_str += f"\n ⦿ X_{i} ∈ "
            pb_str += f"({self.x_l[i]}, " if self.x_l[i] == -np.inf else f"[{self.x_l[i]}, "
            pb_str += f"{self.x_u[i]})" if self.x_u[i] == np.inf else f"{self.x_u[i]}]"
        return pb_str +"\n"
