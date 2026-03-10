# %% Classes
import numpy as np
from .primal_simplex import SLP_Model
from .param import *


class Constraint:
    """Defines a linear constraint of the form a*x ? b, where ? is symbol of equality / inequality

    Attributes:
        a (np.ndarray): Constraint coefficients (for the linear combinaison of x).
        b (np.float64): 2nd member of the constraint.
        symbol (str):   Operator that must be either an equality `==` or an inequality `<=` / `>=`.
    """
    a:         np.ndarray
    b:         np.float64
    symbol:    str

    def __init__(self, a: np.ndarray, b: float,  symbol: str):
        """ Initializes an instance of Constraint.

        Args:
            a (np.ndarray): Vector of coefficients constraints.
            b (float):      2nd member of the constraint.
            symbol (str):   Operator that must be either an equality `==` or an inequality `<=` / `>=`.
        """
        assert any(symbol == s for s in ["<=",">=","=="]), f"[Constraint]: `symbol` argument must be either '<=' or '>=' (given : '{symbol}')"
        self.a = a
        self.b = np.float64(b)
        self.symbol = symbol

    def __str__(self):
        """Returns string that displays the full constraint."""
        cstr_row = " ⦿ "
        for i in range(len(self.a)-1):
            cstr_row += f" {self.a[i]}•X_{i} +"
        cstr_row += f" {self.a[-1]}•X_{len(self.a)-1}"
        cstr_row += f" {self.symbol} {self.b}"
        return cstr_row


class LinearProblem:
    """Represents a general Linear Programming (LP).

    Attributes:
        n (int): number of variables (problem dimension).
        m (int): number of constraints.
        constraints (List[Constraints]): list of linear constraints.
        c (np.ndarray): vector of the objective function.
        flag_max (bool): boolean value that is true if the given objective is a 'Max' instead of a 'Min'.
    """
    n: int
    m: int
    constraints: list[Constraint]
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
            extremum (str): character string to indicate wheter it is a 'Min' or 'Max' objective.
            c (np.ndarray): vector of the objective function.
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

    def set_constraints(self, constraints: list[Constraint]):
        """Set the given constraints for the LP problem.
        Args:
            constraints (list[Constraints]): list of linear constraints.
        """
        assert len(constraints) <= self.n, f"[set_constraints]: the number of constraints ({len(constraints)}) can not exceed the number variables ({self.n})"
        self.m = 0
        for cstr in constraints:
            assert len(cstr.a) == self.n, f"[set_constraints]: constraint `a` length ({len(cstr.a)}) does not match number of variables ({self.n})"
            self.constraints.append(cstr)
            self.m += 1


    def to_SLP(self):
        """Converts a general linear programming problem into a standard linear programming (SLP) formulation."""
        slp = SLP_Model()

        # build matrix of constraints
        deviation_ind = np.zeros((self.m),dtype=bool) # number of deviation variables to add when switching to the SLP formulation
        nb_deviation = 0
        A = []
        b = []
        for j in range(self.m):
            if self.constraints[j].symbol == ">=":
                A.append(self.constraints[j].a)
                b.append(self.constraints[j].b)
                deviation_ind[j] = True
                nb_deviation += 1
            elif self.constraints[j].symbol == "<=":
                A.append(-self.constraints[j].a)
                b.append(-self.constraints[j].b)
                deviation_ind[j] = True
                nb_deviation += 1
            else: # equality "=="
                A.append(self.constraints[j].a)
                b.append(self.constraints[j].b)
                deviation_ind[j] = False
        A = np.array(A)
        dev_matrix = np.zeros((self.m,nb_deviation), dtype='d') # constraint matrix for deviation coefficients s
        dev_matrix[deviation_ind, range(nb_deviation)] = -1.
        slp.A = np.hstack((A,dev_matrix))

        # 2nd member of constraints
        slp.n = self.n + nb_deviation
        slp.m = self.m
        slp.b = np.array(b)

        # # Case where b must be postive
        # for i in range(slp.m):
        #     # multiply a_i*x - s = b_i by (-1) if b_i < 0
        #     if b[i] < 0:
        #         slp.A[i,:] *= -1.
        #         slp.b[i] *= -1.

        # objective function
        new_c = -self.c if self.flag_max else self.c
        slp.c = np.concatenate([new_c,np.zeros((nb_deviation),dtype='d')])

        return slp

    def solve(self):

        # Display
        print("----------------------")
        print("Resolution of problem:\n")
        print(self)

        # Phase I
        slp = self.to_SLP()
        slpI, baseI = slp.modelPhaseI()
        baseII_tmp = slpI.primalSimplex(baseI)

        # Phase II
        baseII = baseII_tmp.extract_baseII()
        optiBasis = slp.primalSimplex(baseII)

        # Results
        z_slp = optiBasis.x.dot(slp.c)
        z = -z_slp if self.flag_max else z_slp
        print(f"Basis that gives optimal value z = {z}")
        print(optiBasis)
        print("----------------------")


    def __str__(self):
        """Displays the general LP problem."""
        pb_string = "Max " if self.flag_max else "Min "
        for i in range(len(self.c)-1):
            pb_string += f" {self.c[i]}•X_{i} +"
        pb_string += f" {self.c[-1]}•X_{len(self.c)-1}"
        pb_string +=("\nsubject to :")
        for cstr in self.constraints:
            pb_string += f"\n{cstr}"
        return pb_string +"\n"

# %% Test

a1 = np.array([1,3,0,0])
a2 = np.array([1,0,-4,2])
a3 = np.array([0,3,-1,0])
b1 = -2
b2 = 0
b3 = 1
c = np.array([1,2,3,4])

cstr1 = Constraint(a1,b1,">=")
cstr2 = Constraint(a2,b2,"==")
cstr3 = Constraint(a3,b3,"<=")

lp = LinearProblem()
lp.set_objective("Min",c)

lp.set_constraints([cstr1,cstr2,cstr3])

# lp.diplay()

slp = lp.to_SLP()
phaseI, baseI = slp.modelPhaseI()
