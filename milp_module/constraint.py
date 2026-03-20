import numpy as np

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
