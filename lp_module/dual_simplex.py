from .primal_simplex import SLP_Model

class DSLP_Model:
    """Represents the dual of a Linear Programming problem in its standard formluation (DSLP).
    Attributes:
        A_T (np.ndarray): Contraints matrix.
        c (np.ndarray): 2nd member of the constraints.
        b (np.ndarray): Vector of the objective function.
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