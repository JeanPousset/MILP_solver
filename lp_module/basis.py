import numpy as np
from .param import *

class Basis:
    """Stores data of a simplex basis."""
    n:     int
    m:     int
    B:     np.ndarray
    N:     np.ndarray
    invA_B: np.ndarray
    x:     np.ndarray

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
        self.invA_B = np.zeros((m,m),dtype='d')
        self.x = np.zeros((n),dtype='d')

    def extract_baseII(self):
        """Extracts a feasible basis (baseII) from the optimal basis resulting of phase I of the simplex initialization.
        Returns:
            (Basis): a feasible basis for the initial SLP.
        """
        # Check that initialization variables z are nul
        if any(abs(self.x[range(self.n-self.m, self.n)]) > DIGITAL_0):
            raise ValueError(f"[solve]: problem is not feasible !")
        # Remove initialization variables z from baseI
        baseII = Basis(self.n-self.m,self.m)
        baseII.x = self.x[0:baseII.n]
        baseII.B = self.B
        baseII.N = np.setdiff1d(self.N,np.arange(baseII.n,self.n))
        baseII.invA_B = self.invA_B
        return baseII

    def __str__(self):
        """Gives strings to display when calling `print(:Basis)`."""
        res = f"Basis({self.n},{self.m})\n"
        res += f"in-basis: {self.B},\t out-basis: {self.N}\n"
        res += f"x = {self.x}"
        return res
