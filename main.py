from milp_module import *
import numpy as np


# Knapsack problems
a1 = np.array([5., 7., 4., 3.])
b1_l = -np.inf
b1_u = 14.
cstr1 = Constraint(a1,"<=",b1_l,b1_u)
x_l = np.array([0., 0., 0., 0.])
x_u = np.array([1., 1., 1., 1.])
int_vars = np.array([True, True, True, True])
c = np.array([8., 11., 6., 4.])
ks = MILP_Problem()
ks.set_objective("Max",c)
ks.set_variable_bounds(x_l,x_u,int_vars)
ks.set_constraints([cstr1])


x_opti, z_opti = ks.solve(verbosity=0)
