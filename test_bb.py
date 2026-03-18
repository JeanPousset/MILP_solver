from lp_module import *
import numpy as np
from scipy import sparse

## Basic discrete product choices
# a1 = np.array([1., 2.])
# a2 = np.array([10., 6.])
# x_l = np.array([0, 0])
# x_u = np.array([50.,50])
# b1_l = -np.inf
# b1_u = 5.
# b2_l = -np.inf
# b2_u = 45.
# cstr1 = Constraint(a1,"<=",b1_l,b1_u)
# cstr2 = Constraint(a2,"<=",b2_l,b2_u)
# c = np.array([5., 4.])
# milp = MILP_Problem()
# milp.set_objective("Max",c)
# milp.set_variable_bounds(x_l,x_u, np.array([True, True], dtype=bool))
# milp.set_constraints([cstr1,cstr2])

# x_opti, z_opti = milp.branch_and_bound(verbosity=1)

# print(f"end of B&B : z = {z_opti}\n\t • x = {x_opti}")

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

print(ks)

x_opti, z_opti = ks.branch_and_bound(verbosity=1)

print(f"end of B&B : z = {z_opti}\n\t • x = {x_opti}")