from milp_module import *
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


x_opti, z_opti = ks.solve(verbosity=0)


# %% Basic 2D test


# Problem definition
# a1 = np.array([-1., 1.])
# a2 = np.array([-0.5, -1.])
# x_l = np.array([0.0,0.0])
# x_u = np.array([np.inf,np.inf])
# b1_l = -1.
# b1_u = np.inf
# b2_l = -2.
# b2_u = np.inf
# cstr1 = Constraint(a1,">=",b1_l,b1_u)
# cstr2 = Constraint(a2,">=",b2_l,b2_u)
# c = np.array([1., 1.])
# lp = LinearProblem()
# lp.set_objective("Max",c)
# lp.set_variable_bounds(x_l,x_u)
# lp.set_constraints([cstr1,cstr2])

# optiBasis, slp = lp.solve()


# # add constraint
# a3 = sparse.csc_matrix([-2., -1., 0., 0.])
# b3 = -4

# dual_base, slp_r = slp.restraint(optiBasis, a3, b3)
# opti_base = slp_r.dual_simplex(dual_base, verbosity=1)
# print(opti_base)


# %%

# Solving
# slp = lp.to_SLP()
# print(slp.A)
# print(slp.b)
# print(slp.c)

# slpI, baseI = slp.modelPhaseI()
# print(slpI.A)
# print(slpI.b)
# print(slpI.c)
# print(baseI.B)
# print(baseI.N)
# print(baseI.x)
# print(baseI.n)
# print(baseI.m)
# print(baseI.invA_B)

# baseII_tmp = slpI.primalSimplex(baseI)
# print(baseII_tmp.B)
# print(baseII_tmp.N)
# print(baseII_tmp.x)
# print(baseII_tmp.n)
# print(baseII_tmp.m)
# print(baseII_tmp.invA_B)

# baseII = baseII_tmp.extract_baseII()
# print(baseII.B)
# print(baseII.N)
# print(baseII.x)
# print(baseII.n)
# print(baseII.m)
# print(baseII.invA_B)

# optiBasis = slp.primalSimplex(baseII)
# print(optiBasis.B)
# print(optiBasis.N)
# print(optiBasis.x)
# print(optiBasis.n)
# print(optiBasis.m)
# print(optiBasis.invA_B)
# %%
"""
Rajouter des flags pour la verbosité des logs
"""

# %%

# a = np.array([0,1,2,3,4])

# %% Old test
# from lp_module import *
# import numpy as np

# a1 = np.array([1,3,0,0])
# a2 = np.array([1,0,-4,2])
# a3 = np.array([0,3,-1,0])
# b1 = -2
# b2 = 0
# b3 = 1
# c = np.array([1,2,3,4])

# cstr1 = Constraint(a1,b1,">=")
# cstr2 = Constraint(a2,b2,"==")
# cstr3 = Constraint(a3,b3,"<=")

# lp = LinearProblem()
# lp.set_objective("Min",c)

# lp.set_constraints([cstr1,cstr2,cstr3])

# # lp.diplay()

# slp = lp.to_SLP()
# slpI, baseI = slp.modelPhaseI()

# baseII = slpI.primalSimplex(baseI)
# # solution = primalSimplex(slp,baseII)
# print(baseII.x)
# # modifier solution baseII pour supprimer les varaibles z utiles uniquement pour la phase I
# # Faire fonctino solveLP(lp: LinearProblem) (juste ça)
