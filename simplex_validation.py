from lp_module import *
import numpy as np
import highspy

lp_maros = LinearProblem.from_mps("lp_instances/maros.mps")
lp_maros.solve()


h = highspy.Highs()
h.readModel("lp_instances/maros.mps")
h.setOptionValue("output_flag", False)
h.run()
sol = h.getSolution()
info = h.getInfo()
print(f" x_HiGHS = {sol.col_value}")
print(f" z_HiGHS = {info.objective_function_value}")
