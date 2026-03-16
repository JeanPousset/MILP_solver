from lp_module import *
import numpy as np
import highspy
import os

def solve_HiGHS(mps_file: str):
    """Solves a LP problem with HiGHS solver from a MPS file.
    Args:
        mps_file (str): MPS file path.
    Returns:
        (np.ndarray): Optimal solution.
        (float): Optimal value.
    """
    h = highspy.Highs()
    h.readModel(mps_file)
    h.setOptionValue("output_flag", False)
    h.run()
    sol = h.getSolution()
    info = h.getInfo()
    return sol.col_value, info.objective_function_value

def solve_primal_simplex(mps_file: str):
    """Solves a LP problem with our hand-made primal simplex from a MPS file.
    Args:
        mps_file (str): MPS file path.
    Returns:
        (np.ndarray): Optimal solution.
        (float): Optimal value.
    """
    lp = LinearProblem.from_mps(mps_file)
    optimal_basis, slp = lp.solve(verbosity=-1)
    return lp.getResult(optimal_basis, slp.col_scales)

mps_file_names = ["adlittle","afiro","empstest","maros","nazareth","testprobs"]
mps_file_names = ["sc50b"]
mps_inf_neg_var = ["empstest","nazareth","testprob","adlittle"]
mps_doable = ["maros","afiro"]
mps_repo = "unit_tests/lp_instances/Netlib/"


mps_file_names = [
    os.path.splitext(f)[0]  # removes ".mps"
    for f in os.listdir(mps_repo)
    if f.endswith('.mps')
]


def primal_simplex_validation():

    for mps in mps_file_names:
        mps_path = mps_repo + mps + ".mps"

        print(f"-> test file : {mps}")
        
        x_highs, z_highs = solve_HiGHS(mps_path)
        x_ps, z_ps = solve_primal_simplex(mps_path)

        rel_err_z = np.abs(z_ps-z_highs)/(np.abs(z_ps)+1)
        err_x = np.linalg.norm(x_ps-x_highs)
        test_str = f" • {mps} : |rel_err_z| = {rel_err_z},\t ||err_x|| = {err_x} -->"
        test_str += "[passed]" if (rel_err_z <= TOL_Z) else "[FAILED !]"
        print(test_str)


def primal_simplex_unique_validation(mps_name):

    mps_path = mps_repo + mps_name + ".mps"

    print(f"-> test file : {mps_name}")
    
    x_highs, z_highs = solve_HiGHS(mps_path)
    x_ps, z_ps = solve_primal_simplex(mps_path)

    rel_err_z = np.abs(z_ps-z_highs)/(np.abs(z_ps)+1)
    err_x = np.linalg.norm(x_ps-x_highs)
    test_str = f" • {mps_name} : |rel_err_z| = {rel_err_z},\t ||err_x|| = {err_x} -->"
    test_str += "[passed]" if (rel_err_z <= TOL_Z) else "[FAILED !]"
    print(test_str)


# primal_simplex_validation()

def primal_simplex_unique_validation(mps_name):

    mps_path = mps_repo + "Passed/" + mps_name + ".mps"

    print(f"-> test file : {mps_name}")
    
    x_highs, z_highs = solve_HiGHS(mps_path)
    x_ps, z_ps = solve_primal_simplex(mps_path)

    rel_err_z = np.abs(z_ps-z_highs)/(np.abs(z_ps)+1)
    err_x = np.linalg.norm(x_ps-x_highs)
    test_str = f" • {mps_name} : |rel_err_z| = {rel_err_z},\t ||err_x|| = {err_x} -->"
    test_str += "[passed]" if (rel_err_z <= TOL_Z) else "[FAILED !]"
    print(test_str)


primal_simplex_unique_validation("ship12l")