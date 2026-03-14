from lp_module import *
import numpy as np
import highspy

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
    optimal_basis = lp.solve()
    return lp.getResult(optimal_basis)

mps_file_names = ["adlittle","afiro","empstest","nazareth","testprobs"]
mps_repo = "lp_instances/"


def primal_simplex_validation():

    for mps in mps_file_names:
        mps_path = mps_repo + mps + ".mps"
        
        x_highs, z_highs = solve_HiGHS(mps_path)
        x_ps, z_ps = solve_primal_simplex(mps_path)

        err_z = np.abs(z_ps-z_highs)
        err_x = np.linalg.norm(x_ps-x_highs)
        test_str = f" • {mps} : |err_z| = {err_z},\t ||err_x|| = {err_x} -->"
        test_str += "[passed]" if (err_x <= DIGITAL_0 and err_z <= DIGITAL_0) else "[FAILED !]"
        print(test_str)
