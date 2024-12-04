import numpy as np
import pandas as pd

def compute_errors_and_l2_norm():

    grids = [101, 201, 401, 601, 801]
    t_star = 1.8 
    r_star = 0.1 
    x_l, x_r = -1, 1
    
    errors = []
    l2_norms = []

    for m in grids:
        analytic_filename = f"analytic_solution_m_{m}.csv"
        analytic_df = pd.read_csv(analytic_filename)
        p_analytic = analytic_df["Pressure (p)"].values

        numerical_filename = f"numerical_solution_6th_m_{m}.csv"
        numerical_df = pd.read_csv(numerical_filename)
        p_numerical = numerical_df["Pressure (p)"].values

        error = p_analytic - p_numerical
        errors.append(error)

        h = (x_r - x_l) / (m - 1)

        l2_norm = np.sqrt(h) * np.linalg.norm(error)
        l2_norms.append(l2_norm)

    results = {"Grid Size (m)": grids, "L2-Norm of Error": l2_norms}
    results_df = pd.DataFrame(results)
    results_df.to_csv("error_analysis.csv", index = False)

    print("Error analysis saved to error_analysis.csv")
    print(results_df)

# Execute the error computation
compute_errors_and_l2_norm()
