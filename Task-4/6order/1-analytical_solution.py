import numpy as np
import pandas as pd

def analytic_solution(x, t, r_star = 0.1):

    theta_1 = np.exp(-(x - t) ** 2 / r_star)
    theta_2 = -np.exp(-(x + t) ** 2 / r_star)

    p_exact = theta_1 - theta_2
    v_exact = theta_1 + theta_2
    
    return p_exact, v_exact

def compute_and_save_analytic_solutions():

    x_l = -1
    x_r = 1
    t_star = 1.8 
    r_star = 0.1

    grid_sizes = [101, 201, 401, 601, 801] 

    for m in grid_sizes:

        x = np.linspace(x_l, x_r, m)

        p_exact, v_exact = analytic_solution(x, t_star, r_star)

        data = {"x" : x, 
                "Pressure (p)" : p_exact, 
                "Velocity (v)" : v_exact}

        df = pd.DataFrame(data)

        output_filename = f"analytic_solution_m_{m}.csv"
        df.to_csv(output_filename, index=False)

compute_and_save_analytic_solutions()
