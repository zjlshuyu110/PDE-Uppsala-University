import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def analytic_solution(x, t, r_star = 0.1):

    theta_1 = np.exp(-((x - t) / r_star) ** 2 )
    theta_2 = -np.exp(-((x + t) / r_star)** 2 )

    p_exact = theta_2 - theta_1
    v_exact = theta_1 + theta_2

    return p_exact, v_exact

def compute_and_save_analytic_solutions():

    x_l, x_r = -1, 1
    L = x_r - x_l
    t_star = 1.8
    r_star = 0.1
    grid_sizes = [101, 201, 401, 601, 801]

    for m in grid_sizes:
        x = np.linspace(x_l, x_r, m)

        p_exact, v_exact = analytic_solution(x, L - t_star, r_star)
        data = {"x" : x, "Pressure (p)" : p_exact, "Velocity (v)" : v_exact}
        df = pd.DataFrame(data)
        output_filename = f"analytic_solution_m_{m}.csv"
        df.to_csv(output_filename, index=False)

        # Plotting the pressure
        plt.figure()
        plt.plot(x, p_exact, label = "Pressure (p)")
        plt.title(f"Analytical Solution (m = {m})")
        plt.xlabel("x")
        plt.ylabel("Pressure")
        plt.legend()
        plt.grid()
        plt.show()

compute_and_save_analytic_solutions()
