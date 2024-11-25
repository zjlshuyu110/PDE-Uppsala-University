import numpy as np
import pandas as pd
import operators as ops

def compute_numerical_solution_6th(m, t_star, cfl = 0.05, r_star = 0.1):
    
    # Define domain and initial conditions
    x_l = -1
    x_r = 1

    c = 1
    h = (x_r - x_l) / (m - 1)
    k = cfl * h
    
    x = np.linspace(x_l, x_r, m)

    # Define SBP operator
    H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(m, h)
    Dx = D1 

    theta_1 = np.exp(-(x - 0)**2 / r_star)
    theta_2 = -np.exp(-(x + 0)**2 / r_star)

    p_exact = theta_1 - theta_2
    v_exact = theta_1 + theta_2

    p_numerical = p_exact.copy()
    v_numerical = v_exact.copy()

    # Time-stepping using RK4
    t = 0
    while t < t_star:
        if t + k > t_star:
            k = t_star - t

        # RK4 steps for pressure and velocity
        k1_p = -c * Dx @ v_numerical
        k1_v = -c * Dx @ p_numerical

        k2_p = -c * Dx @ (v_numerical + 0.5 * k * k1_v)
        k2_v = -c * Dx @ (p_numerical + 0.5 * k * k1_p)

        k3_p = -c * Dx @ (v_numerical + 0.5 * k * k2_v)
        k3_v = -c * Dx @ (p_numerical + 0.5 * k * k2_p)

        k4_p = -c * Dx @ (v_numerical + k * k3_v)
        k4_v = -c * Dx @ (p_numerical + k * k3_p)

        # Update pressure and velocity
        p_numerical += (k / 6) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        v_numerical += (k / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        t += k

    return x, p_numerical, v_numerical


def compute_and_save_numerical_solutions_6th():
    grids = [101, 201, 401, 601, 801]
    t_star = 1.8

    for m in grids:

        x, p_numerical, v_numerical = compute_numerical_solution_6th(m, t_star)

        # Save results to a CSV file
        data = {"x": x, "Pressure (p)": p_numerical, "Velocity (v)": v_numerical}
        df = pd.DataFrame(data)
        output_filename = f"numerical_solution_6th_m_{m}.csv"
        df.to_csv(output_filename, index=False)

        # Print first few values for verification
        print(f"Numerical solution for m = {m} saved to {output_filename}")
        print(f"x values (first 5): {x[:5]}")
        print(f"Pressure (first 5): {p_numerical[:5]}")
        print(f"Velocity (first 5): {v_numerical[:5]}")
        print("----------------------------------------")

compute_and_save_numerical_solutions_6th()
