import numpy as np
import pandas as pd
import operators as ops
import matplotlib.pyplot as plt
from tqdm import tqdm 


def generate_initial_gaussian_pv(x: np.ndarray, t: float, r_star: float) -> tuple[np.ndarray, np.ndarray]:
    theta_1 = np.exp(-((x - t) / r_star) ** 2)
    theta_2 = -np.exp(-((x + t) / r_star) ** 2)
    p = theta_1 - theta_2
    v = theta_1 + theta_2
    return p, v


def compute_numerical_solution_6th(m, t_star, cfl = 0.05, r_star = 0.1):
    # Define domain and initial conditions
    x_l, x_r = -1, 1
    h = (x_r - x_l) / (m - 1)
    c = 1
    rho = 1
    dt = cfl * h / np.max(c)
    x = np.linspace(x_l, x_r, m)

    # Generate initial conditions
    p, v = generate_initial_gaussian_pv(x, t = 0, r_star = r_star)
    u = np.concatenate([p, v])

    # Define SBP operator
    H, HI, Dp, Dm, e_l, e_r = ops.sbp_upwind_7th(m, h)

    D_x = np.block([[np.zeros((m, m)), Dp],
                    [Dm, np.zeros((m, m))]])

    # Construct the inverse SBP operator C
    C_inv = np.block([[np.eye(m) / (rho * c**2), np.zeros((m, m))],
                      [np.zeros((m, m)), np.eye(m) / rho]])
    L = np.block([[np.kron(np.array([1, 0]), e_l)],
                  [np.kron(np.array([1, 0]), e_r)]])
    HI_block = np.block([[HI, np.zeros_like(HI)],
                         [np.zeros_like(HI), HI]])

    P = np.eye(2 * m) - HI_block @ L.T @ np.linalg.inv(L @ HI_block @ L.T) @ L
    # Compute the system matrix M
    M = -P @ C_inv @ D_x @ P

    # Time-stepping using RK4
    t = 0
    with tqdm(total=t_star, desc=f"Grid size {m}", unit="s", position=1) as pbar:
        while t < t_star:
            if t + dt > t_star:
                dt = t_star - t
            k1 = dt * M @ u
            k2 = dt * M @ (u + 0.5 * k1)
            k3 = dt * M @ (u + 0.5 * k2)
            k4 = dt * M @ (u + k3)
            u += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t += dt
            pbar.update(dt)

    return x, u[:m], u[m:]


def compute_and_save_numerical_solutions_6th():
    grids = [101, 201, 401, 601, 801]
    t_star = 1.8

    for m in grids:
        x, p_numerical, v_numerical = compute_numerical_solution_6th(m, t_star)

        # Save results to a CSV file
        data = {"x": x, "Pressure (p)": p_numerical, "Velocity (v)": v_numerical}
        df = pd.DataFrame(data)
        output_filename = f"numerical_solution_7th_m_{m}.csv"
        df.to_csv(output_filename, index = False)

        # Plot results
        plt.figure()
        plt.plot(x, p_numerical, label="Numerical Pressure (p)")
        plt.xlabel("x")
        plt.ylabel("Pressure")
        plt.title(f"Numerical Solution at t={t_star} (m={m})")
        plt.grid()
        plt.legend()
        plt.show()

        print(f"Numerical solution for m = {m} saved to {output_filename}")
        print("----------------------------------------")

compute_and_save_numerical_solutions_6th()
