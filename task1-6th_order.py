import numpy as np
import matplotlib.pyplot as plt
import operators as ops  # Assume this module provides SBP operators

def solve_wave_equation(m, t_end, plot_times, filename_suffix):
    # 1. Domain and Parameters
    xl, xr = -2, 4
    x_interface = 1
    c1, c2 = 1, 2
    rho1, rho2 = 1, 2
    r_star = 0.1  # Gaussian width
    CFL = 0.01

    # 2. Grid and Medium Properties
    h = (xr - xl) / (m - 1)
    x = np.linspace(xl, xr, m) 
    medium = np.where(x < x_interface, 1, 2)  # Medium indices
    c = np.where(medium == 1, c1, c2)
    rho = np.where(medium == 1, rho1, rho2)

    # 3. Corrected Initial Conditions
    theta_1 = np.exp(-((x / r_star) ** 2))  # θ^(1) at t = 0
    p0 = 2 * theta_1  # Pressure field starts as twice the Gaussian profile
    v0 = np.zeros_like(x)  # Velocity field starts as zero
    u = np.concatenate([p0, v0])  # Initial state vector (pressure + velocity)

    # 4. SBP Operators
    H, HI, D1, _, e_l, e_r, _, _ = ops.sbp_cent_6th(m, h)
    D_x = np.block([
        [np.zeros((m, m)), D1.toarray()], 
        [D1.toarray(), np.zeros((m, m))]
    ])
    C_inv = np.block([
        [np.diag(1 / (rho * c**2)), np.zeros((m, m))],
        [np.zeros((m, m)), np.diag(1 / rho)]
    ])
    P = np.eye(2 * m)  # Projection operator (identity for now)
    M = -P @ C_inv @ D_x @ P

    # 5. Time Integration Setup (RK4)
    dt = CFL * h / np.max(c)  # Time step based on CFL condition
    num_steps = round(t_end / dt)
    results = {}

    for step in range(num_steps):
        # RK4 integration
        k1 = dt * (M @ u)
        k2 = dt * (M @ (u + 0.5 * k1))
        k3 = dt * (M @ (u + 0.5 * k2))
        k4 = dt * (M @ (u + k3))
        u += (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Store results for specified times
        current_time = step * dt
        for t in plot_times:
            if np.isclose(current_time, t, atol=1e-3):  # Check time precision
                results[t] = u.copy()

    # 6. Validation and Visualization
    for t in plot_times:
        if t in results:
            u_result = results[t]
            p = u_result[:m]  # Pressure field
            v = u_result[m:]  # Velocity field

            # Separate the domain into left and right of the interface
            left_region = x < x_interface
            right_region = x >= x_interface

            # Incident, reflected, and transmitted waves
            incident_wave = p[left_region]
            reflected_wave = p[left_region]  # Same region, post-interaction
            transmitted_wave = p[right_region]

            # Measure amplitudes
            A_i = np.max(np.abs(incident_wave))
            A_r = np.max(np.abs(reflected_wave))
            A_t = np.max(np.abs(transmitted_wave))

            # Numerical reflection and transmission coefficients
            R_num = A_r / A_i if A_i != 0 else 0
            T_num = A_t / A_i if A_i != 0 else 0

            # Acoustic impedance
            Z1 = rho[0] * c[0]  # Impedance in Medium 1
            Z2 = rho[-1] * c[-1]  # Impedance in Medium 2

            # Theoretical reflection and transmission coefficients
            R_theory = (Z2 - Z1) / (Z1 + Z2)
            T_theory = 2 * Z2 / (Z1 + Z2)

            # Print validation results for this time
            print(f"=== Validation Results at t={t:.2f} ===")
            print(f"Reflection Coefficient (Numerical): {R_num:.4f}")
            print(f"Reflection Coefficient (Theoretical): {R_theory:.4f}")
            print(f"Transmission Coefficient (Numerical): {T_num:.4f}")
            print(f"Transmission Coefficient (Theoretical): {T_theory:.4f}")

            # Plot the results
            plt.figure(figsize=(10, 6))
            plt.plot(x, p, label=f"Pressure at t={t:.2f}")
            plt.plot(x, v, label=f"Velocity at t={t:.2f}")
            plt.axvline(x=x_interface, color="k", linestyle="--", label="Interface")
            plt.xlabel("Position (x)")
            plt.ylabel("Amplitude")
            plt.title(f"Wave Propagation at t={t:.2f}")
            plt.legend()
            plt.grid()
            plt.savefig(f"wave_{filename_suffix}_t{t:.2f}.png")  # Save graph
            plt.close()  # Ensure the plot is cleared after saving

# Example usage of the function
m = 201  # Number of grid points
t_end = 2.5  # End time for simulation
plot_times = [1.5, 2.5]  # Times to extract results for validation and plotting
filename_suffix = "wave_task1_corrected"

solve_wave_equation(m, t_end, plot_times, filename_suffix)
