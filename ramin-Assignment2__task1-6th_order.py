from dataclasses import dataclass
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from scipy.sparse import data
import sbp.operators as ops

def wave_equation_with_interface_6th(m, t_end, plot_times, filename_suffix):
    """
    Solve the 1D wave equation with two materials using 6th-order SBP-Projection method.
    """
    # Parameters
    xl = -2
    xr = 4
    x_interface = 1

    # Material properties
    c1 = 1
    c2 = 2
    rho1 = 1
    rho2 = 2

    # Artificial dissipation parameters
    beta = 0
    CFL = 0.05  # Based on the assignment_1

    # Generate grid and spacing
    h = (xr - xl) / (m - 1)
    x = np.linspace(xl, xr, m)

    # Define the material interface
    medium = np.where(x < x_interface, 1, 2)

    # Define material properties across the grid
    c = np.where(medium == 1, c1, c2)
    rho = np.where(medium == 1, rho1, rho2)

    # Construct the 6th-order SBP operators
    H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(m, h)

    # Construct the derivative operator
    D_x = np.block([[np.zeros((m, m)), D1.toarray()],
                    [D1.toarray(), np.zeros((m, m))]])

    # Construct the inverse SBP operator C
    C_inv = np.block([[np.eye(m) / (rho * c**2), np.zeros((m, m))],
                      [np.zeros((m, m)), np.eye(m) / rho]])

    # Construct the L matrix and HI block matrix to compute the projection operator P
    L = np.block([[np.kron(np.array([1, 1]), e_l.toarray())], 
                  [np.kron(np.array([1, -1]), e_r.toarray())]])
    HI_block = np.block([[HI.toarray(), np.zeros_like(HI.toarray())],
                         [np.zeros_like(HI.toarray()), HI.toarray()]])

    P = np.eye(2 * m) - HI_block @ L.T @ np.linalg.inv(L @ HI_block @ L.T) @ L

    # Compute the system matrix M
    M = -P @ C_inv @ D_x @ P

    # Initial conditions
    r_star = 0.1
    # p = np.exp(-((x - (-1)) / r_star) ** 2) - np.exp(-((x + 1) / r_star) ** 2)  # Initial pressure
    p = np.exp(-x ** 2 / (2 * r_star ** 2))
    # v = np.exp(-((x - (-1)) / r_star) ** 2) + np.exp(-((x + 1) / r_star) ** 2)  # Initial velocity
    v = np.zeros_like(p)

    u0 = np.concatenate([p, v])

    # Time integration parameters
    dt = CFL * h / np.max(c)
    num_steps = int(t_end / dt)

    # Initialize snapshot storage
    snapshots = {}

    def plot_animated_graph(u0):
        @dataclass
        class Temp:
            u: np.ndarray

        u = Temp(u0)
        fig, ax = plt.subplots()
        ax.set_ylim(-1, 1)
        (line,) = ax.plot(x, u.u[:m])

        def update(frame, u: Temp):
            # RK4 integration steps
            k1 = dt * M @ u.u
            k2 = dt * M @ (u.u + 0.5 * k1)
            k3 = dt * M @ (u.u + 0.5 * k2)
            k4 = dt * M @ (u.u + k3)

            u.u += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            line.set_ydata(u.u[:m])
            ax.set_title(f"t={frame/num_steps:.3f}")
            if frame >= num_steps - 1:
                ani.event_source.stop()
            return (line,)

        ani = FuncAnimation(
            fig,  # Figure to animate
            update,
            fargs=(u,),  # Update function
            frames=num_steps,  # Number of frames
            interval=1,  # Delay between frames in milliseconds
            blit=True
        )
        plt.show()


    plot_animated_graph(u0)

    u = u0

    # Time integration using RK4
    for step in range(num_steps + 1):
        current_time = step * dt
        if any(np.isclose(current_time, plot_times, atol=dt)):
            snapshots[current_time] = u[:m].copy()  # Save pressure field snapshot

        # RK4 integration steps
        k1 = dt * M @ u
        k2 = dt * M @ (u + 0.5 * k1)
        k3 = dt * M @ (u + 0.5 * k2)
        k4 = dt * M @ (u + k3)

        u += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Plot snapshots at desired times
    for t, snapshot in snapshots.items():
        plt.figure(figsize=(8, 6))
        plt.plot(x, snapshot, label=f"Pressure at t = {t:.2f}")
        plt.axvline(x_interface, color='k', linestyle='--', label="Interface")
        plt.title(f"Wave Propagation at t = {t:.2f}, m = {m} (6th Order)")
        plt.xlabel("x")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"wave_6th_order_{filename_suffix}_t_{t:.2f}.png", dpi=300)
        plt.show()

    # Calculate and print transmission and reflection coefficients
    Z1 = rho1 * c1  # Impedance
    Z2 = rho2 * c2
    T = 2 * Z2 / (Z1 + Z2)
    R = (Z2 - Z1) / (Z1 + Z2)
    print(f"Grid size m = {m}")
    print(f"Transmission Coefficient (T): {T}")
    print(f"Reflection Coefficient (R): {R}")

# Run simulations for m = 201 and m = 401
wave_equation_with_interface_6th(m=201, t_end=2.5, plot_times=[1.5, 2.5], filename_suffix="m201")
wave_equation_with_interface_6th(m=401, t_end=2.5, plot_times=[1.5, 2.5], filename_suffix="m401")