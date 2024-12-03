import matplotlib.pyplot as plt
import numpy as np
import sbp.operators as ops

from dataclasses import dataclass
from matplotlib.animation import FuncAnimation


def generate_initial_gaussian_pv(
    x: np.ndarray, t: float, r_star: float
) -> tuple[np.ndarray, np.ndarray]:
    p = np.exp(-(((x - t) / r_star) ** 2)) + np.exp(-(((x + t) / r_star) ** 2))
    v = np.exp(-(((x - t) / r_star) ** 2)) - np.exp(-(((x + t) / r_star) ** 2))
    return p, v


@dataclass
class SimulationParams:
    xl = -2
    xr = 4
    x_interface = 1

    # Material properties
    c1 = 1
    c2 = 2
    rho1 = 1
    rho2 = 2

    beta = 0  # Artificial dissipation parameters
    CFL = 1   # Based on the assignment_1


PARAMS = SimulationParams()


def wave_equation_with_interface_7th(
    m: int, t_end: float, plot_times: list[float], filename_suffix: str
):
    """
    Solve the 1D wave equation with two materials using 7th-order SBP-Upwind method.
    """

    # Generate grid and spacing
    h = (PARAMS.xr - PARAMS.xl) / (m - 1)
    x = np.linspace(PARAMS.xl, PARAMS.xr, m)

    # Define the material interface
    medium = np.where(x < PARAMS.x_interface, 1, 2)

    # Define material properties across the grid
    c = np.where(medium == 1, PARAMS.c1, PARAMS.c2)
    rho = np.where(medium == 1, PARAMS.rho1, PARAMS.rho2)

    # Construct the 7th-order SBP operators
    H, HI, Dp, Dm, e_l, e_r = ops.sbp_upwind_7th(m, h)

    # Construct the derivative operator using upwind scheme
    D_x = np.block([[np.zeros((m, m)), Dp], [Dm, np.zeros((m, m))]])

    # Construct the inverse SBP operator C
    C_inv = np.block(
        [
            [np.eye(m) * (rho * c**2), np.zeros((m, m))],
            [np.zeros((m, m)), np.eye(m) / rho],
        ]
    )

    # Construct the L matrix and HI block matrix to compute the projection operator P
    L = np.block(
        [
            [np.kron(np.array([1, 1]), e_l)],
            [np.kron(np.array([1, -2]), e_r)],
        ]
    )
    HI_block = np.block(
        [
            [HI, np.zeros_like(HI)],
            [np.zeros_like(HI), HI],
        ]
    )

    P = np.eye(2 * m) - HI_block @ L.T @ np.linalg.inv(L @ HI_block @ L.T) @ L

    # Compute the system matrix M
    M = -P @ C_inv @ D_x @ P

    p, v = generate_initial_gaussian_pv(x, t=0, r_star=0.1)

    u0 = np.concatenate([p, v])

    # Time integration parameters
    dt = PARAMS.CFL * h / np.max(c)
    num_steps = int(t_end / dt)

    def plot_animated_graph(u0):
        @dataclass
        class Temp:
            u: np.ndarray

        u = Temp(np.copy(u0))
        fig, ax = plt.subplots()
        ax.set_ylim(bottom=-1.1, top=2)
        ax.axvline(PARAMS.x_interface, color="k", linestyle="--", label="Interface")
        ax.set_xlabel("x")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(visible=True)
        (line,) = ax.plot(x, u.u[:m])

        def update(frame, u: Temp):
            # RK4 integration steps
            k1 = dt * M @ u.u
            k2 = dt * M @ (u.u + 0.5 * k1)
            k3 = dt * M @ (u.u + 0.5 * k2)
            k4 = dt * M @ (u.u + k3)

            u.u += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            line.set_ydata(u.u[:m])
            ax.set_title(f"t={(frame + 1)* dt:.3f}")
            if frame >= num_steps - 1:
                ani.event_source.stop()
            return (line,)

        ani = FuncAnimation(
            fig,  # Figure to animate
            update,
            fargs=(u,),  # Update function
            frames=num_steps + 1,  # Number of frames
            interval=int(dt * 1000),  # Delay between frames in milliseconds
        )
        plt.show()

    plot_animated_graph(u0)

    u = u0

    def iterate_solver(u):
        snapshots = {}
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
        return snapshots

    snapshots = iterate_solver(u0)

    # Plot snapshots at desired times and calculate coefficients
    for t, snapshot in snapshots.items():
        plt.figure(figsize=(8, 6))
        plt.plot(x, snapshot, label=f"Pressure at t = {t:.2f}")
        plt.ylim(bottom=-1.1, top=1.1)
        plt.axvline(PARAMS.x_interface, color="k", linestyle="--", label="Interface")
        plt.title(f"Wave Propagation at t = {t:.2f}, m = {m} (7th Order)")
        plt.xlabel("x")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"wave_7th_order_{filename_suffix}_t_{t:.2f}.png", dpi=300)
        plt.show()

        left_region = x < PARAMS.x_interface
        right_region = x >= PARAMS.x_interface

        # Incident, reflected, and transmitted waves
        reflected_wave = snapshot[left_region]  # Same region, post-interaction
        transmitted_wave = snapshot[right_region]

        # Measure amplitudes
        A_i = 1  # Incident wave amplitude (assuming normalized)
        A_r = np.max(np.abs(reflected_wave))  # Maximum amplitude of reflected wave
        A_t = np.max(np.abs(transmitted_wave))  # Maximum amplitude of transmitted wave

        # Numerical reflection and transmission coefficients
        R_num = A_r / A_i
        T_num = A_t / A_i

        print(f"At time t = {t:.2f}:")
        print(f"Reflection Coefficient (Numerical): {R_num:.4f}")
        print(f"Transmission Coefficient (Numerical): {T_num:.4f}")


# Run simulations for m = 601
wave_equation_with_interface_7th(
    m=601, t_end=2.5, plot_times=[1.8, 2.5], filename_suffix="m601"
)
