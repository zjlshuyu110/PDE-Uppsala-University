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
    xl = -4
    xr = 6
    x_wall_start = 1
    x_wall_end = 2
    board_thickness = 0.2  # Thickness of the thin boards
    separation = 0.6       # Separation between the boards

    # Material properties
    c1 = 1    # Speed of sound in MEDIUM1 (air)
    rho1 = 1  # Density in MEDIUM1 (air)
    beta1 = 0 # Damping in MEDIUM1

    c2 = 2    # Speed of sound in MEDIUM2 (boards)
    rho2 = 2  # Density in MEDIUM2 (boards)
    beta2 = 0 # Damping in MEDIUM2 (will be set to 10 in the second test)

    CFL = 1   # Courant–Friedrichs–Lewy condition

PARAMS = SimulationParams()

def wave_equation_with_composite_wall(
    m: int, t_end: float, plot_times: list[float], filename_suffix: str, beta2_value: float, order: int = 6
):
    """
    Solve the 1D wave equation with two thin boards (composite wall) using 6th or 7th-order SBP-Projection method.
    """
    # Update beta2 in PARAMS
    PARAMS.beta2 = beta2_value

    # Generate grid and spacing
    h = (PARAMS.xr - PARAMS.xl) / (m - 1)
    x = np.linspace(PARAMS.xl, PARAMS.xr, m)

    # Define the positions of the thin boards
    board1_start = PARAMS.x_wall_start
    board1_end = PARAMS.x_wall_start + PARAMS.board_thickness
    board2_start = PARAMS.x_wall_end - PARAMS.board_thickness
    board2_end = PARAMS.x_wall_end

    # Initialize material properties
    c = np.full_like(x, PARAMS.c1)
    rho = np.full_like(x, PARAMS.rho1)
    beta = np.full_like(x, PARAMS.beta1)

    # Assign properties for the thin boards (MEDIUM2)
    board_indices = ((x >= board1_start) & (x <= board1_end)) | ((x >= board2_start) & (x <= board2_end))
    c[board_indices] = PARAMS.c2
    rho[board_indices] = PARAMS.rho2
    beta[board_indices] = PARAMS.beta2

    # Construct the SBP operators
    if order == 6:
        H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_6th(m, h)
        order_str = "6th Order"
    elif order == 7:
        H, HI, D1, D2, e_l, e_r, d1_l, d1_r = ops.sbp_cent_7th(m, h)
        order_str = "7th Order"
    else:
        raise ValueError("Unsupported order. Choose 6 or 7.")

    # Construct the derivative operator
    D_x = np.block([[np.zeros((m, m)), D1.toarray()], [D1.toarray(), np.zeros((m, m))]])

    # Construct the inverse SBP operator C (including damping)
    C_inv = np.block([
        [np.diag(rho * c**2), np.zeros((m, m))],
        [np.zeros((m, m)), np.diag(1 / rho)]
    ])

    # Construct the damping matrix
    Beta = np.diag(beta)
    Beta_block = np.block([
        [Beta, np.zeros((m, m))],
        [np.zeros((m, m)), np.zeros((m, m))]
    ])
 # also can cheng the name to D
    # Construct the L matrix and HI block matrix to compute the projection operator P
    L = np.block(
        [
            [np.kron(np.array([1, 1]), e_l.toarray())],
            [np.kron(np.array([1, -1]), e_r.toarray())],
        ]
    )
    HI_block = np.block(
        [
            [HI.toarray(), np.zeros_like(HI.toarray())],
            [np.zeros_like(HI.toarray()), HI.toarray()],
        ]
    )

    P = np.eye(2 * m) - HI_block @ L.T @ np.linalg.inv(L @ HI_block @ L.T) @ L

    # Compute the system matrix M including damping
    M = -P @ C_inv @ (D_x + Beta_block) @ P 

    # Generate initial conditions
    p, v = generate_initial_gaussian_pv(x, t=0, r_star=0.1)
    u0 = np.concatenate([p, v])

    # Time integration parameters
    dt = PARAMS.CFL * h / np.max(c)
    num_steps = int(t_end / dt)

    # Time integration using RK4
    u = u0.copy()
    snapshots = {}
    for step in range(num_steps + 1):
        current_time = step * dt
        if any(np.isclose(current_time, plot_times, atol=dt)):
            snapshots[current_time] = u[:m].copy()  # Save pressure field snapshot

        # RK4 integration steps
        k1 = dt * (M @ u)
        k2 = dt * (M @ (u + 0.5 * k1))
        k3 = dt * (M @ (u + 0.5 * k2))
        k4 = dt * (M @ (u + k3))

        u += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Plot snapshots at desired times
    for t, snapshot in snapshots.items():
        plt.figure(figsize=(12, 6))
        plt.plot(x, snapshot, label=f"Pressure at t = {t:.2f}")
        plt.axvspan(board1_start, board1_end, color='grey', alpha=0.3, label='Board')
        plt.axvspan(board2_start, board2_end, color='grey', alpha=0.3)
        plt.title(f"Wave Propagation at t = {t:.2f}, m = {m} ({order_str})")
        plt.xlabel("x")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"wave_composite_{filename_suffix}_t_{t:.2f}.png", dpi=300)
        plt.show()


# Run simulations for both tests

# Test 1: No Damping (beta1 = beta2 = 0)
print("Running Test 1: No Damping")
wave_equation_with_composite_wall(
    m=1001, t_end=5.0, plot_times=[5.0], filename_suffix="test1_no_damping_order6", beta2_value=0, order=6
)

# Uncomment below to run with 7th order operator if available
# wave_equation_with_composite_wall(
#     m=1001, t_end=5.0, plot_times=[5.0], filename_suffix="test1_no_damping_order7", beta2_value=0, order=7
# )

# Test 2: With Damping in Boards (beta1 = 0, beta2 = 10)
print("Running Test 2: With Damping in Boards")
wave_equation_with_composite_wall(
    m=1001, t_end=5.0, plot_times=[5.0], filename_suffix="test2_with_damping_order6", beta2_value=10, order=6
)

# Uncomment below to run with 7th order operator if available
# wave_equation_with_composite_wall(
#     m=1001, t_end=5.0, plot_times=[5.0], filename_suffix="test2_with_damping_order7", beta2_value=10, order=7
# )
