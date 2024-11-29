import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
import operators as ops  # Importing the provided SBP operators


def compute_eigenvalues_and_cfl_for_dirichlet(option: str):
    """
    Compute eigenvalues and CFL number for the 7th-order SBP operator with Dirichlet BCs.
    """
    # Parameters
    grids = [51, 101]  # Grid sizes
    xl = 0  # Left boundary
    xr = 1  # Right boundary
    rho = 1  # Density
    c = 1  # Speed of sound
    rk4_stability_limit = 2.8  # Stability domain of RK4

    for m in grids:
        h = (xr - xl) / (m - 1)  # Grid spacing

        # Retrieve 7th-order SBP operators
        H, HI, D_plus, D_minus, e_l, e_r = ops.sbp_upwind_7th(m, h)

        # Construct the 1st-order derivative operator D_x
        D_x = np.block([[np.zeros((m, m)), D_plus],
                        [D_minus, np.zeros((m, m))]])

        # Construct the inverse of the SBP operator C
        C_inv = np.block([[np.eye(m) / (rho * c**2), np.zeros((m, m))],
                          [np.zeros((m, m)), np.eye(m) / rho]])

        # Define the boundary operator L
        # Two rows for two boundaries
        # L = np.zeros((2, 2 * m))
        # L[0, m] = 1  # Left boundary velocity term
        # L[1, 2 * m - 1] = 1  # Right boundary velocity term
        if option == "Dirichlet":
            L = np.block([[np.kron(np.array([1, 0]), e_l)], [np.kron(np.array([1, 0]), e_r)]])
        else:
            L = np.block([[np.kron(np.array([1, 0.5]), e_l)], [np.kron(np.array([1, -0.5]), e_r)]])

        # Extend HI to a block diagonal form for the block system
        HI_block = np.block([[HI, np.zeros_like(HI)],
                             [np.zeros_like(HI), HI]])
        # Define the projection matrix PS
        P = np.eye(2 * m) - HI_block @ L.T @ np.linalg.inv(L @ HI_block @ L.T) @ L

        # Compute the matrix M
        M = -P @ C_inv @ D_x @ P

        # Normalize matrix with h
        normalized_M = h * M

        # Compute eigenvalues
        eigenvalues = eigvals(normalized_M)

        # Compute the spectral radius
        spectral_radius = np.max(np.abs(eigenvalues))

        # Compute the maximum CFL number
        cfl_number = rk4_stability_limit / spectral_radius

        # Print spectral radius and CFL number
        print(f"Grid size m = {m}")
        print(f"Spectral radius: {spectral_radius:.6f}")
        print(f"Maximum CFL number: {cfl_number:.4f}\n")

        # Plot eigenvalues in the complex plane
        plt.figure(figsize=(8, 6))
        plt.scatter(eigenvalues.real, eigenvalues.imag, label=f"m = {m}", marker='o')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.title(f"Eigenvalues of Normalized Matrix ({option}, m = {m})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"eigenvalues_{option}_7th_order_m_{m}.jpg", format="jpg", dpi=300)
        plt.show()

# Run the computation
compute_eigenvalues_and_cfl_for_dirichlet(option="Dirichlet")
compute_eigenvalues_and_cfl_for_dirichlet(option="Characteristics")
