import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
import operators as ops  # Importing the provided SBP operators

def compute_eigenvalues_and_cfl_for_dirichlet():

    # Parameters
    grids = [51, 101]
    xl = 0  
    xr = 1 
    rho = 1
    c = 1
    rk4_stability_limit = 2.8 


    for m in grids:
        h = (xr - xl) / (m - 1) 

        # Retrieve 7th-order SBP operators
        H, HI, D_plus, D_minus, e_l, e_r = ops.sbp_upwind_7th(m, h)

        # Construct the 1st-order derivative operator D_x
        D_x = np.block([
            [np.zeros((m, m)), D_plus],
            [D_minus, np.zeros((m, m))]
        ])

        # Construct the inverse of the SBP operator C
        C_inv = np.block([
            [np.eye(m) / (rho * c**2), np.zeros((m, m))],
            [np.zeros((m, m)), np.eye(m) / rho]
        ])
        
        # Define the projection matrix P for Dirichlet BCs
        P = np.eye(2 * m)
        # Enforce v = 0 at the left and right boundaries
        P[m, :] = 0  
        P[2 * m - 1, :] = 0

        M = -P @ C_inv @ D_x @ P

        normalized_M = h * M
        eigenvalues = eigvals(normalized_M)

        spectral_radius = np.max(np.abs(eigenvalues))

        cfl_number = rk4_stability_limit / spectral_radius

        print(f"Grid size m = {m}")
        print(f"Spectral radius: {spectral_radius:.6f}")
        print(f"Maximum CFL number: {cfl_number:.4f}\n")

        plt.figure(figsize=(8, 6))
        plt.scatter(eigenvalues.real, eigenvalues.imag, label=f"m = {m}", marker='o')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.title(f"Eigenvalues of Normalized Matrix (m = {m})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"eigenvalues_dirichlet_7th_order_m_{m}.jpg", format="jpg", dpi=300)
        plt.show()

compute_eigenvalues_and_cfl_for_dirichlet()
