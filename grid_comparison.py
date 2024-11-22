import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
import operators as ops

def compute_eigenvalues_for_grids():
    
    grids = [51, 101]
    xl = 0
    xr = 1
    c = 1

    for m in grids:
        h = (xr - xl) / (m - 1)

        H, HI, D_plus, D_minus, e_l, e_r = ops.sbp_upwind_7th(m, h)

        Dx = np.block([[np.zeros((m, m)), D_plus],
                       [D_minus, np.zeros((m, m))]])
        M = -c * Dx

        normalized_M = h * M

        eigenvalues = eigvals(normalized_M)

        plt.figure(figsize=(8, 6))
        plt.scatter(eigenvalues.real, eigenvalues.imag, label=f"m = {m}", marker='o')
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.title(f"Eigenvalues of Normalized Matrix (m = {m})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"eigenvalues_7th_order_m_{m}.jpg", format="jpg", dpi=300)
        plt.show()

compute_eigenvalues_for_grids()
