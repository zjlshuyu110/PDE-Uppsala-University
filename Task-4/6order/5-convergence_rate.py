import numpy as np
import pandas as pd

def compute_convergence_rate():

    df = pd.read_csv("error_analysis.csv")
    grid_sizes = df["Grid Size (m)"]
    l2_norm_errors = df["L2-Norm of Error"]

    convergence_rates = []

    for i in range(len(l2_norm_errors) - 1):
        q = np.log10(l2_norm_errors[i] / l2_norm_errors[i + 1]) / np.log10(grid_sizes[i + 1] / grid_sizes[i])
        convergence_rates.append(q)

    data = {"Grid Size (m)": grid_sizes, "L2-Norm of Error": l2_norm_errors, "Convergence Rate (q)": convergence_rates + ["-"]}

    df = pd.DataFrame(data)
    df.to_csv("convergence_rates_6th.csv", index = False)

    print(df)

compute_convergence_rate()
