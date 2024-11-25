import numpy as np
import pandas as pd

def compute_convergence_rate():

    grid_sizes = [101, 201, 401, 601, 801]
    l2_norm_errors = [0.357200, 0.220653, 0.130770, 0.133153, 0.151892]

    convergence_rates = []

    for i in range(len(l2_norm_errors) - 1):
        q = np.log10(l2_norm_errors[i] / l2_norm_errors[i + 1]) / np.log10(grid_sizes[i + 1] / grid_sizes[i])
        convergence_rates.append(q)

    data = {"Grid Size (m)": grid_sizes, "L2-Norm of Error": l2_norm_errors, "Convergence Rate (q)": convergence_rates + ["-"]}

    df = pd.DataFrame(data)
    df.to_csv("convergence_rates_6th.csv", index = False)

    print(df)

compute_convergence_rate()
