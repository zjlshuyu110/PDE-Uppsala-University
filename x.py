import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("numerical_solution_6th_m_801.csv")
plt.plot(df["x"], df["Pressure (p)"])
plt.show()
