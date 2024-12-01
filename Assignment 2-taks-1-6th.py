from dataclasses import dataclass
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from sbp.operators import sbp_cent_6th

# Set up parameters
x_l, x_r = -2, 4  # Domain limits
m = 201  # Grid points
h = (x_r - x_l) / (m - 1)  # Grid spacing
x = np.linspace(x_l, x_r, m)  # Grid

# Material properties
rho1, c1 = 1, 1
rho2, c2 = 2, 2
Z1 = rho1 * c1
Z2 = rho2 * c2
T_theory = 2 * Z2 / (Z1 + Z2)
R_theory = (Z2 - Z1) / (Z1 + Z2)

# Initial Gaussian profile
r_star = 0.1
x0 = 0
u0 = np.exp(-(x - x0) ** 2 / (2 * r_star ** 2))
v0 = np.zeros_like(u0)  # Initial velocity

# SBP operator
H, HI, D1, D2, e_l, e_r, d1_l, d1_r = sbp_cent_6th(m, h)

c = np.where(x <= 1, c1, c2)  # Speed of sound varies by medium
dt = 0.8 * h / max(c1, c2)  # CFL condition
t_max = 2.5
t_steps = int(t_max / dt)

# Time stepping (Leapfrog scheme)
u = u0.copy()
v = v0.copy()
for t in range(t_steps):
    u_new = u + dt * v
    v_new = v + dt * (c ** 2) * (D1 @ (D1 @ u))
    u, v = u_new, v_new

# Plot results at t_max
plt.figure(figsize=(10, 6))
plt.plot(x, u, label="Numerical Solution")
plt.axvline(1, color='k', linestyle='--', label="Interface at x=1")
plt.title(f"Wave propagation at t={t_max}")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.legend()
plt.grid()
plt.show()

# Calculate reflection and transmission coefficients
A_r = max(u[x < 1])
A_t = max(u[x > 1])
T_num = A_t / max(u0)
R_num = A_r / max(u0)

# Print results
print(f"Theoretical Transmission Coefficient (T): {T_theory:.3f}")
print(f"Numerical Transmission Coefficient (T): {T_num:.3f}")
print(f"Theoretical Reflection Coefficient (R): {R_theory:.3f}")
print(f"Numerical Reflection Coefficient (R): {R_num:.3f}")

def plot_animated_graph(u, v):
    fig, ax = plt.subplots()
    (line,) = ax.plot(x, u)

    @dataclass
    class ParamsForUpdate:
        u: np.ndarray
        v: np.ndarray

    def update(frame, data: ParamsForUpdate):
        u_new = data.u + dt * data.v
        v_new = data.v + dt * (c**2) * (D1 @ (D1 @ data.u))
        data.u, data.v = u_new, v_new
        line.set_ydata(data.u)
        ax.set_title(f"t={frame}")
        if frame >= t_steps - 1:
            ani.event_source.stop()
        return (line,)

    uv = ParamsForUpdate(u, v)
    ani = FuncAnimation(
        fig,  # Figure to animate
        update,
        fargs=(uv,),  # Update function
        frames=t_steps,  # Number of frames
        interval=int(t_max / t_steps * 1000),  # Delay between frames in milliseconds
    )
    plt.show()


plot_animated_graph(u0, v0)
