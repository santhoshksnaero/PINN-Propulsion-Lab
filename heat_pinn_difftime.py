import sys
sys.path.append(r'E:\Stark_Packages')

import deepxde as dde
import numpy as np

# 1. Define the Physics (1D Heat Equation)
def heat_equation(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1) # Derivative w.r.t time
    dy_xx = dde.grad.hessian(y, x, i=0, j=0) # Second derivative w.r.t space
    return dy_t - dy_xx

# 2. Define the Geometry (The "Engine" part)
geom = dde.geometry.Interval(-1, 3)
timedomain = dde.geometry.TimeDomain(0, 3)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 3. Define the AI Model
data = dde.data.TimePDE(geomtime, heat_equation, [], num_domain=400)
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# 4. Train the "Brain"
model.compile("adam", lr=0.001)
model.train(iterations=1000)

print("Stark Simulation Complete.")

import matplotlib.pyplot as plt

# 1. Create a function to get predictions at a specific time
def get_time_snapshot(model, time_value):
    x_space = np.linspace(-1, 3, 100).reshape(-1, 1)
    t_space = np.ones((100, 1)) * time_value
    test_points = np.hstack((x_space, t_space))
    return x_space, model.predict(test_points)

# 2. Grab snapshots from your trained model
x_vals, y_t01 = get_time_snapshot(model, 1) # Early stage
_, y_t05 = get_time_snapshot(model, 2)      # Mid stage
_, y_t10 = get_time_snapshot(model, 3)      # Final stage

# 3. Create the "Time Evolution" Plot
plt.figure(figsize=(10, 6))

plt.plot(x_vals, y_t01, label="Time = 1s (Start)", color='green', linestyle=':')
plt.plot(x_vals, y_t05, label="Time = 2s (Middle)", color='orange', linestyle='--')
plt.plot(x_vals, y_t10, label="Time = 3s (End)", color='red', linewidth=2)

plt.xlabel("Position (x)")
plt.ylabel("Temperature (y)")
plt.title("Stark Industries: Heat Dissipation Over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
