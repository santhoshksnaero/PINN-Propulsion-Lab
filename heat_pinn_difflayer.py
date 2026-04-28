import sys
sys.path.append(r'E:\Stark_Packages')
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# --- 1. THE STARK LAB FUNCTION ---
def train_with_brain(layer_config, label):
    def heat_equation(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - dy_xx

    geom = dde.geometry.Interval(-1, 3)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    data = dde.data.TimePDE(geomtime, heat_equation, [], num_domain=400)
    
    # We pass the custom layer_config here
    net = dde.nn.FNN(layer_config, "tanh", "Glorot normal")
    model = dde.Model(data, net)
    
    model.compile("adam", lr=0.001)
    model.train(iterations=1000)
    
    # Predict for plotting
    x_test = np.linspace(-1, 3, 100).reshape(-1, 1)
    t_test = np.ones((100, 1)) * 1.0
    test_points = np.hstack((x_test, t_test))
    return x_test, model.predict(test_points)

# --- 2. EXPERIMENT WITH 3 DIFFERENT "BRAINS" ---

# A: The "Small" Brain (Too simple?)
print("\n--- Training: Small Brain (2 neurons per layer) ---")
x_small, y_small = train_with_brain([2] + [2]*3 + [1], "Small")

# B: The "Standard" Brain (Your original)
print("\n--- Training: Standard Brain (20 neurons per layer) ---")
x_std, y_std = train_with_brain([2] + [20]*3 + [1], "Standard")

# C: The "Deep" Brain (10 layers)
print("\n--- Training: Deep Brain (10 layers) ---")
x_deep, y_deep = train_with_brain([2] + [20]*10 + [1], "Deep")

print("\n--- Training: Deep Brain (100 layers) ---")
x_deep, y_deep1 = train_with_brain([2] + [20]*100 + [1], "Deep1")

# --- 3. COMPARE THE THINKING POWER ---
plt.figure(figsize=(10, 6))
plt.plot(x_small, y_small, label="Small (2x3): Underpowered", color='gray', linestyle=':')
plt.plot(x_std, y_std, label="Standard (20x3): Balanced", color='blue', linewidth=2)
plt.plot(x_deep, y_deep, label="Deep (20x10): High Capacity", color='purple', linewidth=2)
plt.plot(x_deep, y_deep1, label="Deep (20x100): High Capacity1", color='black', linewidth=2)

plt.xlabel("Position (x)")
plt.ylabel("Temperature (y)")
plt.title("Stark Lab: Brain Complexity vs. Physics Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
