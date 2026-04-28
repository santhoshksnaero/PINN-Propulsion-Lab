import sys
sys.path.append(r'E:\Stark_Packages')
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SET UP THE PHYSICS WITH VARIABLE ALPHA ---
def run_simulation(alpha_value):
    def heat_equation(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t - alpha_value * dy_xx # Alpha applied here

    geom = dde.geometry.Interval(-1, 3)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    data = dde.data.TimePDE(geomtime, heat_equation, [], num_domain=400)
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    
    model.compile("adam", lr=0.001)
    model.train(iterations=1000)
    
    # Predict results for plotting
    x_test = np.linspace(-1, 3, 100).reshape(-1, 1) # Matches geometry -1 to 3
    t_test = np.ones((100, 1)) * 1.0
    test_points = np.hstack((x_test, t_test))
    y_pred = model.predict(test_points)
    
    return x_test, y_pred

# --- 2. RUN FOR TWO DIFFERENT MATERIALS ---
print("--- Training Model 1: Standard Conductor (Alpha = 1.0) ---")
x1, y1 = run_simulation(1.0)

print("\n--- Training Model 2: Slow Conductor (Alpha = 0.1) ---")
x2, y2 = run_simulation(0.1)

# --- 3. PLOT BOTH RESULTS TOGETHER ---
plt.figure(figsize=(10, 6))
plt.plot(x1, y1, label="Alpha = 1.0 (Fast Spread)", color='red', linewidth=2)
plt.plot(x2, y2, label="Alpha = 0.1 (Slow Spread)", color='blue', linewidth=2, linestyle='--')

plt.xlabel("Position (x)")
plt.ylabel("Temperature (y)")
plt.title("Stark Industries: Impact of Thermal Diffusivity (Alpha)")
plt.legend()
plt.grid(True)
plt.show()

print("Comparison Plot Complete.")
