## Project: Aero-Thermal Digital Twin (PINN)

##  Objective
Developing Physics-Informed Neural Networks (PINNs) to simulate heat transfer in propulsion components. This project bridges the gap between **Classical Thermodynamics** and **Deep Learning**.

## Tech Stack
- **Framework:** DeepXDE (TensorFlow backend)
- **Environment:** VS Code / Python 3.10
- **Concepts:** PINNs, Partial Differential Equations (PDEs), Heat Diffusion.

##  Current Milestone: 1D Heat Equation
Successfully modeled 1D heat diffusion along a variable-length rod. 
### Key Findings:
1. **Material Impact:** Simulated the effect of varying Thermal Diffusivity ($\alpha$).
2. **Neural Capacity:** Discovered the "Overfitting Cliff" phenomenon when using high-capacity networks (20x100 neurons) without sufficient regularization.
3. **Optimized Architecture:** Found that a 20x3 "Standard" brain provides the best balance of speed and physics accuracy for 1D problems.

##  Next Steps
- Transition to **1D Nozzle Flow** (Burgers' Equation).
- Implement **Inverse Problem Solving** to predict material properties from sensor data.
