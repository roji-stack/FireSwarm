# Handover Sign-off: Z-Axis PID Control Swarm Simulation

This sign-off document is prepared for future agents to quickly orient themselves with the current progress of the project and proceed with the next milestones.

---

## 1. Project Context & Status

We have successfully migrated the drone simulation to a headless, code-driven architecture using `gym-pybullet-drones` (Gymnasium environment based on PyBullet) and developed a custom Z-axis PID altitude hold controller.

- **Status**: The altitude hold controller has been scaled to a 3-drone swarm. Each drone tracks its own target altitude independently.
- **Damping**: The system uses actual measured vertical velocity (`state[12]`) as the derivative feedback term instead of numerical differentiation of error.
- **Tuning**: The gains are tuned for critical damping ($\zeta = 1.0$) based on the Crazyflie mass parameter ($0.027\text{ kg}$).
- **Parameters**: 
  - $K_p = 1.20$
  - $K_d = 0.36$
  - $K_i = 0.00$
  - Gravity pre-compensation (feedforward) is implemented using `env.GRAVITY = env.G * env.M` (approx. $0.2646\text{ N}$).

---

## 2. Key Files

All development occurs within the workspace directory: [DroneSwarm](file:///Users/ryanjin/Documents/DroneSwarm)

- **[fly.py](file:///Users/ryanjin/Documents/DroneSwarm/fly.py)**: Single-drone baseline simulation with static and dynamic altitude tracking.
- **[swarm.py](file:///Users/ryanjin/Documents/DroneSwarm/swarm.py)**: Multi-drone simulation for a 3-drone swarm tracking different setpoints.
  - Command to run static swarm target hold:
    ```bash
    ./venv/bin/python swarm.py --output swarm_plot.png
    ```
  - Command to run dynamic swarm target hold (targets step up by $1.0\text{ m}$ at $t = 5.0\text{ s}$):
    ```bash
    ./venv/bin/python swarm.py --dynamic --output swarm_dynamic_plot.png
    ```
- **Plots**:
  - [swarm_plot.png](file:///Users/ryanjin/.gemini/antigravity/brain/c746130e-bdd6-4b56-8407-42bf0fa119da/swarm_plot.png): Static target response.
  - [swarm_dynamic_plot.png](file:///Users/ryanjin/.gemini/antigravity/brain/c746130e-bdd6-4b56-8407-42bf0fa119da/swarm_dynamic_plot.png): Dynamic target step response.

---

## 3. Recommended Next Milestones

1. **Horizontal Position Control (X-Y plane)**: Implement X-Y controllers (PID or LQR) so drones can move dynamically to different coordinates or fly in formation.
2. **Collision Avoidance**: Integrate artificial potential fields or simple distance-based repulsion forces between drones to prevent collisions when paths cross.
3. **Obstacle Navigation**: Enable collision checking with static environment objects.
