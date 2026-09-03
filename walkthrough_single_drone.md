# Drone Swarm Migration - Control and Autotuning Walkthrough

We have successfully migrated the drone swarm simulation to a headless, code-driven architecture, implemented a custom PID altitude controller, visually verified the response, and verified the system's tracking behavior under a dynamic setpoint.

## Phase 1: Environment Setup

- Successfully configured a Python 3.10 virtual environment (`venv`).
- Cloned the `learnsyslab/gym-pybullet-drones` repository.
- Successfully built and installed `pybullet` from source by patching a macro redefine issue in `zutil.h` (resolved conflicting `fdopen` macro redefine with the macOS SDK).
- Installed `gym-pybullet-drones` in editable mode.
- Confirmed library import works correctly inside the virtual environment.

## Phase 2: Base Simulation & PID Implementation

Implemented a custom PID controller in `fly.py`.
- **Constraint Compliance**: Did not use the built-in `DSLPIDControl` class.
- **Damping Loop**: Used the measured vertical velocity from the physics simulator (`state[12]`) as the D-term feedback for superior stability.
- **Gains**: Started with $K_p=1.2$ and $K_d=0.8$.
- **Gravity Feedforward**: Pre-compensated for gravity using `env.GRAVITY = env.G * env.M` (approx. $0.2646\text{ N}$) to keep the system linear.

## Phase 3: Visual Verification & Autonomous Tuning

When evaluating the step response with the initial gains ($K_p=1.2, K_d=0.8$) at a target altitude of $2.0\text{ m}$, the drone exhibited a highly stable, heavily overdamped response ($\zeta \approx 2.22$):
- **Rise time (10%-90%)**: $1.417\text{ s}$
- **Settling time (5% band)**: $1.988\text{ s}$
- **Overshoot**: $0.0\%$
- **Oscillations**: $0$

To optimize and speed up the response while ensuring a damped, stable ascent, we tuned the gains using analytical calculations for critical damping ($\zeta = 1.0$) under the Crazyflie mass parameters ($0.027\text{ kg}$). The optimal critically damped gain relation was found to be $K_d = 2 \sqrt{K_p \cdot M} \approx 0.328 \sqrt{K_p}$.

For $K_p = 1.2$, the optimal critically damped $K_d$ is $0.36$. Running this optimized configuration yielded:
- **Rise time (10%-90%)**: $0.621\text{ s}$ (more than 2x faster!)
- **Settling time (5% band)**: $0.883\text{ s}$ (more than 2x faster!)
- **Overshoot**: $0.0\%$
- **Oscillations**: $0$

### Baseline Tuning Plot (Kp=1.20, Kd=0.36)
![Altitude Step Response (Kp=1.20, Kd=0.36)](/Users/ryanjin/.gemini/antigravity/brain/ce92cf52-63be-42f8-af94-475ba0f1e0c0/altitude_plot.png)

## Phase 4: Dynamic Setpoint Tracking Test

Using the optimal tuned gains ($K_p=1.2, K_d=0.36$), we tested a dynamic target change mid-flight. The target altitude started at $2\text{ m}$ and changed to $5\text{ m}$ at $5.0\text{ s}$.

The tuned gains handled the transition flawlessly:
- **Overshoot on transition**: $0.0\%$
- **Settling time after step**: $0.858\text{ s}$ (at $t = 5.858\text{ s}$)
- **Oscillations**: $0$
- **Stability**: Flatlined exactly at the target altitudes with zero steady-state error.

### Dynamic Setpoint Tracking Plot (Kp=1.20, Kd=0.36)
![Dynamic Altitude Step Response (Kp=1.20, Kd=0.36)](/Users/ryanjin/.gemini/antigravity/brain/ce92cf52-63be-42f8-af94-475ba0f1e0c0/step_response_tracking.png)

## Final Tuned Parameters

- **Proportional Gain ($K_p$)**: $1.20$
- **Integral Gain ($K_i$)**: $0.00$ (exactly zero as constrained)
- **Derivative Gain ($K_d$)**: $0.36$ (critically damped)
