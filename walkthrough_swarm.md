# Drone Swarm Migration - Multi-Agent Z-Axis PID Control Walkthrough

This document summarizes the migration, implementation, and verification of our scaled Z-axis PID control architecture for a multi-agent drone swarm.

## Summary of Changes

We developed the multi-agent control script [swarm.py](file:///Users/ryanjin/Documents/DroneSwarm/swarm.py) using the single-drone baseline from `fly.py`. The simulation configurations and loops were scaled to support a 3-drone swarm.

1. **Multi-Agent Setup**:
   - Spaced the drones 1.0 meter apart along the X-axis (`initial_xyzs = [[0.0, 0.0, 0.1], [1.0, 0.0, 0.1], [2.0, 0.0, 0.1]]`) to prevent collisions on spawn.
   - Initialized 3 independent instances of our custom Z-axis PID controller, maintaining the critically damped gains ($K_p = 1.20$, $K_d = 0.36$, $K_i = 0.00$).

2. **Refactored Array-Based Control Loop**:
   - Extracted independent $Z$ (altitude) and $V_z$ (vertical velocity) states for each drone from the environment's observation array.
   - Assigned distinct targets to each drone:
     - **Drone 0**: $1.5\text{ m}$ (steps to $2.5\text{ m}$ under `--dynamic` test)
     - **Drone 1**: $2.0\text{ m}$ (steps to $3.0\text{ m}$ under `--dynamic` test)
     - **Drone 2**: $2.5\text{ m}$ (steps to $3.5\text{ m}$ under `--dynamic` test)
   - Calculated motor RPM targets independently for each drone, bound actions to a shape of `(3, 4)` representing motor commands for all 3 drones, and stepped the simulation.

3. **Plotting**:
   - Refactored plotting to graph all 3 drones on a single plot using distinct colors (blue, green, purple) and clear labels.

---

## Simulation Verification Results

### Case 1: Static Independent Altitude Hold
We ran the simulation with static setpoints:
```bash
./venv/bin/python swarm.py --output swarm_plot.png
```

#### Performance Metrics:
- **Drone 0 (Target: 1.500 m)**:
  - Max/Final Altitude: $1.500\text{ m}$
  - Overshoot: $0.0\%$
  - Rise Time (10%-90%): $0.600\text{ s}$
  - Settling Time (5% band): $0.829\text{ s}$
- **Drone 1 (Target: 2.000 m)**:
  - Max/Final Altitude: $2.000\text{ m}$
  - Overshoot: $0.0\%$
  - Rise Time (10%-90%): $0.621\text{ s}$
  - Settling Time (5% band): $0.883\text{ s}$
- **Drone 2 (Target: 2.500 m)**:
  - Max/Final Altitude: $2.500\text{ m}$
  - Overshoot: $0.0\%$
  - Rise Time (10%-90%): $0.637\text{ s}$
  - Settling Time (5% band): $0.933\text{ s}$

#### Step Response Plot:
![Swarm Static Target Plot](/Users/ryanjin/.gemini/antigravity/brain/c746130e-bdd6-4b56-8407-42bf0fa119da/swarm_plot.png)

---

### Case 2: Dynamic Mid-Flight Setpoint Tracking
We verified independent setpoint tracking under dynamic step inputs (targets step up by $1.0\text{ m}$ at $t = 5.0\text{ s}$):
```bash
./venv/bin/python swarm.py --dynamic --output swarm_dynamic_plot.png
```

#### Performance Metrics (Settled at Step 2):
- **Drone 0 (Target: 1.500 m -> 2.500 m)**:
  - Max/Final Altitude: $2.500\text{ m}$
  - Overshoot: $0.0\%$
  - Settling Time after transition: $0.617\text{ s}$ (settled at $t = 5.617\text{ s}$)
- **Drone 1 (Target: 2.000 m -> 3.000 m)**:
  - Max/Final Altitude: $3.000\text{ m}$
  - Overshoot: $0.0\%$
  - Settling Time after transition: $0.579\text{ s}$ (settled at $t = 5.579\text{ s}$)
- **Drone 2 (Target: 2.500 m -> 3.500 m)**:
  - Max/Final Altitude: $3.500\text{ m}$
  - Overshoot: $0.0\%$
  - Settling Time after transition: $0.550\text{ s}$ (settled at $t = 5.550\text{ s}$)

#### Dynamic Step Response Plot:
![Swarm Dynamic Target Plot](/Users/ryanjin/.gemini/antigravity/brain/c746130e-bdd6-4b56-8407-42bf0fa119da/swarm_dynamic_plot.png)

---

## Conclusion
The custom Z-axis PID controller successfully scales to a multi-agent drone swarm with independent setpoint tracking. The tuned critically damped parameters ($K_p = 1.20$, $K_d = 0.36$) yield quick, stable rise times (approx. $0.6\text{ s}$), zero overshoot, and zero oscillations across all agents.
