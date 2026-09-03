# FireSwarm: Multi-Agent Autonomous Drone Swarm Simulation & Control

FireSwarm is an autonomous multi-quadcopter swarm platform featuring physics-based simulation, custom cascaded PID flight controllers, artificial potential field (APF) collision avoidance, and a distributed ROS 2 architecture.

---

## Key Features

- **Physics-Based Simulation**: Built on PyBullet and `gym-pybullet-drones` simulating realistic Crazyflie 2.x quadcopter aerodynamics and motor dynamics.
- **Cascaded PID Control**:
  - **Outer Loop**: Position controller generating desired pitch/roll angles and total thrust.
  - **Inner Loop (Altitude Hold)**: Critically damped ($\zeta = 1.0$) Z-axis velocity-derivative feedback controller with gravity feedforward compensation.
- **Swarm Collision Avoidance**: Real-time 2D/3D Artificial Potential Field (APF) repulsion forces preventing inter-drone collisions during coordinated formation flight.
- **Distributed ROS 2 Architecture**: Modular ROS 2 workspace (`ros2_swarm_ws`) with dedicated nodes for simulation, trajectory generation, and flight control.
- **Visualization & Telemetry**: Dynamic trajectory tracking plots, step-response telemetry, and 3D visualizers.

---

## Repository Structure

```
.
├── fly.py                  # Single-drone altitude hold baseline simulation
├── fly_3d.py               # Full 3D trajectory tracking and position control
├── swarm.py                # Multi-drone swarm altitude hold & dynamic setpoint tracking
├── sim.py                  # Core simulation primitives
├── portfolio_showcase.py   # Multi-agent 3D swarm demonstration with collision avoidance
├── ros2_swarm_ws/          # ROS 2 colcon workspace
│   └── src/
│       └── swarm_sim/      # ROS 2 package (sim, trajectory, and control nodes)
├── *.png                   # Telemetry, step-response, and 3D flight trajectory plots
├── *.md                    # Architectural sign-off and verification walkthroughs
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyBullet 3.2+
- `gym-pybullet-drones`
- ROS 2 (Humble / Iron / Rolling) [Optional, for ROS 2 package]

### Installation

```bash
git clone https://github.com/roji-stack/FireSwarm.git
cd FireSwarm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pybullet gymnasium matplotlib scipy
pip install --upgrade git+https://github.com/learnsyslab/gym-pybullet-drones.git
```

---

## Running Simulations

### 1. Multi-Agent 3D Swarm Showcase
Runs a coordinated 3-drone swarm with APF collision avoidance and generates trajectory plots:
```bash
python portfolio_showcase.py --gui false --output swarm_3d_tracking.png
```

### 2. Multi-Drone Altitude Swarm Hold
Static target hold:
```bash
python swarm.py --output swarm_plot.png
```
Dynamic step setpoint tracking:
```bash
python swarm.py --dynamic --output swarm_dynamic_plot.png
```

### 3. Single Drone 3D Flight
```bash
python fly_3d.py --output full_3d_flight_response.png
```

### 4. ROS 2 Swarm Simulation
```bash
cd ros2_swarm_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch swarm_sim swarm.launch.py
```
