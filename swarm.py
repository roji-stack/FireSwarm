import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync, str2bool

class CustomPIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target_val, current_val, current_vel, dt):
        error = target_val - current_val
        self.integral += error * dt
        
        # Derivative term uses the actual measured velocity (current_vel)
        # Since target velocity is 0, derivative of error is -current_vel.
        derivative = -current_vel
        
        self.prev_error = error
        
        # Compute control output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output, error

def calculate_repulsion(pos_i, pos_j, d_0, k_rep):
    # Calculate Euclidean distance using only X and Y coordinates to prevent altitude dodging
    pi = np.array(pos_i[:2])
    pj = np.array(pos_j[:2])
    diff = pi - pj
    d = np.linalg.norm(diff)
    if 0 < d < d_0:
        repulsive_magnitude = k_rep * (1.0 / d - 1.0 / d_0) * (1.0 / (d ** 2))
        return repulsive_magnitude * (diff / d)
    else:
        return np.zeros(2)

def run_simulation(kp_z, kd_z, kp_xy, kd_xy, kp_att, kd_att, duration_sec=15.0, plot_filename="swarm_3d_tracking.png", gui=True, d_0=0.6, k_rep=0.5):
    ctrl_freq = 240
    pyb_freq = 240
    num_drones = 3
    
    # Spawn the drones 1 meter apart along the Y-axis so they do not collide on initialization
    initial_xyzs = np.array([
        [0.0, 0.0, 0.1],
        [0.0, 1.0, 0.1],
        [0.0, -1.0, 0.1]
    ])
    
    # Initialize the 3-drone environment
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=num_drones,
        initial_xyzs=initial_xyzs,
        initial_rpys=np.zeros((num_drones, 3)),
        physics=Physics.PYB,
        pyb_freq=pyb_freq,
        ctrl_freq=ctrl_freq,
        gui=gui,
        obstacles=False,
        user_debug_gui=False
    )
    
    # Set up controllers for each of the 3 drones (Ki = 0.0)
    pids_z = [CustomPIDController(kp=kp_z, ki=0.0, kd=kd_z) for _ in range(num_drones)]
    pids_x = [CustomPIDController(kp=kp_xy, ki=0.0, kd=kd_xy) for _ in range(num_drones)]
    pids_y = [CustomPIDController(kp=kp_xy, ki=0.0, kd=kd_xy) for _ in range(num_drones)]
    
    pids_roll = [CustomPIDController(kp=kp_att, ki=0.0, kd=kd_att) for _ in range(num_drones)]
    pids_pitch = [CustomPIDController(kp=kp_att, ki=0.0, kd=kd_att) for _ in range(num_drones)]
    pids_yaw = [CustomPIDController(kp=kp_att, ki=0.0, kd=kd_att) for _ in range(num_drones)]
    
    # Track variables for plotting (for each drone)
    time_points = []
    positions = [[] for _ in range(num_drones)]
    target_positions = [[] for _ in range(num_drones)]
    rolls = [[] for _ in range(num_drones)]
    pitches = [[] for _ in range(num_drones)]
    yaws = [[] for _ in range(num_drones)]
    
    obs, info = env.reset()
    
    num_steps = int(duration_sec * ctrl_freq)
    dt = 1.0 / ctrl_freq
    
    # Physical Constants
    KF = env.KF
    KM = env.KM
    L = env.L
    GRAVITY = env.GRAVITY
    MAX_RPM = env.MAX_RPM
    Jxx, Jyy, Jzz = env.J[0, 0], env.J[1, 1], env.J[2, 2]
    
    if gui:
        start_time = time.time()
        
    for step in range(num_steps):
        t = step * dt
        action = np.zeros((num_drones, 4))
        
        # Leader's actual current position (Drone 0)
        leader_state = obs[0]
        leader_x = leader_state[0]
        leader_y = leader_state[1]
        
        for i in range(num_drones):
            # 1. Swarm Path Planning
            if i == 0:
                # Leader: Program a sharp, high-speed figure-8 trajectory
                tx = 1.5 * np.sin(0.8 * t)
                ty = 1.5 * np.sin(1.6 * t)
                tz = 2.0
            elif i == 1:
                # Follower 1: Offset to the back-left of Leader's actual position
                tx = leader_x - 0.8
                ty = leader_y + 0.8
                tz = 2.0
            elif i == 2:
                # Follower 2: Offset to the back-right of Leader's actual position
                tx = leader_x - 0.8
                ty = leader_y - 0.8
                tz = 2.0
                
            # APF Collision Avoidance
            total_repulsion = np.zeros(2)
            pos_i = obs[i][:3]
            for j in range(num_drones):
                if j != i:
                    pos_j = obs[j][:3]
                    total_repulsion += calculate_repulsion(pos_i, pos_j, d_0, k_rep)
            tx += total_repulsion[0]
            ty += total_repulsion[1]
                
            # 2. Extract state vector
            state = obs[i]
            current_x, current_y, current_z = state[0], state[1], state[2]
            current_vx, current_vy, current_vz = state[10], state[11], state[12]
            current_roll, current_pitch, current_yaw = state[7], state[8], state[9]
            current_wx, current_wy, current_wz = state[13], state[14], state[15]
            
            # 3. Outer loop position controllers
            # Compute Z control force (thrust)
            pid_force_z, error_z = pids_z[i].compute(tz, current_z, current_vz, dt)
            
            # X position controller -> target pitch
            target_pitch, error_x = pids_x[i].compute(tx, current_x, current_vx, dt)
            
            # Y position controller -> target roll
            # Note: positive roll yields negative thrust along Y, so we invert the output
            pid_out_y, error_y = pids_y[i].compute(ty, current_y, current_vy, dt)
            target_roll = -pid_out_y
            
            # Clamp target pitch and roll to +/- 0.4 radians
            target_pitch = np.clip(target_pitch, -0.4, 0.4)
            target_roll = np.clip(target_roll, -0.4, 0.4)
            
            # Apply tilt compensation to Z thrust to maintain altitude
            cos_pitch = np.cos(target_pitch)
            cos_roll = np.cos(target_roll)
            tilt_comp = cos_pitch * cos_roll
            # Ensure we don't divide by zero
            if tilt_comp < 0.1:
                tilt_comp = 0.1
                
            total_thrust = (GRAVITY + pid_force_z) / tilt_comp
            total_thrust = max(0.0, total_thrust)
            
            # 4. Inner loop attitude control outputs (angular accelerations in rad/s^2)
            roll_out, error_roll = pids_roll[i].compute(target_roll, current_roll, current_wx, dt)
            pitch_out, error_pitch = pids_pitch[i].compute(target_pitch, current_pitch, current_wy, dt)
            yaw_out, error_yaw = pids_yaw[i].compute(0.0, current_yaw, current_wz, dt)
            
            # Convert angular accelerations to torques (N*m) using inertia
            tau_x = Jxx * roll_out
            tau_y = Jyy * pitch_out
            tau_z = Jzz * yaw_out
            
            # 5. Quadcopter X-configuration mixing formulas for individual motor forces
            F_0 = total_thrust / 4.0 - tau_x * np.sqrt(2) / (4.0 * L) - tau_y * np.sqrt(2) / (4.0 * L) - tau_z * (KF / (4.0 * KM))
            F_1 = total_thrust / 4.0 - tau_x * np.sqrt(2) / (4.0 * L) + tau_y * np.sqrt(2) / (4.0 * L) + tau_z * (KF / (4.0 * KM))
            F_2 = total_thrust / 4.0 + tau_x * np.sqrt(2) / (4.0 * L) + tau_y * np.sqrt(2) / (4.0 * L) - tau_z * (KF / (4.0 * KM))
            F_3 = total_thrust / 4.0 + tau_x * np.sqrt(2) / (4.0 * L) - tau_y * np.sqrt(2) / (4.0 * L) + tau_z * (KF / (4.0 * KM))
            
            # Convert motor forces to RPM: RPM = sqrt(F / KF)
            motor_rpms = np.zeros(4)
            motor_rpms[0] = np.sqrt(max(0.0, F_0) / KF)
            motor_rpms[1] = np.sqrt(max(0.0, F_1) / KF)
            motor_rpms[2] = np.sqrt(max(0.0, F_2) / KF)
            motor_rpms[3] = np.sqrt(max(0.0, F_3) / KF)
            
            # Action space expects shape (num_drones, 4)
            action[i, :] = np.clip(motor_rpms, 0, MAX_RPM)
            
            # Log data
            positions[i].append([current_x, current_y, current_z])
            target_positions[i].append([tx, ty, tz])
            rolls[i].append(current_roll)
            pitches[i].append(current_pitch)
            yaws[i].append(current_yaw)
            
        time_points.append(t)
        
        # Step simulation using physical commands directly
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Sync with wall-clock for visualization in GUI mode
        if gui:
            sync(step, start_time, dt)
            
    env.close()
    
    positions = np.array(positions)           # shape (num_drones, num_steps, 3)
    target_positions = np.array(target_positions) # shape (num_drones, num_steps, 3)
    time_points = np.array(time_points)
    
    # Generate and save a premium, comprehensive 2x2 multi-panel plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#1f77b4', '#2ca02c', '#9467bd'] # Blue, Green, Purple
    drone_names = ["Leader (Drone 0)", "Follower 1 (Drone 1)", "Follower 2 (Drone 2)"]
    
    # X translation subplot
    for i in range(num_drones):
        axs[0, 0].plot(time_points, positions[i, :, 0], label=f'{drone_names[i]} Actual', color=colors[i], linewidth=2.5)
        axs[0, 0].plot(time_points, target_positions[i, :, 0], label=f'{drone_names[i]} Target', color=colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
    axs[0, 0].set_title("X Translation (Forward/Back)", fontsize=12, fontweight='semibold')
    axs[0, 0].set_ylabel("Position (meters)", fontsize=10)
    axs[0, 0].grid(True, which='both', linestyle=':', alpha=0.5)
    axs[0, 0].legend(loc='lower left', fontsize=9)
    
    # Y translation subplot
    for i in range(num_drones):
        axs[0, 1].plot(time_points, positions[i, :, 1], label=f'{drone_names[i]} Actual', color=colors[i], linewidth=2.5)
        axs[0, 1].plot(time_points, target_positions[i, :, 1], label=f'{drone_names[i]} Target', color=colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
    axs[0, 1].set_title("Y Translation (Lateral)", fontsize=12, fontweight='semibold')
    axs[0, 1].set_ylabel("Position (meters)", fontsize=10)
    axs[0, 1].grid(True, which='both', linestyle=':', alpha=0.5)
    axs[0, 1].legend(loc='lower left', fontsize=9)
    
    # Z altitude subplot
    for i in range(num_drones):
        axs[1, 0].plot(time_points, positions[i, :, 2], label=f'{drone_names[i]} Actual', color=colors[i], linewidth=2.5)
        axs[1, 0].plot(time_points, target_positions[i, :, 2], label=f'{drone_names[i]} Target', color=colors[i], linestyle='--', linewidth=1.5, alpha=0.7)
    axs[1, 0].set_title("Z Translation (Altitude)", fontsize=12, fontweight='semibold')
    axs[1, 0].set_xlabel("Time (seconds)", fontsize=10)
    axs[1, 0].set_ylabel("Position (meters)", fontsize=10)
    axs[1, 0].grid(True, which='both', linestyle=':', alpha=0.5)
    axs[1, 0].legend(loc='lower left', fontsize=9)
    
    # XY trajectory subplot
    for i in range(num_drones):
        axs[1, 1].plot(positions[i, :, 0], positions[i, :, 1], label=drone_names[i], color=colors[i], linewidth=2.5)
        axs[1, 1].scatter(positions[i, 0, 0], positions[i, 0, 1], marker='o', color=colors[i], s=50, zorder=5, label='_nolegend_')
        axs[1, 1].scatter(positions[i, -1, 0], positions[i, -1, 1], marker='x', color=colors[i], s=50, zorder=5, label='_nolegend_')
    axs[1, 1].set_title("XY Flight Path Trajectory", fontsize=12, fontweight='semibold')
    axs[1, 1].set_xlabel("X Position (meters)", fontsize=10)
    axs[1, 1].set_ylabel("Y Position (meters)", fontsize=10)
    axs[1, 1].grid(True, which='both', linestyle=':', alpha=0.5)
    axs[1, 1].legend(loc='upper right', fontsize=9)
    axs[1, 1].axis('equal')
    
    fig.suptitle(f"6 DoF Swarm V-Formation Trajectory Tracking\n(Outer Kp_xy={kp_xy}, Inner Kp_att={kp_att})", fontsize=14, fontweight='bold', color='#333333')
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print("\n--- Final Simulation Metrics ---")
    for i in range(num_drones):
        final_pos = positions[i, -1, :]
        target_pos = target_positions[i, -1, :]
        error = np.linalg.norm(final_pos - target_pos)
        print(f"\n{drone_names[i]}:")
        print(f"  Final Position: X={final_pos[0]:.3f} m, Y={final_pos[1]:.3f} m, Z={final_pos[2]:.3f} m")
        print(f"  Final Target  : X={target_pos[0]:.3f} m, Y={target_pos[1]:.3f} m, Z={target_pos[2]:.3f} m")
        print(f"  Tracking Error: {error:.3f} m")
        
    # Calculate minimum distance between any pair of drones during flight
    min_dist = float('inf')
    min_dist_step = -1
    min_dist_pair = None
    for step in range(num_steps):
        pos0 = positions[0, step, :2]
        pos1 = positions[1, step, :2]
        pos2 = positions[2, step, :2]
        
        d01 = np.linalg.norm(pos0 - pos1)
        d02 = np.linalg.norm(pos0 - pos2)
        d12 = np.linalg.norm(pos1 - pos2)
        
        if d01 < min_dist:
            min_dist = d01
            min_dist_step = step
            min_dist_pair = (0, 1)
        if d02 < min_dist:
            min_dist = d02
            min_dist_step = step
            min_dist_pair = (0, 2)
        if d12 < min_dist:
            min_dist = d12
            min_dist_step = step
            min_dist_pair = (1, 2)
            
    print(f"\nMinimum XY distance between any pair of drones during flight: {min_dist:.3f} meters (Pair {min_dist_pair} at step {min_dist_step}, t={min_dist_step*dt:.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fly a drone swarm in 3D using outer-loop position and inner-loop attitude controllers")
    parser.add_argument("--kp_z", type=float, default=1.2, help="Z Proportional gain")
    parser.add_argument("--kd_z", type=float, default=0.8, help="Z Derivative gain")
    parser.add_argument("--kp_xy", type=float, default=0.2, help="X/Y Proportional gain")
    parser.add_argument("--kd_xy", type=float, default=0.15, help="X/Y Derivative gain")
    parser.add_argument("--kp_att", type=float, default=20.0, help="Attitude Proportional gain (scaled)")
    parser.add_argument("--kd_att", type=float, default=5.0, help="Attitude Derivative gain (scaled)")
    parser.add_argument("--duration", type=float, default=15.0, help="Simulation duration in seconds")
    parser.add_argument("--output", type=str, default="swarm_3d_tracking.png", help="Plot filename")
    parser.add_argument("--gui", type=str2bool, default=True, help="Whether to use PyBullet GUI")
    parser.add_argument("--d_0", type=float, default=0.6, help="APF safe distance threshold (meters)")
    parser.add_argument("--k_rep", type=float, default=0.5, help="APF repulsive force gain")
    args = parser.parse_args()
    
    print("Starting Multi-Drone 3D V-Formation Simulation...")
    print(f"  Gains: Kp_z={args.kp_z}, Kd_z={args.kd_z}")
    print(f"         Kp_xy={args.kp_xy}, Kd_xy={args.kd_xy}")
    print(f"         Kp_att={args.kp_att}, Kd_att={args.kd_att}")
    print(f"  APF  : d_0={args.d_0}, k_rep={args.k_rep}")
    print(f"  GUI: {args.gui}, Duration: {args.duration}s")
    
    run_simulation(
        kp_z=args.kp_z,
        kd_z=args.kd_z,
        kp_xy=args.kp_xy,
        kd_xy=args.kd_xy,
        kp_att=args.kp_att,
        kd_att=args.kd_att,
        duration_sec=args.duration,
        plot_filename=args.output,
        gui=args.gui,
        d_0=args.d_0,
        k_rep=args.k_rep
    )
