import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

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
        
        # Derivative term uses the actual measured linear velocity (current_vel)
        # Since target velocity is 0, derivative of error is -current_vel.
        derivative = -current_vel
        
        self.prev_error = error
        
        # Compute control output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output, error

def run_simulation(kp_z, kd_z, kp_xy, kd_xy, kp_att, kd_att, target_pos, duration_sec=10.0, plot_filename="full_3d_flight_response.png"):
    ctrl_freq = 240
    pyb_freq = 240
    
    # Initialize the single-drone environment
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.1]]), # start slightly above ground
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=Physics.PYB,
        pyb_freq=pyb_freq,
        ctrl_freq=ctrl_freq,
        gui=False,
        obstacles=False,
        user_debug_gui=False
    )
    
    # Set up controllers
    pid_z = CustomPIDController(kp=kp_z, ki=0.0, kd=kd_z)
    pid_x = CustomPIDController(kp=kp_xy, ki=0.0, kd=kd_xy)
    pid_y = CustomPIDController(kp=kp_xy, ki=0.0, kd=kd_xy)
    
    # Inner attitude loop controllers (Roll, Pitch, Yaw)
    pid_roll = CustomPIDController(kp=kp_att, ki=0.0, kd=kd_att)
    pid_pitch = CustomPIDController(kp=kp_att, ki=0.0, kd=kd_att)
    pid_yaw = CustomPIDController(kp=kp_att, ki=0.0, kd=kd_att)
    
    # Track variables for plotting
    time_points = []
    positions = []
    target_positions = []
    rolls = []
    pitches = []
    yaws = []
    motor_commands = []
    
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
    
    for step in range(num_steps):
        t = step * dt
        
        # Get target positions
        tx, ty, tz = target_pos(t)
        
        # State vector indices (from BaseAviary.py):
        # 0: X, 1: Y, 2: Z
        # 7: roll, 8: pitch, 9: yaw
        # 10: VX, 11: VY, 12: VZ
        # 13: WX, 14: WY, 15: WZ
        state = obs[0]
        current_x = state[0]
        current_y = state[1]
        current_z = state[2]
        
        current_vx = state[10]
        current_vy = state[11]
        current_vz = state[12]
        
        current_roll = state[7]
        current_pitch = state[8]
        current_yaw = state[9]
        
        current_wx = state[13]
        current_wy = state[14]
        current_wz = state[15]
        
        # Compute Z control force (thrust)
        pid_force_z, error_z = pid_z.compute(tz, current_z, current_vz, dt)
        
        # Outer loop position controllers
        # X position controller -> target pitch
        target_pitch, error_x = pid_x.compute(tx, current_x, current_vx, dt)
        
        # Y position controller -> target roll
        # Note: positive roll yields negative thrust along Y, so we invert the output
        pid_out_y, error_y = pid_y.compute(ty, current_y, current_vy, dt)
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
        
        # Inner loop attitude control outputs (angular accelerations in rad/s^2)
        roll_out, error_roll = pid_roll.compute(target_roll, current_roll, current_wx, dt)
        pitch_out, error_pitch = pid_pitch.compute(target_pitch, current_pitch, current_wy, dt)
        yaw_out, error_yaw = pid_yaw.compute(0.0, current_yaw, current_wz, dt)
        
        # Convert angular accelerations to torques (N*m) using inertia
        tau_x = Jxx * roll_out
        tau_y = Jyy * pitch_out
        tau_z = Jzz * yaw_out
        
        # Quadcopter X-configuration mixing formulas for individual motor forces
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
        action = np.array([np.clip(motor_rpms, 0, MAX_RPM)])
        
        # Step simulation using physical commands directly
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log data
        time_points.append(t)
        positions.append([current_x, current_y, current_z])
        target_positions.append([tx, ty, tz])
        rolls.append(current_roll)
        pitches.append(current_pitch)
        yaws.append(current_yaw)
        motor_commands.append(action[0])
        
    env.close()
    
    positions = np.array(positions)
    target_positions = np.array(target_positions)
    time_points = np.array(time_points)
    rolls = np.array(rolls) * (180.0 / np.pi) # convert to degrees
    pitches = np.array(pitches) * (180.0 / np.pi)
    yaws = np.array(yaws) * (180.0 / np.pi)
    motor_commands = np.array(motor_commands)
    
    # Generate and save a premium, comprehensive 2x2 multi-panel plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # X translation subplot
    axs[0, 0].plot(time_points, positions[:, 0], label='Actual X', color='#1f77b4', linewidth=2.5)
    axs[0, 0].plot(time_points, target_positions[:, 0], label='Target X', color='#1f77b4', linestyle='--', linewidth=1.5)
    axs[0, 0].set_title("X Translation (Forward/Back)", fontsize=12, fontweight='semibold')
    axs[0, 0].set_ylabel("Position (meters)", fontsize=10)
    axs[0, 0].grid(True, which='both', linestyle=':', alpha=0.5)
    axs[0, 0].legend(loc='lower right')
    
    # Y translation subplot
    axs[0, 1].plot(time_points, positions[:, 1], label='Actual Y', color='#2ca02c', linewidth=2.5)
    axs[0, 1].plot(time_points, target_positions[:, 1], label='Target Y', color='#2ca02c', linestyle='--', linewidth=1.5)
    axs[0, 1].set_title("Y Translation (Lateral)", fontsize=12, fontweight='semibold')
    axs[0, 1].set_ylabel("Position (meters)", fontsize=10)
    axs[0, 1].grid(True, which='both', linestyle=':', alpha=0.5)
    axs[0, 1].legend(loc='lower right')
    
    # Z altitude subplot
    axs[1, 0].plot(time_points, positions[:, 2], label='Actual Z', color='#d62728', linewidth=2.5)
    axs[1, 0].plot(time_points, target_positions[:, 2], label='Target Z', color='#d62728', linestyle='--', linewidth=1.5)
    axs[1, 0].set_title("Z Translation (Altitude)", fontsize=12, fontweight='semibold')
    axs[1, 0].set_xlabel("Time (seconds)", fontsize=10)
    axs[1, 0].set_ylabel("Position (meters)", fontsize=10)
    axs[1, 0].grid(True, which='both', linestyle=':', alpha=0.5)
    axs[1, 0].legend(loc='lower right')
    
    # Attitude angles subplot
    axs[1, 1].plot(time_points, rolls, label='Roll', color='#9467bd', linewidth=2)
    axs[1, 1].plot(time_points, pitches, label='Pitch', color='#ff7f0e', linewidth=2)
    axs[1, 1].plot(time_points, yaws, label='Yaw', color='#17becf', linewidth=2)
    axs[1, 1].set_title("Euler Attitude Angles", fontsize=12, fontweight='semibold')
    axs[1, 1].set_xlabel("Time (seconds)", fontsize=10)
    axs[1, 1].set_ylabel("Angle (degrees)", fontsize=10)
    axs[1, 1].grid(True, which='both', linestyle=':', alpha=0.5)
    axs[1, 1].legend(loc='upper right')
    
    # Add a main title with parameters
    fig.suptitle(f"6 DoF Flight Response\n(Outer Kp_xy={kp_xy}, Inner Kp_att={kp_att}, Kd_att={kd_att})", fontsize=14, fontweight='bold', color='#333333')
    
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    return positions, target_positions, time_points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fly a drone in 3D using outer-loop position and inner-loop attitude controllers")
    parser.add_argument("--kp_z", type=float, default=1.2, help="Z Proportional gain")
    parser.add_argument("--kd_z", type=float, default=0.8, help="Z Derivative gain")
    parser.add_argument("--kp_xy", type=float, default=0.2, help="X/Y Proportional gain")
    parser.add_argument("--kd_xy", type=float, default=0.15, help="X/Y Derivative gain")
    parser.add_argument("--kp_att", type=float, default=20.0, help="Attitude Proportional gain (scaled)")
    parser.add_argument("--kd_att", type=float, default=5.0, help="Attitude Derivative gain (scaled)")
    parser.add_argument("--target_x", type=float, default=1.5, help="Target X position")
    parser.add_argument("--target_y", type=float, default=1.5, help="Target Y position")
    parser.add_argument("--target_z", type=float, default=2.0, help="Target Z position")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration in seconds")
    parser.add_argument("--output", type=str, default="full_3d_flight_response.png", help="Plot filename")
    args = parser.parse_args()
    
    def target_pos_func(t):
        # Fly vertically to target_z first, then translate to target_x, target_y at t=2s
        if t < 2.0:
            return 0.0, 0.0, args.target_z
        else:
            return args.target_x, args.target_y, args.target_z
        
    print(f"Running full 3D simulation with: ")
    print(f"  Outer loop: Kp_xy={args.kp_xy}, Kd_xy={args.kd_xy}")
    print(f"  Inner loop: Kp_att={args.kp_att}, Kd_att={args.kd_att}")
    print(f"  Target: ({args.target_x}, {args.target_y}, {args.target_z}) starting at t=2.0s")
    
    positions, target_positions, time_points = run_simulation(
        kp_z=args.kp_z,
        kd_z=args.kd_z,
        kp_xy=args.kp_xy,
        kd_xy=args.kd_xy,
        kp_att=args.kp_att,
        kd_att=args.kd_att,
        target_pos=target_pos_func,
        duration_sec=args.duration,
        plot_filename=args.output
    )
    
    # Calculate some metrics
    final_x = positions[-1, 0]
    final_y = positions[-1, 1]
    final_z = positions[-1, 2]
    
    print("\n--- Simulation Metrics ---")
    print(f"Final Position: X={final_x:.3f} m, Y={final_y:.3f} m, Z={final_z:.3f} m")
    print(f"Tracking Error at End: X_error={abs(args.target_x - final_x):.3f} m, Y_error={abs(args.target_y - final_y):.3f} m, Z_error={abs(args.target_z - final_z):.3f} m")
