import os
import argparse
import time
import numpy as np
import pybullet as p
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

def run_simulation(kp_z, kd_z, kp_xy, kd_xy, kp_att, kd_att, duration_sec=30.0, gui=True, d_0=0.6, k_rep=0.5, cam_rot_speed=12.0):
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
    
    # Track previous positions to draw flight trails
    prev_positions = [obs[i][:3].copy() for i in range(num_drones)]
    
    # Trail colors matching the drones' identities:
    # Drone 0 (Leader): Bright Red
    # Drone 1 (Follower 1): Neon Green
    # Drone 2 (Follower 2): Electric Blue
    trail_colors = [
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.5, 1.0]
    ]
    
    # Enable shadows and visual tweaks if GUI is active
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Hides Pybullet UI overlays for a clean portfolio capture
        start_time = time.time()
        
    print("Portfolio Showcase Simulation Started...")
    print("Press Ctrl+C in terminal or Esc/close window in GUI to exit.")
    
    for step in range(num_steps):
        t = step * dt
        action = np.zeros((num_drones, 4))
        
        # Leader's actual current position (Drone 0)
        leader_state = obs[0]
        leader_x = leader_state[0]
        leader_y = leader_state[1]
        
        for i in range(num_drones):
            # 1. Trajectory and Formations
            if i == 0:
                # Leader: Figures-8 flight path
                tx = 1.5 * np.sin(0.8 * t)
                ty = 1.5 * np.sin(1.6 * t)
                tz = 2.0
            elif i == 1:
                # Follower 1: Back-left of Leader
                tx = leader_x - 0.8
                ty = leader_y + 0.8
                tz = 2.0
            elif i == 2:
                # Follower 2: Back-right of Leader
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
                
            # 2. Extract State Vector
            state = obs[i]
            current_x, current_y, current_z = state[0], state[1], state[2]
            current_vx, current_vy, current_vz = state[10], state[11], state[12]
            current_roll, current_pitch, current_yaw = state[7], state[8], state[9]
            current_wx, current_wy, current_wz = state[13], state[14], state[15]
            
            # 3. Draw flight path trails dynamically in PyBullet
            if gui and step % 4 == 0: # draw trails every few steps to save performance
                curr_pos = np.array([current_x, current_y, current_z])
                dist = np.linalg.norm(curr_pos - prev_positions[i])
                if dist > 0.01: # only draw if drone has moved noticeably
                    p.addUserDebugLine(
                        lineFromXYZ=prev_positions[i],
                        lineToXYZ=curr_pos,
                        lineColorRGB=trail_colors[i],
                        lineWidth=2.5,
                        lifeTime=10.0 # Trails fade out after 10 seconds for a clean look
                    )
                    prev_positions[i] = curr_pos
            
            # 4. Outer Loop Controllers
            pid_force_z, error_z = pids_z[i].compute(tz, current_z, current_vz, dt)
            target_pitch, error_x = pids_x[i].compute(tx, current_x, current_vx, dt)
            
            pid_out_y, error_y = pids_y[i].compute(ty, current_y, current_vy, dt)
            target_roll = -pid_out_y
            
            target_pitch = np.clip(target_pitch, -0.4, 0.4)
            target_roll = np.clip(target_roll, -0.4, 0.4)
            
            # Tilt compensation
            cos_pitch = np.cos(target_pitch)
            cos_roll = np.cos(target_roll)
            tilt_comp = max(0.1, cos_pitch * cos_roll)
                
            total_thrust = (GRAVITY + pid_force_z) / tilt_comp
            total_thrust = max(0.0, total_thrust)
            
            # 5. Inner Loop Controllers
            roll_out, error_roll = pids_roll[i].compute(target_roll, current_roll, current_wx, dt)
            pitch_out, error_pitch = pids_pitch[i].compute(target_pitch, current_pitch, current_wy, dt)
            yaw_out, error_yaw = pids_yaw[i].compute(0.0, current_yaw, current_wz, dt)
            
            tau_x = Jxx * roll_out
            tau_y = Jyy * pitch_out
            tau_z = Jzz * yaw_out
            
            # 6. Quadcopter X Mixing Formulas
            F_0 = total_thrust / 4.0 - tau_x * np.sqrt(2) / (4.0 * L) - tau_y * np.sqrt(2) / (4.0 * L) - tau_z * (KF / (4.0 * KM))
            F_1 = total_thrust / 4.0 - tau_x * np.sqrt(2) / (4.0 * L) + tau_y * np.sqrt(2) / (4.0 * L) + tau_z * (KF / (4.0 * KM))
            F_2 = total_thrust / 4.0 + tau_x * np.sqrt(2) / (4.0 * L) + tau_y * np.sqrt(2) / (4.0 * L) - tau_z * (KF / (4.0 * KM))
            F_3 = total_thrust / 4.0 + tau_x * np.sqrt(2) / (4.0 * L) - tau_y * np.sqrt(2) / (4.0 * L) + tau_z * (KF / (4.0 * KM))
            
            motor_rpms = np.zeros(4)
            motor_rpms[0] = np.sqrt(max(0.0, F_0) / KF)
            motor_rpms[1] = np.sqrt(max(0.0, F_1) / KF)
            motor_rpms[2] = np.sqrt(max(0.0, F_2) / KF)
            motor_rpms[3] = np.sqrt(max(0.0, F_3) / KF)
            
            action[i, :] = np.clip(motor_rpms, 0, MAX_RPM)
            
        # Step simulation using physical commands directly
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 7. Cinematic Camera Tracking System
        if gui:
            # Camera tracks the leader (Drone 0)
            leader_pos = obs[0][:3]
            
            # Slow orbital rotation around the swarm
            camera_yaw = (t * cam_rot_speed) % 360
            p.resetDebugVisualizerCamera(
                cameraDistance=2.8,       # Zoomed out slightly to keep all drones in frame
                cameraYaw=camera_yaw,     # Dynamic rotation
                cameraPitch=-22.0,        # Cinematic angled top-down look
                cameraTargetPosition=leader_pos
            )
            
            # Sync with wall-clock for real-time smooth playback
            sync(step, start_time, dt)
            
    env.close()
    print("Showcase simulation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indefinite/cinematic portfolio flight showcase with orbiting camera and trails")
    parser.add_argument("--kp_z", type=float, default=1.2, help="Z Proportional gain")
    parser.add_argument("--kd_z", type=float, default=0.8, help="Z Derivative gain")
    parser.add_argument("--kp_xy", type=float, default=0.2, help="X/Y Proportional gain")
    parser.add_argument("--kd_xy", type=float, default=0.15, help="X/Y Derivative gain")
    parser.add_argument("--kp_att", type=float, default=20.0, help="Attitude Proportional gain")
    parser.add_argument("--kd_att", type=float, default=5.0, help="Attitude Derivative gain")
    parser.add_argument("--duration", type=float, default=300.0, help="Simulation duration (seconds, default: 300s)")
    parser.add_argument("--gui", type=str2bool, default=True, help="Whether to use PyBullet GUI")
    parser.add_argument("--d_0", type=float, default=0.6, help="APF safe distance threshold (meters)")
    parser.add_argument("--k_rep", type=float, default=0.5, help="APF repulsive force gain")
    parser.add_argument("--cam_speed", type=float, default=10.0, help="Camera rotation speed (deg/s)")
    args = parser.parse_args()
    
    run_simulation(
        kp_z=args.kp_z,
        kd_z=args.kd_z,
        kp_xy=args.kp_xy,
        kd_xy=args.kd_xy,
        kp_att=args.kp_att,
        kd_att=args.kd_att,
        duration_sec=args.duration,
        gui=args.gui,
        d_0=args.d_0,
        k_rep=args.k_rep,
        cam_rot_speed=args.cam_speed
    )
