#!/usr/bin/env python3
"""
control_node.py
ROS2 Node running PID attitude/position control and swarm V-formation coordination.
"""

import sys
from functools import partial
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

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

def euler_from_quaternion(x, y, z, w):
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)
        
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

def calculate_cpa_repulsion(pos_i, vel_i, pos_j, vel_j, d_0, k_rep, max_time):
    p_i = np.array(pos_i[:2])
    v_i = np.array(vel_i[:2])
    p_j = np.array(pos_j[:2])
    v_j = np.array(vel_j[:2])
    
    p_rel = p_i - p_j
    v_rel = v_i - v_j
    
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm < 1e-9:
        return np.zeros(2)
        
    t_cpa = -np.dot(p_rel, v_rel) / np.dot(v_rel, v_rel)
    t_cpa = np.clip(t_cpa, 0.0, max_time)
    
    proj_pos_i = p_i + v_i * t_cpa
    proj_pos_j = p_j + v_j * t_cpa
    
    diff = proj_pos_i - proj_pos_j
    d = np.linalg.norm(diff)
    
    if 0.0 < d < d_0:
        repulsive_magnitude = k_rep * (1.0 / d - 1.0 / d_0) * (1.0 / (d ** 2))
        return repulsive_magnitude * (diff / d)
    else:
        return np.zeros(2)


class SwarmControlNode(Node):
    def __init__(self):
        super().__init__('swarm_control_node')
        
        self.num_drones = 3
        self.leader_id = 0
        self.ctrl_freq = 240
        self.dt = 1.0 / self.ctrl_freq
        self.step_count = 0
        
        # Physical Drone Constants (CF2X model parameters from gym-pybullet-drones)
        self.KF = 3.16e-10
        self.KM = 7.94e-12
        self.L = 0.0397
        self.GRAVITY = 0.2646
        self.MAX_RPM = 21702.64
        self.Jxx = 1.4e-05
        self.Jyy = 1.4e-05
        self.Jzz = 2.17e-05
        
        # APF Constants
        self.d_0 = 0.6
        self.k_rep = 0.5
        self.feedforward_gain = 0.15
        self.max_time = 2.0
        
        # Initialize internal state arrays (spawn positions spaced 1m along X)
        self.positions = np.array([
            [-1.0, 0.0, 0.1],
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.1]
        ])
        self.velocities = np.zeros((self.num_drones, 3))
        self.rpy = np.zeros((self.num_drones, 3))
        self.ang_vel = np.zeros((self.num_drones, 3))
        
        # Setup Cascading PID Controllers (matching tuned gains from swarm.py)
        kp_z, kd_z = 1.2, 0.8
        kp_xy, kd_xy = 0.2, 0.15
        kp_att, kd_att = 20.0, 5.0
        
        self.pids_z = [CustomPIDController(kp=kp_z, ki=0.0, kd=kd_z) for _ in range(self.num_drones)]
        self.pids_x = [CustomPIDController(kp=kp_xy, ki=0.0, kd=kd_xy) for _ in range(self.num_drones)]
        self.pids_y = [CustomPIDController(kp=kp_xy, ki=0.0, kd=kd_xy) for _ in range(self.num_drones)]
        
        self.pids_roll = [CustomPIDController(kp=kp_att, ki=0.0, kd=kd_att) for _ in range(self.num_drones)]
        self.pids_pitch = [CustomPIDController(kp=kp_att, ki=0.0, kd=kd_att) for _ in range(self.num_drones)]
        self.pids_yaw = [CustomPIDController(kp=kp_att, ki=0.0, kd=kd_att) for _ in range(self.num_drones)]
        
        # Create subscribers for /swarm/drone_i/odom
        self.subscribers_odom = []
        for i in range(self.num_drones):
            sub = self.create_subscription(
                Odometry,
                f'/swarm/drone_{i}/odom',
                partial(self.odom_callback, drone_id=i),
                10
            )
            self.subscribers_odom.append(sub)
            
        # Create publishers for /swarm/drone_i/cmd_motor
        self.publishers_cmd = []
        for i in range(self.num_drones):
            pub = self.create_publisher(
                Float32MultiArray,
                f'/swarm/drone_{i}/cmd_motor',
                10
            )
            self.publishers_cmd.append(pub)
            
        # Target coordinates for Drone 0
        self.leader_target_x = -1.0
        self.leader_target_y = 0.0
        self.leader_target_z = 2.0
        
        # Leader feedforward velocity state
        self.leader_feedforward_vx = 0.0
        self.leader_feedforward_vy = 0.0
        self.leader_feedforward_vz = 0.0
        
        # Create subscriber for /swarm/leader_trajectory
        self.sub_leader_trajectory = self.create_subscription(
            Odometry,
            '/swarm/leader_trajectory',
            self.leader_trajectory_callback,
            10
        )
        
        # Create high-frequency control loop timer at 240 Hz
        self.timer = self.create_timer(self.dt, self.control_loop_callback)
        self.get_logger().info(f"Swarm Control Node initialized. Control loop started at {self.ctrl_freq} Hz.")

    def odom_callback(self, msg, drone_id):
        # Extract position
        self.positions[drone_id, 0] = msg.pose.pose.position.x
        self.positions[drone_id, 1] = msg.pose.pose.position.y
        self.positions[drone_id, 2] = msg.pose.pose.position.z
        
        # Extract and convert quaternion orientation to Euler angles (RPY)
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        roll, pitch, yaw = euler_from_quaternion(qx, qy, qz, qw)
        self.rpy[drone_id, 0] = roll
        self.rpy[drone_id, 1] = pitch
        self.rpy[drone_id, 2] = yaw
        
        # Extract linear velocities
        self.velocities[drone_id, 0] = msg.twist.twist.linear.x
        self.velocities[drone_id, 1] = msg.twist.twist.linear.y
        self.velocities[drone_id, 2] = msg.twist.twist.linear.z
        
        # Extract angular velocities
        self.ang_vel[drone_id, 0] = msg.twist.twist.angular.x
        self.ang_vel[drone_id, 1] = msg.twist.twist.angular.y
        self.ang_vel[drone_id, 2] = msg.twist.twist.angular.z

    def leader_trajectory_callback(self, msg):
        self.leader_target_x = msg.pose.pose.position.x
        self.leader_target_y = msg.pose.pose.position.y
        self.leader_target_z = msg.pose.pose.position.z
        
        self.leader_feedforward_vx = msg.twist.twist.linear.x
        self.leader_feedforward_vy = msg.twist.twist.linear.y
        self.leader_feedforward_vz = msg.twist.twist.linear.z

    def control_loop_callback(self):
        t = self.step_count * self.dt
        
        # Leader's current position (Drone 0)
        leader_x = self.positions[0, 0]
        leader_y = self.positions[0, 1]
        leader_z = self.positions[0, 2]
        
        # Leader's current velocity (Drone 0)
        leader_vx = self.velocities[0, 0]
        leader_vy = self.velocities[0, 1]
        leader_vz = self.velocities[0, 2]
        
        # 240 Hz control loop time step
        dt = 1.0 / 240.0
        
        for i in range(self.num_drones):
            # 1. Swarm Path Planning (V-Formation target trajectory)
            if i == 0:
                tx = self.leader_target_x
                ty = self.leader_target_y
                tz = self.leader_target_z
            elif i == 1:
                # Follower 1: Offset to the back-left of Leader's actual position with feedforward
                tx = leader_x - 0.8 + self.feedforward_gain * self.leader_feedforward_vx
                ty = leader_y + 0.8 + self.feedforward_gain * self.leader_feedforward_vy
                tz = (leader_z - 0.1) + self.feedforward_gain * self.leader_feedforward_vz
            elif i == 2:
                # Follower 2: Offset to the back-right of Leader's actual position with feedforward
                tx = leader_x - 0.8 + self.feedforward_gain * self.leader_feedforward_vx
                ty = leader_y - 0.8 + self.feedforward_gain * self.leader_feedforward_vy
                tz = (leader_z - 0.1) + self.feedforward_gain * self.leader_feedforward_vz
                
            # Predictive Collision Avoidance (RVO-lite)
            total_repulsion = np.zeros(2)
            if i != self.leader_id:
                # Calculate distance to designated V-formation target coordinate (position error) before APF
                pos_error = np.linalg.norm(self.positions[i] - np.array([tx, ty, tz]))
                
                # Proximity Attenuation: if settled within 0.15m, hold ground (total_repulsion = [0.0, 0.0])
                if pos_error >= 0.15:
                    pos_i = self.positions[i]
                    vel_i = self.velocities[i]
                    for j in range(self.num_drones):
                        if j != i:
                            pos_j = self.positions[j]
                            vel_j = self.velocities[j]
                            repulsion = calculate_cpa_repulsion(
                                pos_i, vel_i, pos_j, vel_j, self.d_0, self.k_rep, self.max_time
                            )
                            # Right-of-Way hierarchy and asymmetric weight application
                            if j == self.leader_id:
                                # Leader has absolute priority
                                multiplier = 1.8
                            else:
                                # Between followers, lower ID has right-of-way
                                if i < j:
                                    # Follower A has right-of-way over Follower B
                                    multiplier = 0.2
                                else:
                                    # Follower B must yield to Follower A
                                    multiplier = 1.8
                            total_repulsion += repulsion * multiplier
            tx += total_repulsion[0]
            ty += total_repulsion[1]
            
            # Extract state vector
            current_x, current_y, current_z = self.positions[i]
            current_vx, current_vy, current_vz = self.velocities[i]
            current_roll, current_pitch, current_yaw = self.rpy[i]
            current_wx, current_wy, current_wz = self.ang_vel[i]
            
            # 2. Outer loop position controllers
            # Z controller -> total thrust
            pid_force_z, _ = self.pids_z[i].compute(tz, current_z, current_vz, self.dt)
            
            # X position controller -> target pitch
            target_pitch, _ = self.pids_x[i].compute(tx, current_x, current_vx, self.dt)
            
            # Y position controller -> target roll
            pid_out_y, _ = self.pids_y[i].compute(ty, current_y, current_vy, self.dt)
            target_roll = -pid_out_y
            
            # Clamp target pitch and roll to +/- 0.4 radians
            target_pitch = np.clip(target_pitch, -0.4, 0.4)
            target_roll = np.clip(target_roll, -0.4, 0.4)
            
            # Apply tilt compensation to Z thrust
            cos_pitch = np.cos(target_pitch)
            cos_roll = np.cos(target_roll)
            tilt_comp = cos_pitch * cos_roll
            if tilt_comp < 0.1:
                tilt_comp = 0.1
                
            total_thrust = (self.GRAVITY + pid_force_z) / tilt_comp
            total_thrust = max(0.0, total_thrust)
            
            # 3. Inner loop attitude control outputs (angular accelerations in rad/s^2)
            roll_out, _ = self.pids_roll[i].compute(target_roll, current_roll, current_wx, self.dt)
            pitch_out, _ = self.pids_pitch[i].compute(target_pitch, current_pitch, current_wy, self.dt)
            yaw_out, _ = self.pids_yaw[i].compute(0.0, current_yaw, current_wz, self.dt)
            
            # Convert angular accelerations to torques (N*m) using inertia
            tau_x = self.Jxx * roll_out
            tau_y = self.Jyy * pitch_out
            tau_z = self.Jzz * yaw_out
            
            # 4. Quadcopter X-configuration mixing formulas for individual motor forces
            F_0 = total_thrust / 4.0 - tau_x * np.sqrt(2) / (4.0 * self.L) - tau_y * np.sqrt(2) / (4.0 * self.L) - tau_z * (self.KF / (4.0 * self.KM))
            F_1 = total_thrust / 4.0 - tau_x * np.sqrt(2) / (4.0 * self.L) + tau_y * np.sqrt(2) / (4.0 * self.L) + tau_z * (self.KF / (4.0 * self.KM))
            F_2 = total_thrust / 4.0 + tau_x * np.sqrt(2) / (4.0 * self.L) + tau_y * np.sqrt(2) / (4.0 * self.L) - tau_z * (self.KF / (4.0 * self.KM))
            F_3 = total_thrust / 4.0 + tau_x * np.sqrt(2) / (4.0 * self.L) - tau_y * np.sqrt(2) / (4.0 * self.L) + tau_z * (self.KF / (4.0 * self.KM))
            
            # Convert motor forces to RPM: RPM = sqrt(F / KF)
            motor_rpms = np.zeros(4)
            motor_rpms[0] = np.sqrt(max(0.0, F_0) / self.KF)
            motor_rpms[1] = np.sqrt(max(0.0, F_1) / self.KF)
            motor_rpms[2] = np.sqrt(max(0.0, F_2) / self.KF)
            motor_rpms[3] = np.sqrt(max(0.0, F_3) / self.KF)
            
            # Clip motor RPMs to max RPM and construct the ROS2 message
            clipped_rpms = np.clip(motor_rpms, 0.0, self.MAX_RPM)
            
            msg = Float32MultiArray()
            msg.data = [float(val) for val in clipped_rpms]
            self.publishers_cmd[i].publish(msg)
            
        # Increment global controller step count
        self.step_count += 1

def main(args=None):
    rclpy.init(args=args)
    node = SwarmControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
