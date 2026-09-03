#!/usr/bin/env python3
"""
trajectory_node.py
ROS2 Node generating smooth cubic polynomial trajectories through a set of 3D waypoints.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class TrajectoryPlannerNode(Node):
    def __init__(self):
        super().__init__('trajectory_planner_node')
        
        # Declare parameter to toggle between autonomous trajectory and manual teleop
        self.declare_parameter('use_teleop', False)
        
        # Publisher for the target trajectory carrying position and velocity
        self.pub_trajectory = self.create_publisher(
            Odometry,
            '/swarm/leader_trajectory',
            10
        )
        
        # Subscriber for manual keyboard teleop cmd_vel
        self.sub_cmd_vel = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # 3D waypoints: [-1.0, 0.0, 2.0] -> [0.0, 0.0, 2.0] -> [2.0, 2.0, 3.0] -> [0.0, 4.0, 2.0] -> [-2.0, 2.0, 3.0] -> [0.0, 0.0, 2.0]
        # To make transition smooth on startup, we begin the waypoint list at the leader's target takeoff point [-1.0, 0.0, 2.0]
        self.waypoints = [
            [-1.0, 0.0, 2.0],  # Spawn/takeoff position
            [0.0, 0.0, 2.0],
            [2.0, 2.0, 3.0],
            [0.0, 4.0, 2.0],
            [-2.0, 2.0, 3.0],
            [0.0, 0.0, 2.0]    # Final Stop
        ]
        self.trajectory_finished = False
        
        # Leg duration T in seconds
        self.T = 5.0
        
        # Internal clock t (seconds) and 50 Hz timer
        self.freq = 50
        self.dt = 1.0 / self.freq
        self.t = 0.0
        self.current_wp_idx = 0
        
        # Teleop/Manual velocities
        self.cmd_vel_x = 0.0
        self.cmd_vel_y = 0.0
        self.cmd_vel_z = 0.0
        
        # Smooth mode transition variables
        self.last_pos = np.array([-1.0, 0.0, 2.0])
        self.teleop_pos = np.array([-1.0, 0.0, 2.0])
        self.teleop_initialized = False
        
        # Start timer loop
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info(f"Trajectory Planner Node started at {self.freq} Hz.")
        
    def cmd_vel_callback(self, msg):
        self.cmd_vel_x = msg.linear.x
        self.cmd_vel_y = msg.linear.y
        self.cmd_vel_z = msg.linear.z
        
    def timer_callback(self):
        use_teleop = self.get_parameter('use_teleop').value
        
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'world'
        odom_msg.child_frame_id = 'leader_target'
        
        if use_teleop:
            if not self.teleop_initialized:
                # Initialize teleop position to last known position to prevent jumps
                self.teleop_pos = np.copy(self.last_pos)
                self.teleop_initialized = True
                self.trajectory_finished = False
            
            # Integrate manual velocities
            self.teleop_pos[0] += self.cmd_vel_x * self.dt
            self.teleop_pos[1] += self.cmd_vel_y * self.dt
            self.teleop_pos[2] += self.cmd_vel_z * self.dt
            self.teleop_pos[2] = max(0.5, self.teleop_pos[2])  # ground protection clamp
            
            pos = self.teleop_pos
            vel = np.array([self.cmd_vel_x, self.cmd_vel_y, self.cmd_vel_z])
            
        else:
            if self.teleop_initialized:
                # Re-starting from current manual location for a smooth takeoff to circuit
                self.waypoints[0] = list(self.teleop_pos)
                self.current_wp_idx = 0
                self.t = 0.0
                self.trajectory_finished = False
                self.teleop_initialized = False
                
            if self.trajectory_finished:
                pos = np.array(self.waypoints[-1])
                vel = np.zeros(3)
            else:
                # Reset clock and transition to the next leg when t >= T
                if self.t >= self.T:
                    self.t = 0.0
                    if self.current_wp_idx == len(self.waypoints) - 2:
                        self.trajectory_finished = True
                        self.current_wp_idx += 1
                    else:
                        self.current_wp_idx += 1
                        
                if self.trajectory_finished:
                    pos = np.array(self.waypoints[-1])
                    vel = np.zeros(3)
                else:
                    # Determine start and end waypoints for the current leg
                    wp_start = np.array(self.waypoints[self.current_wp_idx])
                    wp_end = np.array(self.waypoints[self.current_wp_idx + 1])
                    
                    # Cubic polynomial coefficients calculated dynamically:
                    # tau = t / T in [0, 1]
                    # P(tau) = P_start + (P_end - P_start) * (3*tau^2 - 2*tau^3)
                    # V(tau) = (P_end - P_start)/T * (6*tau - 6*tau^2)
                    tau = np.clip(self.t / self.T, 0.0, 1.0)
                    
                    pos = wp_start + (wp_end - wp_start) * (3 * tau**2 - 2 * tau**3)
                    vel = (wp_end - wp_start) / self.T * (6 * tau - 6 * tau**2)
                    
                    # Advance internal clock
                    self.t += self.dt
            
        # Store last position for smooth transition
        self.last_pos = np.copy(pos)
        
        # Pack message fields
        odom_msg.pose.pose.position.x = float(pos[0])
        odom_msg.pose.pose.position.y = float(pos[1])
        odom_msg.pose.pose.position.z = float(pos[2])
        
        odom_msg.twist.twist.linear.x = float(vel[0])
        odom_msg.twist.twist.linear.y = float(vel[1])
        odom_msg.twist.twist.linear.z = float(vel[2])
        
        # Publish
        self.pub_trajectory.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
