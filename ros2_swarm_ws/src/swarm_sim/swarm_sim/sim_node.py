#!/usr/bin/env python3
"""
sim_node.py
ROS2 Node hosting the PyBullet simulation loop for the 3-drone swarm.
"""

import sys
from functools import partial
import numpy as np
import pybullet as p

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class SwarmSimulatorNode(Node):
    def __init__(self):
        super().__init__('swarm_simulator_node')
        
        self.num_drones = 3
        self.ctrl_freq = 240
        self.pyb_freq = 240
        
        # Spawn the drones 1 meter apart along the X-axis as requested
        self.initial_xyzs = np.array([
            [-1.0, 0.0, 0.1],
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.1]
        ])
        
        # Initialize the 3-drone PyBullet environment with GUI enabled
        self.get_logger().info("Initializing PyBullet Swarm Environment...")
        self.env = CtrlAviary(
            drone_model=DroneModel.CF2X,
            num_drones=self.num_drones,
            initial_xyzs=self.initial_xyzs,
            initial_rpys=np.zeros((self.num_drones, 3)),
            physics=Physics.PYB,
            pyb_freq=self.pyb_freq,
            ctrl_freq=self.ctrl_freq,
            gui=True,
            obstacles=False,
            user_debug_gui=False
        )
        
        # Reset the environment to obtain initial observation
        self.obs, _ = self.env.reset()
        
        # Internal array to store the latest motor commands (4-rotor RPMs) for all 3 drones
        # We start with zeros (or hover RPM, but standard is zeros on startup until controlled)
        self.motor_rpms = np.zeros((self.num_drones, 4))
        
        # Create subscribers for /swarm/drone_i/cmd_motor
        self.subscribers_cmd = []
        for i in range(self.num_drones):
            sub = self.create_subscription(
                Float32MultiArray,
                f'/swarm/drone_{i}/cmd_motor',
                partial(self.cmd_motor_callback, drone_id=i),
                10
            )
            self.subscribers_cmd.append(sub)
            self.get_logger().info(f"Subscribed to /swarm/drone_{i}/cmd_motor")
            
        # Create publishers for /swarm/drone_i/odom
        self.publishers_odom = []
        for i in range(self.num_drones):
            pub = self.create_publisher(
                Odometry,
                f'/swarm/drone_{i}/odom',
                10
            )
            self.publishers_odom.append(pub)
            self.get_logger().info(f"Created publisher for /swarm/drone_{i}/odom")
            
        # Create high-frequency timer loop (240 Hz matching pyb_freq and ctrl_freq)
        self.timer_period = 1.0 / self.ctrl_freq
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info(f"High-frequency simulator timer started at {self.ctrl_freq} Hz.")

    def cmd_motor_callback(self, msg, drone_id):
        # Update the corresponding row in the internal motor command array
        if len(msg.data) == 4:
            self.motor_rpms[drone_id, :] = msg.data
        else:
            self.get_logger().warn(
                f"Received invalid motor command length ({len(msg.data)}) for drone {drone_id}. Expected 4."
            )

    def timer_callback(self):
        # Step the PyBullet environment using the current motor command array
        self.obs, _, _, _, _ = self.env.step(self.motor_rpms)
        
        # Loop through drones, extract state, and publish Odometry
        for i in range(self.num_drones):
            state = self.obs[i]
            
            # Create Odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = 'world'
            odom_msg.child_frame_id = f'drone_{i}'
            
            # Position (state[0:3])
            odom_msg.pose.pose.position.x = float(state[0])
            odom_msg.pose.pose.position.y = float(state[1])
            odom_msg.pose.pose.position.z = float(state[2])
            
            if i == 0:
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.5,
                    cameraYaw=-45.0,
                    cameraPitch=-30.0,
                    cameraTargetPosition=[float(state[0]), float(state[1]), float(state[2])]
                )
            
            # Orientation Quaternion (state[3:7])
            odom_msg.pose.pose.orientation.x = float(state[3])
            odom_msg.pose.pose.orientation.y = float(state[4])
            odom_msg.pose.pose.orientation.z = float(state[5])
            odom_msg.pose.pose.orientation.w = float(state[6])
            
            # Twist Linear Velocity in world/body frame (state[10:13])
            odom_msg.twist.twist.linear.x = float(state[10])
            odom_msg.twist.twist.linear.y = float(state[11])
            odom_msg.twist.twist.linear.z = float(state[12])
            
            # Twist Angular Velocity in world/body frame (state[13:16])
            odom_msg.twist.twist.angular.x = float(state[13])
            odom_msg.twist.twist.angular.y = float(state[14])
            odom_msg.twist.twist.angular.z = float(state[15])
            
            # Publish state telemetry
            self.publishers_odom[i].publish(odom_msg)

    def destroy_node(self):
        # Close PyBullet environment cleanly
        self.get_logger().info("Closing PyBullet Environment...")
        self.env.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SwarmSimulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
