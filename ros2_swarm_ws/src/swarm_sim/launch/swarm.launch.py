from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='swarm_sim',
            executable='sim_node',
            name='sim_node',
            output='screen'
        ),
        Node(
            package='swarm_sim',
            executable='control_node',
            name='control_node',
            output='screen'
        ),
        Node(
            package='swarm_sim',
            executable='trajectory_node',
            name='trajectory_node',
            output='screen'
        )
    ])
