import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'swarm_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ryanjin',
    maintainer_email='ryanjin@todo.todo',
    description='Drone swarm simulation and control ROS2 package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_node = swarm_sim.sim_node:main',
            'control_node = swarm_sim.control_node:main',
            'trajectory_node = swarm_sim.trajectory_node:main',
        ],
    },
)
