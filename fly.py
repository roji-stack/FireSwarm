import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class CustomPIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target_z, current_z, current_vz, dt):
        error = target_z - current_z
        self.integral += error * dt
        
        # Using the actual measured vertical velocity (current_vz) for the D term.
        # Since target velocity is 0, derivative of error is -current_vz.
        derivative = -current_vz
        
        self.prev_error = error
        
        # Compute control output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output, error

def run_simulation(kp, kd, target_z_func, duration_sec=6.0, plot_filename="altitude_plot.png"):
    # Target_z_func is a function time -> target_z
    ctrl_freq = 240
    pyb_freq = 240
    
    # Initialize the single-drone environment
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.1]]), # start slightly above ground to prevent collision issues
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=Physics.PYB,
        pyb_freq=pyb_freq,
        ctrl_freq=ctrl_freq,
        gui=False,
        obstacles=False,
        user_debug_gui=False
    )
    
    # Set up controller
    # Ki must be exactly 0.0 as per physical simulation constraints
    pid = CustomPIDController(kp=kp, ki=0.0, kd=kd)
    
    # Track variables for plotting
    time_points = []
    altitudes = []
    target_altitudes = []
    thrust_values = []
    rpms = []
    
    obs, info = env.reset()
    
    num_steps = int(duration_sec * ctrl_freq)
    dt = 1.0 / ctrl_freq
    
    for step in range(num_steps):
        t = step * dt
        current_target_z = target_z_func(t)
        
        # State vector indices:
        # 0: X, 1: Y, 2: Z
        # 12: VZ (vertical velocity)
        state = obs[0]
        current_z = state[2]
        current_vz = state[12]
        
        # Compute PID control force (thrust force in Newtons)
        # Added env.GRAVITY to pre-compensate for gravity (feedforward term)
        pid_force, error = pid.compute(current_target_z, current_z, current_vz, dt)
        total_thrust = env.GRAVITY + pid_force
        
        # Ensure thrust is non-negative
        total_thrust = max(0.0, total_thrust)
        
        # Convert total thrust force to RPM for each of the 4 motors
        # Total force = 4 * KF * RPM^2 => RPM = sqrt(total_thrust / (4 * KF))
        target_rpm = np.sqrt(total_thrust / (4 * env.KF))
        
        # Clip the target RPM to the maximum allowed RPM
        target_rpm = min(target_rpm, env.MAX_RPM)
        
        # Action space expects shape (num_drones, 4)
        action = np.array([[target_rpm, target_rpm, target_rpm, target_rpm]])
        
        # Step simulation
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log data
        time_points.append(t)
        altitudes.append(current_z)
        target_altitudes.append(current_target_z)
        thrust_values.append(total_thrust)
        rpms.append(target_rpm)
        
    env.close()
    
    # Generate and save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, altitudes, label='Actual Altitude', color='blue', linewidth=2)
    plt.plot(time_points, target_altitudes, label='Target Altitude', color='red', linestyle='--', linewidth=2)
    plt.title(f"Altitude Step Response (Kp={kp:.2f}, Kd={kd:.2f})", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Altitude (meters)", fontsize=12)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.legend(loc='lower right', fontsize=12)
    
    # Add metrics text to plot
    max_alt = max(altitudes)
    final_alt = altitudes[-1]
    overshoot = (max_alt - target_z_func(duration_sec)) / target_z_func(duration_sec) * 100.0 if max_alt > target_z_func(duration_sec) else 0.0
    plt.text(0.05, 0.95, f"Max Altitude: {max_alt:.2f} m\nOvershoot: {overshoot:.1f}%\nFinal Altitude: {final_alt:.2f} m", 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    # Programmatic evaluation metrics
    altitudes = np.array(altitudes)
    time_points = np.array(time_points)
    
    # Check if settled: find time when it enters and stays in 5% band of target_z (for the final target_z)
    final_target = target_z_func(duration_sec)
    band = 0.05 * final_target
    within_band = np.abs(altitudes - final_target) <= band
    
    settling_time = None
    for idx in range(len(altitudes)):
        if np.all(within_band[idx:]):
            settling_time = time_points[idx]
            break
            
    # Calculate rise time: time to go from 10% to 90% of final_target
    idx_10 = np.where(altitudes >= 0.1 * final_target)[0]
    idx_90 = np.where(altitudes >= 0.9 * final_target)[0]
    rise_time = None
    if len(idx_10) > 0 and len(idx_90) > 0:
        rise_time = time_points[idx_90[0]] - time_points[idx_10[0]]
        
    # Check if drone oscillates wildly (sign changes in velocity after reaching 90% of target)
    vz_arr = np.diff(altitudes) / dt
    idx_reached = np.where(altitudes >= 0.9 * final_target)[0]
    oscillations = 0
    if len(idx_reached) > 0:
        post_reach_vz = vz_arr[idx_reached[0]:]
        # Count zero-crossings of velocity after reaching 90% target
        zero_crossings = np.where(np.diff(np.sign(post_reach_vz)))[0]
        oscillations = len(zero_crossings)
        
    return {
        "max_altitude": max_alt,
        "final_altitude": final_alt,
        "overshoot": overshoot,
        "settling_time": settling_time,
        "rise_time": rise_time,
        "oscillations": oscillations
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fly a drone with a custom Z-axis PID controller")
    parser.add_argument("--kp", type=float, default=1.2, help="Proportional gain")
    parser.add_argument("--kd", type=float, default=0.8, help="Derivative gain")
    parser.add_argument("--target", type=float, default=2.0, help="Target altitude (default: 2m)")
    parser.add_argument("--output", type=str, default="altitude_plot.png", help="Plot filename")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic target test (2m to 5m at mid-flight)")
    args = parser.parse_args()
    
    if args.dynamic:
        # Dynamic target: 2m for the first half, 5m for the second half of 10s simulation
        duration = 10.0
        def target_func(t):
            return 2.0 if t < 5.0 else 5.0
    else:
        duration = 6.0
        def target_func(t):
            return args.target
            
    print(f"Running simulation with Kp={args.kp}, Kd={args.kd}, Target={args.target if not args.dynamic else 'Dynamic (2m -> 5m)'}...")
    metrics = run_simulation(args.kp, args.kd, target_func, duration_sec=duration, plot_filename=args.output)
    
    print("\n--- Simulation Metrics ---")
    print(f"Max Altitude: {metrics['max_altitude']:.3f} m")
    print(f"Final Altitude: {metrics['final_altitude']:.3f} m")
    print(f"Overshoot: {metrics['overshoot']:.1f}%")
    rise_time_str = f"{metrics['rise_time']:.3f} s" if metrics['rise_time'] is not None else "N/A"
    settling_time_str = f"{metrics['settling_time']:.3f} s" if metrics['settling_time'] is not None else "N/A"
    print(f"Rise Time (10%-90%): {rise_time_str}")
    print(f"Settling Time (5% band): {settling_time_str}")
    print(f"Oscillations (Post-90%): {metrics['oscillations']}")
