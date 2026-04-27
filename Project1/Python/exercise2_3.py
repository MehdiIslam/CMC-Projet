#!/usr/bin/env python3

import os
import h5py
import matplotlib.pyplot as plt

from farms_core import pylog

from cmc_controllers.metrics import *
from simulate import runsim


BASE_PATH = 'logs/exercise2_3/'
PLOT_PATH = 'results'
ANIMAL_DATA_PATH = 'cmc_project_pack/models/a2sw5_cycle_smoothed.csv'


def get_animal_data(path):
    """Load animal data"""
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    freq = np.zeros(8)
    amp = np.zeros(8)
    ipls = np.zeros(7)
    times = data[10:-10, 0]
    joint_angles = data[10:-10, 1:9]


    freq, amp = compute_mechanical_frequency_amplitude_fft(times, joint_angles)
    
    inds_couples = [[i, i + 1]
                for i in range(joint_angles.shape[1] - 1)]    
    ipls, _ = compute_neural_phase_lags(times, joint_angles, freq, inds_couples)
    

    return freq, np.deg2rad(amp), ipls, times, np.deg2rad(joint_angles)




##the data give us the time and joint_angle
##we can compute: freq amp ipls of each joint so we cannot compare all the previous
## results we got i.e. curvatur fw_speed. 
##  But what we can compare is the freq amp ipls using the func in metric.py
## FFT-> peak_freq and peak_amp, to get the dominant periode
def exercise2_3(**kwargs):
    """ex2.3 main"""
    #pylog.warning("TODO: 2.3 Analyze the provided animal data and compare the animal locomotion performance with your implemented controller.")
    # pylog.set_level('critical')
    

    
    print("\nRunning Standard Robot Simulation (Default Parameters)...")
    #default parameters from Exercise 2.1 
    base_controller = {
    'loader': 'cmc_controllers.CPG_controller.CPGController',
    'config': {
        'drive_left': 3,
        'drive_right': 3,
        'd_low': 1,
        'd_high': 5,
        'a_rate': np.ones(8) * 3,
        'offset_freq': np.ones(8) * 1,
        'offset_amp': np.ones(8) * 0.5,
        'G_freq': np.ones(8) * 0.5,
        'G_amp': np.ones(8) * 0.25,
        'PL': np.ones(7) * np.pi * 2 / 8,
        'coupling_weights_rostral': 5,
        'coupling_weights_caudal': 5,
        'coupling_weights_contra': 10,
        'init_phase': np.random.default_rng(
            seed=42).uniform(
            0.0,
            2 * np.pi,
            size=16)}}
    """
    runsim(
        controller=base_controller,
        base_path=BASE_PATH,
        recording='exercise2_3.mp4',
    )
    """
    
    scaling_factor = np.sqrt(1.0 / 6.5)
    f_animal_scaled = f_animal * scaling_factor
    print("\n--- BONsUS: Running Tuned Robot Simulation ---")   
    TUNED_PATH = 'logs/exercise2_3_tuned/'
    os.makedirs(TUNED_PATH, exist_ok=True)
    
    tuned_controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': 1.0,   # Neutralize drive gain
            'drive_right': 1.0,  # Neutralize drive gain
            'd_low': 1.0,
            'd_high': 5.0,
            'a_rate': np.ones(8) * 3,
            'offset_freq': f_animal_scaled, # Biological Array
            'offset_amp': amp_animal * 2.5,       # Biological Array
            'G_freq': np.zeros(8),
            'G_amp': np.zeros(8),
            'PL': ipls_animal,              # Biological Array
            'coupling_weights_rostral': 5,
            'coupling_weights_caudal': 5,
            'coupling_weights_contra': 10,
            'init_phase': np.zeros(16)      # Synchronize start phase
        }
    }

    runsim(
        controller=tuned_controller,
        base_path=TUNED_PATH,
        recording='exercise2_3_tuned.mp4',
    )
    
    print("Extracting Animal Data...")
    f_animal, amp_animal, ipls_animal, t_animal, q_animal = get_animal_data(ANIMAL_DATA_PATH)

    
    print(f"Animal Mean Freq: {np.mean(f_animal):.2f} Hz -> Scaled Robot Target: {np.mean(f_animal_scaled):.2f} Hz")
    print(f"Animal Mean Amp:  {np.mean(amp_animal):.2f} rad")
    print(f"Animal Mean IPL:  {np.mean(ipls_animal):.2f} rad")
    
    
    print("\nExtracting simu Data...")
        # Load HDF5
    sim_result = BASE_PATH + 'simulation.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    sensor_data_joints_positions = sensor_data_joints[:, :8, 0]

    freq_rob, amp_rob = compute_mechanical_frequency_amplitude_fft(sim_times,sensor_data_joints_positions)
    
    inds_couples = [[i, i + 1]
                for i in range(sensor_data_joints_positions.shape[1] - 1)]    
    ipls_rob, _ = compute_neural_phase_lags(sim_times, sensor_data_joints_positions, freq_rob, inds_couples)
    
    print(f"Measured Robot Mean Freq: {np.mean(freq_rob):.2f} Hz")
    print(f"Measured Robot Mean Amp:  {np.mean(amp_rob):.2f} rad")
    print(f"Measured Robot Mean IPL:  {np.mean(ipls_rob):.2f} rad")
    
    plt.figure(figsize=(15, 5))

    # Plot 1: Kinematics Comparison (Joint 0)
    plt.subplot(1, 3, 1)
    # Shift time to start at 0 to overlay them cleanly
    plt.plot(t_animal - t_animal[0], q_animal[:, 0], label='Animal (Joint 0)', color='blue')
    plt.plot(sim_times - sim_times[0], sensor_data_joints_positions[:, 0], label='Robot (Joint 0)', linestyle='--', color='red')
    plt.xlim(0, 3) # Show 3 seconds
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Kinematics: Animal vs Tuned Robot')
    plt.legend()

    # Plot 2: Amplitude Envelope along the spine
    plt.subplot(1, 3, 2)
    joints = np.arange(8)
    plt.plot(joints, amp_animal, marker='o', label='Animal Target', color='blue')
    plt.plot(joints, amp_rob, marker='x', label='Robot Result', color='red')
    plt.xlabel('Joint Index (Head to Tail)')
    plt.ylabel('Amplitude (rad)')
    plt.title('Amplitude Envelope')
    plt.legend()

    # Plot 3: Phase Lag along the spine
    plt.subplot(1, 3, 3)
    couplings = np.arange(7)
    plt.plot(couplings, ipls_animal, marker='o', label='Animal Target', color='blue')
    plt.plot(couplings, ipls_rob, marker='x', label='Robot Result', color='red')
    plt.xlabel('Coupling Index (0=Head, 6=Tail)')
    plt.ylabel('Phase Lag (rad)')
    plt.title('Intersegmental Phase Lag (IPL)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "Q_2_3_Animal_vs_Robot.png"))
    print(f"\nSuccess! Plot saved to {PLOT_PATH}/Q_2_3_Animal_vs_Robot.png")
    
    

    print("\nExtracting Tuned Simulation Data...")
    sim_result_tuned = TUNED_PATH + 'simulation.hdf5'
    with h5py.File(sim_result_tuned, "r") as f:
        sim_times_tuned = f['times'][:]
        sensor_data_joints_tuned = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    
    # Extract strictly the 8 spine joints
    q_rob_tuned = sensor_data_joints_tuned[:, :8, 0]

    # Compute metrics for the tuned robot
    freq_rob_tuned, amp_rob_tuned = compute_mechanical_frequency_amplitude_fft(sim_times_tuned, q_rob_tuned)
    ipls_rob_tuned, _ = compute_neural_phase_lags(sim_times_tuned, q_rob_tuned, freq_rob_tuned, inds_couples)
    
    print(f"Tuned Robot Mean Freq: {np.mean(freq_rob_tuned):.2f} Hz")
    print(f"Tuned Robot Mean Amp:  {np.mean(amp_rob_tuned):.2f} rad")
    print(f"Tuned Robot Mean IPL:  {np.mean(ipls_rob_tuned):.2f} rad")

    # ==============================================================
    # PLOTTING OVERRIDE: Plot Animal vs TUNED Robot
    # ==============================================================
    plt.figure(figsize=(15, 5))

    # Plot 1: Kinematics Comparison (Joint 0)
    plt.subplot(1, 3, 1)
    # Joint 0 (Tête)
    plt.plot(t_animal - t_animal[0], q_animal[:, 0], label='Animal Head (0)', color='lightblue')
    plt.plot(sim_times_tuned - sim_times_tuned[0], q_rob_tuned[:, 0], label='Robot Head (0)', linestyle='--', color='blue')
    
    # Joint 7 (Queue)
    plt.plot(t_animal - t_animal[0], q_animal[:, 7], label='Animal Tail (7)', color='salmon')
    plt.plot(sim_times_tuned - sim_times_tuned[0], q_rob_tuned[:, 7], label='Robot Tail (7)', linestyle='--', color='red')
    
    plt.xlim(0, 3) 
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Kinematics: Head vs Tail')
    plt.legend(loc='upper right', fontsize='small')

    # Plot 2: Amplitude Envelope along the spine
    plt.subplot(1, 3, 2)
    joints = np.arange(8)
    plt.plot(joints, amp_animal, marker='o', label='Animal Target', color='blue')
    plt.plot(joints, amp_rob_tuned, marker='x', label='Tuned Robot Result', color='red')
    plt.xlabel('Joint Index (Head to Tail)')
    plt.ylabel('Amplitude (rad)')
    plt.title('Amplitude Envelope')
    plt.legend()

    # Plot 3: Phase Lag along the spine
    plt.subplot(1, 3, 3)
    couplings = np.arange(7)
    plt.plot(couplings, ipls_animal, marker='o', label='Animal Target', color='blue')
    plt.plot(couplings, ipls_rob_tuned, marker='x', label='Tuned Robot Result', color='red')
    plt.xlabel('Coupling Index (0=Head, 6=Tail)')
    plt.ylabel('Phase Lag (rad)')
    plt.title('Intersegmental Phase Lag (IPL)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "Q_2_3_Animal_vs_Tuned_Robot.png"))
    print(f"\nSuccess! Tuned plot saved to {PLOT_PATH}/Q_2_3_Animal_vs_Tuned_Robot.png")
    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise2_3(plot=True)

