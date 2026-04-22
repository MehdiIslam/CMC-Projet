#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")

import time
import os
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt


from farms_core import pylog
from farms_core.utils.profile import profile

from simulate import runsim
from cmc_controllers.metrics import filter_signals

BASE_PATH = 'logs/exercise2_1/'
PLOT_PATH = 'results'


def post_processing(base_path):
    """Post processing"""
    # Load HDF5
    sim_result = base_path + 'simulation.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    sensor_data_links_positions = sensor_data_links[:, :, 7:10]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]

    # Load Controller
    with open(base_path + "controller.pkl", "rb") as f:
        controller_data = pickle.load(f)

    
    n_oscillators = 16

    indices = controller_data["indices"]
    left_storage_idx = slice(
            indices["left_body_idx"].start + n_oscillators*2,
            indices["left_body_idx"].stop + n_oscillators*2,
            indices["left_body_idx"].step)
    right_storage_idx = slice(
            indices["right_body_idx"].start + n_oscillators*2,
            indices["right_body_idx"].stop + n_oscillators*2,
            indices["right_body_idx"].step
            )
    
    motor_left = controller_data["state"][:, left_storage_idx]
    motor_right = controller_data["state"][:, right_storage_idx]
    phases = controller_data["state"][:, :n_oscillators]
    amplitudes = controller_data["state"][:, n_oscillators:2 * n_oscillators]
    print(phases.shape, amplitudes.shape, motor_left.shape, motor_right.shape)
    print(indices)
    print(motor_left.shape)

    real_joint_names = [
        "Joint 0", "Joint 1", "Joint 2", "Joint 3", 
        "Joint 4", "Joint 6", "Joint 7", "Joint 8"
    ]

    real_oscillator_names = [
        "Joint 0 - L", "Joint 0 - R", "Joint 1 - L", "Joint 1 - R", 
        "Joint 2 - L", "Joint 2 - R", "Joint 3 - L", "Joint 3 - R", 
        "Joint 4 - L", "Joint 4 - R", "Joint 6 - L", "Joint 6 - R",
        "Joint 7 - L", "Joint 7 - R", "Joint 8 - L", "Joint 8 - R"
    ]
    
    plt.figure(0)
    for i in range(8):
        plt.plot(sim_times, sensor_data_joints_positions[:, i], label=real_joint_names[i])
        
    plt.title("Time Evolution of Spine Joint Angles")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (rad)")

    plt.xlim([0, 5]) 
    plt.grid(True)
    plt.legend(loc="best", fontsize="small", ncol=2)

    
    from cmc_controllers.metrics import LINKS_MASSES,TOTAL_MASS 
    com_x = (sensor_data_links_positions[:, :, 0] @ LINKS_MASSES) / TOTAL_MASS
    com_y = (sensor_data_links_positions[:, :, 1] @ LINKS_MASSES) / TOTAL_MASS

    plt.figure(1)
    plt.plot(com_x, com_y, label="CoM Trajectory", color="blue")
    
    plt.scatter(com_x[0], com_y[0], color='green', marker='o', s=50, label='Start')
    plt.scatter(com_x[-1], com_y[-1], color='red', marker='X', s=50, label='End')

    plt.title("Robot CoM Trajectory in 2D")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    plt.figure(2)
    for i in range(8):
        plt.plot(sim_times, (motor_left[:, i] + motor_right[:, i]), label=real_joint_names[i])
        
    plt.title("Time Evolution of muscle output sum")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque output (unit?)")

    plt.xlim([0, 5]) 
    plt.grid(True)
    plt.legend(loc="best", fontsize="small", ncol=2)

    plt.figure(3)
    for i in range(8):
        plt.plot(sim_times, (motor_left[:, i] - motor_right[:, i]), label=real_joint_names[i])
        
    plt.title("Time Evolution of muscle output diff")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque output (unit?)")

    plt.xlim([0, 5]) 
    plt.grid(True)
    plt.legend(loc="best", fontsize="small", ncol=2)

    plt.figure(4)
    for i in range(n_oscillators):
        plt.plot(sim_times, phases[:, i], label=real_oscillator_names[i])
        
    plt.title("Time Evolution of phases")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")

    plt.xlim([0, 5]) 
    plt.grid(True)
    plt.legend(loc="best", fontsize="small", ncol=2)

    plt.figure(5)
    for i in range(n_oscillators):
        plt.plot(sim_times, amplitudes[:, i], label=real_oscillator_names[i])
        
    plt.title("Time Evolution of amplitudes")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (rad)")

    plt.xlim([0, 5]) 
    plt.grid(True)
    plt.legend(loc="best", fontsize="small", ncol=2)

    plt.figure(0).savefig("results/Q_2_1_joint_angle.png")
    plt.figure(1).savefig("results/Q_2_1_com_trajectory.png")
    plt.figure(2).savefig("results/Q_2_1_sum_muscle.png")
    plt.figure(3).savefig("results/Q_2_1_difference_muscle.png")
    plt.figure(4).savefig("results/Q_2_1_phases.png")
    plt.figure(5).savefig("results/Q_2_1_amplitudes.png")


def main(**kwargs):
    """Run exercise 2.1 simulation and post-processing pipeline."""
    os.makedirs(PLOT_PATH, exist_ok=True)
    controller = {
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

    tic = time.time()
    runsim(
        controller=controller,
        base_path=BASE_PATH,
        recording=False#'exercise2_1.mp4',
    )
    post_processing(BASE_PATH)
    pylog.info('Total simulation time: %s [s]', time.time() - tic)


def exercise2_1(**kwargs):
    """ex2.1 main"""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)
    plot = kwargs.pop('plot', False)
    #if plot:
    #    plt.show()


if __name__ == '__main__':
    exercise2_1(plot=True)

