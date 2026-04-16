#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")

import os
import pickle
import h5py
import matplotlib.pyplot as plt

from farms_core.utils.profile import profile
from farms_core import pylog

from simulate import runsim
from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_frequency_amplitude_fft,
    compute_mechanical_speed,
    compute_neural_phase_lags,
    filter_signals,
)

BASE_PATH = 'logs/exercise1_1/'
PLOT_PATH = 'results'


def post_processing():
    """Post processing"""
    # Load HDF5 ## physics data
    sim_result = BASE_PATH + 'simulation.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]
    sensor_data_links_positions = sensor_data_links[:, :, 7:10]
    sensor_data_links_velocities = sensor_data_links[:, :, 14:17]
    sensor_data_joints_positions = sensor_data_joints[:, :, 0]
    sensor_data_joints_velocities = sensor_data_joints[:, :, 1]
    sensor_data_joints_torques = sensor_data_joints[:, :, 2]

    # Load Controller ##neural signal
    with open(BASE_PATH + "controller.pkl", "rb") as f:
        controller_data = pickle.load(f)

    indices = controller_data["indices"]
    neural_signals = (
        controller_data["state"][:, indices['left_body_idx']]
        - controller_data["state"][:, indices['right_body_idx']]
    )
    neural_signals_smoothed = filter_signals(
        times=sim_times, signals=neural_signals)

    # Metrics computation
    #pylog.warning("TODO: 1.1: Complete metrics implementation in metrics.py")
    ##NM1
    freq, _, amp = compute_frequency_amplitude_fft( 
        times=sim_times, smooth_signals=neural_signals_smoothed)

    inds_couples = [[i, i+1]
                    for i in range(neural_signals_smoothed.shape[1] - 1)]
    ##NM2
    _, ipls_mean = compute_neural_phase_lags(times=sim_times,
                                             smooth_signals=neural_signals_smoothed,
                                             freqs=freq,
                                             inds_couples=inds_couples)
    ##MM1
    mech_freq, mech_amp = compute_mechanical_frequency_amplitude_fft(
        times=sim_times,
        signals=sensor_data_joints_positions[:, :8],
    )
    ##MM2
    speed_forward, speed_lateral = compute_mechanical_speed(
        links_positions=sensor_data_links_positions,
        links_velocities=sensor_data_links_velocities,
    )
    ##MM4
    energy, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=sensor_data_links_positions,
        joints_torques=sensor_data_joints_torques,
        joints_velocities=sensor_data_joints_velocities,
    )

    pylog.warning("TODO: 1.2: Verify the computed metrics are consistent with the expected values")
    print('Estimated neural metrics:')
    print('Frequencies: ', freq, '\nAmplitudes: ', amp,
          '\nMean phase lags (radians): ', ipls_mean)
    print('Estimated mechanical metrics:')
    print(
        'Frequencies: ',
        mech_freq,
        '\nAmplitudes: ',
        mech_amp,
        '\nforward speed: ',
        speed_forward,
        '\nlateral speed: ',
        speed_lateral,
        '\nEnergy: ',
        energy,
        '\nCoT: ',
        cot)

    pylog.warning("TODO: 1.2: Plot joint angles + CoM trajectory")
    from cmc_controllers.metrics import LINKS_MASSES,TOTAL_MASS 
    
    ## Joint angles
    real_joint_names = [
        "Joint 0", "Joint 1", "Joint 2", "Joint 3", 
        "Joint 4", "Joint 6", "Joint 7", "Joint 8"
    ]
    plt.figure(1)
    for i in range(8):
        plt.plot(sim_times, sensor_data_joints_positions[:, i], label=real_joint_names[i])
        
    plt.title("Time Evolution of Spine Joint Angles")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (rad)")

    plt.xlim([0, 4]) ## just the first 4 sec
    plt.grid(True)
    plt.legend(loc="best", fontsize="small", ncol=2)
    ##CoM 

    com_x = (sensor_data_links_positions[:, :, 0] @ LINKS_MASSES) / TOTAL_MASS
    com_y = (sensor_data_links_positions[:, :, 1] @ LINKS_MASSES) / TOTAL_MASS

    plt.figure(2)
    plt.plot(com_x, com_y, label="CoM Trajectory", color="blue")
    
    plt.scatter(com_x[0], com_y[0], color='green', marker='o', s=50, label='Start')
    plt.scatter(com_x[-1], com_y[-1], color='red', marker='X', s=50, label='End')
    
    plt.title("Robot CoM Trajectory in 2D")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    plt.figure(1).savefig("results/Q_1_2_joint_angles.png")
    plt.figure(2).savefig("results/Q_1_2_com_trajectory.png")
  

def main(**kwargs):
    """ex1.1 main"""
    os.makedirs(PLOT_PATH, exist_ok=True)
    controller = {
        'loader': 'cmc_controllers.wave_controller.WaveController',
        'config': {'freq': 2.0,
                   'twl': 1.0,
                   'amp': 2.0}
    }

    runsim(
        controller=controller,
        base_path=BASE_PATH,
        recording=False#'exercise1_1.mp4',
    )

    post_processing()


def exercise1_1(**kwargs):
    """Entry point for exercise 1.1 with optional plotting and runtime flags."""
    profile(function=main, profile_filename='',
            fast=kwargs.pop('fast', False),
            headless=kwargs.pop('headless', False),)

    plot = kwargs.pop('plot', False)
    #if plot:
        #plt.show()


if __name__ == '__main__':
    exercise1_1(plot=True)

