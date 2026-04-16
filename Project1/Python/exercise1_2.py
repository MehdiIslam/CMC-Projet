#!/usr/bin/env python3
"""Run parameter sweeps for exercise 1.2 and plot metric heatmaps."""
import matplotlib
matplotlib.use("Agg")

import os
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from farms_core import pylog

from simulate import run_multiple
from cmc_controllers.metrics import (
    compute_frequency_amplitude_fft,
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    compute_neural_phase_lags,
    filter_signals,
)

# Multiprocessing

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

MAX_WORKERS = 8  # adjust based on your hardware capabilities

pylog.set_level('critical')

BASE_PATH = 'logs/exercise1_2/'
PLOT_PATH = 'results'
RECORDING = None  # disable recording for parallel runs


def get_metrics(twl, amp):
    """Compute mechanical metrics for a single parameter set."""
    # Load HDF5
    sim_result = BASE_PATH + \
        f'simulation_twl{twl:0.3f}_amp{amp:0.3f}.hdf5'
    with h5py.File(sim_result, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    sensor_data_links_positions = sensor_data_links[:, :, 7:10]

    sensor_data_links_velocities = sensor_data_links[:, :, 14:17]
    sensor_data_joints_velocities = sensor_data_joints[:, :, 1]
    sensor_data_joints_torques = sensor_data_joints[:, :, 2]

    speed_forward, _ = compute_mechanical_speed(
        links_positions=sensor_data_links_positions,
        links_velocities=sensor_data_links_velocities,
    )
    _, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=sensor_data_links_positions,
        joints_torques=sensor_data_joints_torques,
        joints_velocities=sensor_data_joints_velocities,
    )

    # Load Controller
    controller_file = os.path.join(
        BASE_PATH,
        f"controller_twl{twl:0.3f}_amp{amp:0.3f}.pkl",
    )
    with open(controller_file, "rb") as f:
        controller_data = pickle.load(f)

    indices = controller_data["indices"]
    neural_signals = (
        controller_data["state"][:, indices['left_body_idx']]
        - controller_data["state"][:, indices['right_body_idx']]
    )
    neural_signals_smoothed = filter_signals(
        times=sim_times, signals=neural_signals)
    signal_freqs, _, _ = compute_frequency_amplitude_fft(
        times=sim_times,
        smooth_signals=neural_signals_smoothed,
    )
    inds_couples = [[i, i + 1]
                    for i in range(neural_signals_smoothed.shape[1] - 1)]
    _, ipls_mean = compute_neural_phase_lags(
        times=sim_times,
        smooth_signals=neural_signals_smoothed,
        freqs=signal_freqs,
        inds_couples=inds_couples,
    )

    return speed_forward, cot, float(ipls_mean)


def exercise1_2(**kwargs):
    """ex1.2 main"""
    os.makedirs(PLOT_PATH, exist_ok=True)
    base_controller = {
        'loader': 'cmc_controllers.wave_controller.WaveController',
        'config': {
            'freq': 1.5,
            'twl': 1.0,
            'amp': 1.0}}
    pylog.warning("TODO: 1.2 Adapt the parameter space according to needs.")
    # Hint: You don't need to test all combinations of parameters with complexity of O(n^3)
    # You can replace range with list of length 1 to keep some parameters fixed
    # while testing others O(n^2) or O(n)

    example_twl_range = np.linspace(0.2, 1.2, 3)
    example_amp_range = np.linspace(1.0, 3.0, 3)
    parameter_grid_example = {
        'twl': example_twl_range,
        'amp': example_amp_range,
    }
    
    grid_size = 10
    twl_range = np.linspace(0.2, 1.5, grid_size)
    amp_range = np.linspace(1.0, 4.0, grid_size)
    parameter_grid = {
        'twl': twl_range,
        'amp': amp_range,
    }

    run_multiple(
        max_workers=MAX_WORKERS,
        controller=base_controller,
        base_path=BASE_PATH,
        parameter_grid=parameter_grid,
        common_kwargs={'fast': True, 'headless': True},
    )

    pylog.warning("TODO: 1.3 Analyze the results of multiple simulations")
    
    ##Getting the data from "get_matrics into grid" 
    # Rows are amp, columns are TWL
    fw_speed_grid = np.zeros((grid_size, grid_size))
    CoT_grid = np.zeros((grid_size, grid_size))
    average_IPL_grid = np.zeros((grid_size, grid_size))

    for i, amp in enumerate(amp_range):
        for j, twl in enumerate(twl_range):
                speed, cot, ipl = get_metrics(twl=twl, amp=amp)
                fw_speed_grid[i, j] = speed
                CoT_grid[i, j] = cot
                average_IPL_grid[i, j] = ipl
                
    ##PLots the 2D-grid parameter investigation 
    
    X, Y = np.meshgrid(twl_range, amp_range)
    plt.figure(figsize=(12, 4))
    
    #Forward Speed
    plt.subplot(1, 3, 1)
    plt.pcolormesh(X, Y, fw_speed_grid) 
    plt.colorbar(label='Forward Speed (m/s)')
    plt.xlabel('Total Wave Lag (TWL)')
    plt.ylabel('Amplitude (A)')
    plt.title('Forward Speed')
    plt.gca().set_box_aspect(1)
    #CoT
    plt.subplot(1, 3, 2)
    plt.pcolormesh(X, Y, CoT_grid)
    plt.colorbar(label='Cost of Transport (CoT)')
    plt.xlabel('Total Wave Lag (TWL)')
    plt.ylabel('Amplitude (A)')
    plt.title('Cost of Transport')
    plt.gca().set_box_aspect(1)
    #IPL
    plt.subplot(1, 3, 3)
    plt.pcolormesh(X, Y, average_IPL_grid)
    plt.colorbar(label='Phase Lag (rad)')
    plt.xlabel('Total Wave Lag (TWL)')
    plt.ylabel('Amplitude (A)')
    plt.title('Intersegmental Phase Lag (IPL)')
    plt.gca().set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "Q_1_3_2D_grid_param_investigation")) 
    
    #plt.show()

if __name__ == '__main__':
    exercise1_2(plot=True)

