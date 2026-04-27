#!/usr/bin/env python3
"""Run exercise 2.2 parameter sweeps and generate heatmaps/trajectory plots."""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from farms_core import pylog

from cmc_controllers.metrics import (
    compute_mechanical_energy_and_cot,
    compute_mechanical_speed,
    compute_trajectory_curvature,
    LINKS_MASSES,
    TOTAL_MASS
)

# Multiprocessing
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from simulate import run_multiple
MAX_WORKERS = 8  # adjust based on your hardware capabilities

# CPG parameters
BASE_PATH = 'logs/exercise2_2/'
PLOT_PATH = 'results'


def load_metrics_from_hdf5(hdf5_path):
    """Load speed and CoT metrics from an HDF5 simulation result."""
    with h5py.File(hdf5_path, "r") as f:
        sim_times = f['times'][:]
        sensor_data_links = f['FARMSLISTanimats']['0']['sensors']['links']['array'][:]
        sensor_data_joints = f['FARMSLISTanimats']['0']['sensors']['joints']['array'][:]

    links_positions = sensor_data_links[:, :, 7:10]
    links_velocities = sensor_data_links[:, :, 14:17]
    joints_velocities = sensor_data_joints[:, :, 1]
    joints_torques = sensor_data_joints[:, :, 2]
    
    

    speed_forward, speed_lateral = compute_mechanical_speed(
        links_positions=links_positions,
        links_velocities=links_velocities,
    )
    _, cot = compute_mechanical_energy_and_cot(
        times=sim_times,
        links_positions=links_positions,
        joints_torques=joints_torques,
        joints_velocities=joints_velocities,
    )

    com_x = (links_positions[:, :, 0] @ LINKS_MASSES) / TOTAL_MASS
    com_y = (links_positions[:, :, 1] @ LINKS_MASSES) / TOTAL_MASS
    com_pos = np.array([com_x, com_y]).T
    dt = np.mean(np.diff(sim_times))
    
    curvature_mean = compute_trajectory_curvature(com_pos, dt)

    return speed_forward, speed_lateral, cot, curvature_mean, com_pos


def exercise2_2(**kwargs):
    #pylog.warning("TODO: 2.2: Explore the effect of drive parameters and body phase bias")
    # pylog.set_level('critical')
    base_controller = {
        'loader': 'cmc_controllers.CPG_controller.CPGController',
        'config': {
            'drive_left': 3.0,
            'drive_right': 3.0,
            'd_low': 1.0,
            'd_high': 5.0,
            'a_rate': np.ones(8) * 3,
            'offset_freq': np.ones(8) * 1,
            'offset_amp': np.ones(8) * 0.5,
            'G_freq': np.ones(8) * 0.5,
            'G_amp': np.ones(8) * 0.25,
            'PL': np.ones(7) * (2 * np.pi / 8),
            'coupling_weights_rostral': 5,
            'coupling_weights_caudal': 5,
            'coupling_weights_contra': 10,
            'init_phase': np.random.default_rng(seed=42).uniform(0.0, 2 * np.pi, size=16)
        }
    }
   
    #----- Grid 1 Parameters
    grid_size = 8
    drive_range = np.linspace(2.0, 4.0, grid_size)
    pl_range = np.linspace(np.pi / 16, 3 * np.pi / 8, grid_size)
    
    # We must generate the list of arrays for PL
    pl_arrays = [np.ones(7) * pl for pl in pl_range]

    parameter_grid_1 = {
        'drive': drive_range,
        'PL': pl_arrays,
    }
    """
    print("Running Grid 1: Symmetric Drive vs Phase Lag...")
    run_multiple(
        max_workers=MAX_WORKERS,
        controller=base_controller,
        base_path=BASE_PATH + 'grid1/',
        parameter_grid=parameter_grid_1,
        common_kwargs={'fast': True, 'headless': True},
    )
    """
    
    #----- Grid 2 Parameters
    grid_size = 8
    drive_range = np.linspace(2.0, 4.0, grid_size)

    parameter_grid_2 = {
        'drive_left': drive_range,
        'drive_right': drive_range,
    }

    """
    print("Running Grid 2: left drive vs right drive")
    run_multiple(
        max_workers=MAX_WORKERS,
        controller=base_controller,
        base_path=BASE_PATH + 'grid2/',
        parameter_grid=parameter_grid_2,
        common_kwargs={'fast': True, 'headless': True},
    )
    """


    print("Extracting data and generating plots...")
    
    
    """
    fw_speed_grid = np.zeros((grid_size, grid_size))
    cot_grid = np.zeros((grid_size, grid_size))

    for i, d in enumerate(drive_range):
        for j, pl_array in enumerate(pl_arrays):
            params = {'drive': d, 'PL': pl_array}
            
            from simulate import _build_default_output_names
            hdf5_name, _ = _build_default_output_names(params)
            hdf5_path = os.path.join(BASE_PATH + 'grid1/', hdf5_name)
            
            # 2. FIXED: Try/Except block to gracefully handle missing simulation files
            try:
                speed_fwd, speed_lat, cot = load_metrics_from_hdf5(hdf5_path)
                fw_speed_grid[i, j] = speed_fwd
                cot_grid[i, j] = cot
            except FileNotFoundError:
                print(f"Warning: File missing for Drive={d:.2f}, PL={pl_range[j]:.3f}. Marking as NaN.")
                fw_speed_grid[i, j] = np.nan
                cot_grid[i, j] = np.nan
            
    # Plotting
    X, Y = np.meshgrid(pl_range, drive_range)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X, Y, fw_speed_grid, shading='auto', cmap='viridis')
    plt.colorbar(label='Forward Speed (m/s)')
    plt.xlabel('Phase Lag (rad)')
    plt.ylabel('Symmetric Drive')
    plt.title('Forward Speed')
    
    plt.subplot(1, 2, 2)
    plt.pcolormesh(X, Y, cot_grid, shading='auto', cmap='viridis')
    plt.colorbar(label='Cost of Transport')
    plt.xlabel('Phase Lag (rad)')
    plt.ylabel('Symmetric Drive')
    plt.title('Cost of Transport')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "Q_2_2_Grid1.png"))
    print(f"Grid 1 completed! Plot saved to {PLOT_PATH}/Q_2_2_Grid1.png")
    """
    from simulate import _build_default_output_names

    #-----------Grid1
    print("Extracting data for Grid 1...")
    fw_speed_grid_1 = np.zeros((grid_size, grid_size))
    cot_grid_1 = np.zeros((grid_size, grid_size))

    for i, d in enumerate(drive_range):
        for j, pl_array in enumerate(pl_arrays):
            params = {'drive': d, 'PL': pl_array}
            hdf5_name, _ = _build_default_output_names(params)
            hdf5_path = os.path.join(BASE_PATH + 'grid1/', hdf5_name)
            
            try:
                speed_fwd, speed_lateral, cot, curvature_mean, com_pos = load_metrics_from_hdf5(hdf5_path)
                fw_speed_grid_1[i, j] = speed_fwd
                cot_grid_1[i, j] = cot
            except FileNotFoundError:
                print(f"Warning: File missing for Drive={d:.2f}, PL={pl_range[j]:.3f}. Marking as NaN.")
                fw_speed_grid_1[i, j] = np.nan
                cot_grid_1[i, j] = np.nan
            
    X1, Y1 = np.meshgrid(pl_range, drive_range)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X1, Y1, fw_speed_grid_1, shading='nearest', cmap='viridis')
    plt.colorbar(label='Forward Speed (m/s)')
    # Add text to each square
    for i in range(grid_size):
        for j in range(grid_size):
            if not np.isnan(fw_speed_grid_1[i, j]):
                plt.text(pl_range[j], drive_range[i], f'{fw_speed_grid_1[i, j]:.4f}', 
                         ha='center', va='center', color='black', fontsize=5)
    plt.xlabel('Phase Lag (rad)')
    plt.ylabel('Symmetric Drive')
    plt.title('Grid 1: Forward Speed')
    
    plt.subplot(1, 2, 2)
    plt.pcolormesh(X1, Y1, cot_grid_1, shading='nearest', cmap='viridis')
    plt.colorbar(label='Cost of Transport')
    # Add text to each square
    for i in range(grid_size):
        for j in range(grid_size):
            if not np.isnan(cot_grid_1[i, j]):
                plt.text(pl_range[j], drive_range[i], f'{cot_grid_1[i, j]:.1f}', 
                         ha='center', va='center', color='white', fontsize=7)
    plt.xlabel('Phase Lag (rad)')
    plt.ylabel('Symmetric Drive')
    plt.title('Grid 1: Cost of Transport')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "Q_2_2_Grid1.png"))
    print(f"Grid 1 plot saved to {PLOT_PATH}/Q_2_2_Grid1.png")

    #-----------Grid2
        
    print("Extracting data for Grid 2...")
    curvature_grid_2 = np.zeros((grid_size, grid_size))
    
    # We will store all 64 Trajectories in a 2D list
    trajectories_grid_2 = [[None for _ in range(grid_size)] for _ in range(grid_size)]

    for i, dl in enumerate(drive_range):
        for j, dr in enumerate(drive_range):
            params = {'drive_left': dl, 'drive_right': dr}
            hdf5_name, _ = _build_default_output_names(params)
            hdf5_path = os.path.join(BASE_PATH + 'grid2/', hdf5_name)
            
            try:
                # We extract all 5, but care about curvature and trajectory for Grid 2
                speed_forward, speed_lateral, cot, curvature_mean, com_pos = load_metrics_from_hdf5(hdf5_path)
                curvature_grid_2[i, j] = curvature_mean
                trajectories_grid_2[i][j] = com_pos
            except FileNotFoundError:
                curvature_grid_2[i, j] = np.nan

    X2, Y2 = np.meshgrid(drive_range, drive_range)
    plt.figure(figsize=(14, 6)) # Made slightly wider to fit the square trajectory box
    
    # Curvature Heatmap
    plt.subplot(1, 2, 1)
    plt.pcolormesh(X2, Y2, curvature_grid_2, shading='nearest', cmap='plasma')
    plt.colorbar(label='Trajectory Curvature (1/m)')
    for i in range(grid_size):
        for j in range(grid_size):
            if not np.isnan(curvature_grid_2[i, j]):
                plt.text(drive_range[j], drive_range[i], f'{curvature_grid_2[i, j]:.2f}', 
                         ha='center', va='center', color='white', fontsize=7)
    plt.xlabel('Drive Right')
    plt.ylabel('Drive Left')
    plt.title('Grid 2: Trajectory Curvature Heatmap')

    # -- 2. All 64 Trajectories (The Fan Plot) --
    ax2 = plt.subplot(1, 2, 2)
    for i in range(grid_size):
        for j in range(grid_size):
            traj = trajectories_grid_2[i][j]
            if traj is not None:
                # Mathematically shift every trajectory so they all start exactly at (0,0)
                x_shifted = traj[:, 0] - traj[0, 0]
                y_shifted = traj[:, 1] - traj[0, 1]
                
                # Plot the path!
                ax2.plot(x_shifted, y_shifted, alpha=0.5, linewidth=1.5)
                
    plt.xlabel('Lateral Position X (m)')
    plt.ylabel('Forward Position Y (m)')
    plt.title('Grid 2: Center of Mass Trajectories (64 Simulations)')
    plt.axis('equal') # Forces the X and Y axes to scale perfectly together so turns aren't distorted
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_PATH, "Q_2_2_Grid2_Curvature_Trajectories.png"))
    print(f"Grid 2 plots saved successfully!")
    
    plot = kwargs.pop('plot', False)
    if plot:
        plt.show()


if __name__ == '__main__':
    exercise2_2(plot=True)

