import h5py
import numpy as np
import corner
import matplotlib.pyplot as plt

def inspect_h5_file(filepath):
    """Prints the structure and contents of an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        def print_attrs(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}, keys: {list(obj.keys())}")
        f.visititems(print_attrs)

# Example usage:
# inspect_h5_file('your_file.h5')

filepath = '/cluster/home/veoehl/xilikelihood/papers/second_paper_2025/analysis/sampler_results_10_l20.h5'
inspect_h5_file(filepath)

def read_sampler_data(filepath):
    """Reads sampler data from an HDF5 file and returns the posterior points and weights."""
    with h5py.File(filepath, 'r') as f:
        sampler_data = f['sampler']
        points = sampler_data['points_0'][:]
        log_l = sampler_data['log_l_0'][:]
        
    return points, log_l

params = ["omega_m", "s8"]
points, log_l = read_sampler_data(filepath)
plt.figure(figsize=(10, 8))
plt.plot(points[:, 0], points[:, 1], 'o', markersize=2, alpha=0.5
         , label='Posterior points')
plt.xlabel(params[0])
plt.ylabel(params[1])
plt.title('Posterior Points from Sampler')
plt.savefig('posterior_points.png')
corner.corner(
    points, bins=20, labels=params, color='purple',
    plot_datapoints=False, range=np.repeat(0.999, len(params)))

plt.savefig('corner_plot.png')