import numpy as np


def sample_klein_bottle_3d(n_samples, r=2):
    """
    Samples points from a 3D Figure-8 Klein Bottle.
    """
    # Randomly sample parameters u and v from 0 to 2*pi
    u = np.random.uniform(0, 2 * np.pi, n_samples)
    v = np.random.uniform(0, 2 * np.pi, n_samples)

    # Parametric equations for the Figure-8 Klein bottle
    x = (r + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)) * np.cos(u)
    y = (r + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)) * np.sin(u)
    z = np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v)

    # Stack into a shape of (n_samples, 3)
    return np.vstack((x, y, z)).T


# 1. Generate the dataset
klein_data = sample_klein_bottle_3d(400)
np.savetxt("klein_400.txt", klein_data)
klein_data = sample_klein_bottle_3d(900)
np.savetxt("klein_900.txt", klein_data)
