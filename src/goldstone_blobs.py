"""
Code for Goldstone blob generation inspired by Mike Mozer's code.
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

# Constants
MAX_DIF = 100 # Maximum dificulty of example.

INTRABLOB_DIST = 100 # Distance between blobs.
CURSOR_RANGE = (0, INTRABLOB_DIST) # Valid range for cursor to be in.

TOTAL_POINTS = 360
ANGLES = np.linspace(0, 2 * np.pi, TOTAL_POINTS)

NUM_TERMS = 8
FREQUENCIES = range(1, NUM_TERMS + 1)

PROTOS = np.array([[3, 5 * np.pi / 8],
                   [7, 2 * np.pi / 17],
                   [5, 3 * np.pi / 2],
                   [10, 7 * np.pi / 4]])

def gen_blob(params, num_samps, make_plot=False):
    """
    Generate the blob based on given params.
    Args:
        params:
            amp: Amplitude of the modes in the series.
            offset: Offset in the series.
        num_samps: Number of samples of points to return to return.
        make_plot: Whether to make a plot of the corresponding blob.
            Need to call plt.show() yourself.
    Returns: List of tuples representing points in the blob from 0 -> 2pi
    """
    amp, offset = params
    max_mag = 0
    points = np.zeros((TOTAL_POINTS, 2))
    for i in xrange(TOTAL_POINTS):
        mag = 0
        for j in xrange(NUM_TERMS):
            mag += amp * (1 + np.cos(offset + FREQUENCIES[j] * ANGLES[i]))
        points[i,:] = [np.sin(ANGLES[i]) * mag, np.cos(ANGLES[i]) * mag]
        max_mag = max([mag, max_mag])
    points /= max_mag
    if make_plot:
        plt.scatter(points[:, 0], points[:, 1])

    spacing = int(TOTAL_POINTS / num_samps)
    to_return = np.zeros((num_samps, 2))
    for ind, samp_ind in enumerate(range(0, TOTAL_POINTS, spacing)):
        to_return[ind,:] = points[samp_ind, :]
    return to_return

def _get_cursor_coords(is_pos, dif):
    """
    Get random cooordinates that fall into the correct range and have the
    correct difficulty.
    Args:
        is_pos: Boolean for whether we want a positive or negative example.
    Returns: (x, y) cursor coordinate.
    """
    if dif < 0 or dif > MAX_DIF:
        raise AttributeError('Invalid difficulty.')
    # Convert dificulty into a distance in coordinate space.
    half_dist = INTRABLOB_DIST / 2
    dif_dist = dif * (half_dist / MAX_DIF) * np.sqrt(2)
    # Draw a random angle away from the anchor.
    theta_lower = 0
    if dif_dist > half_dist:
        theta_lower = np.arccos(half_dist / dif_dist)
    theta = np.random.uniform(theta_lower, np.pi / 2 - theta_lower)
    # Get vector.
    x_dif = abs(dif_dist * np.cos(theta))
    x_dif = -1 * x_dif if np.random.uniform() < 0.5 else x_dif
    y_dif = abs(dif_dist * np.sin(theta))
    y_dif = -1 * y_dif if is_pos else y_dif
    # Calculate points
    return (CURSOR_RANGE[0] + half_dist + x_dif,
            CURSOR_RANGE[is_pos] + y_dif)

if __name__ == '__main__':
    plt.figure(1)
    posns = [221, 222, 223, 224]
    for plot_num in xrange(4):
        plt.subplot(posns[plot_num])
        print gen_blob(PROTOS[plot_num,:], 36, True)
    plt.show()
