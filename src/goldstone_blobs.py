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

PROTOS = np.array([[0.1, 7 * np.pi / 8, 1, 2, 3, 4, 5, 6, 7, 8],
                   [0.3, 3 * np.pi / 2, 2, 4, 6, 8, 10, 12, 14, 16],
                   [1.7, 14 * np.pi / 8, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75, 10],
                   [0.8, 3 * np.pi / 2, 2/3, 4/3, 6/3, 8/3, 10/3, 12/3, 14/3, 16/3]])

def get_example(is_pos, dificulty, num_samps=36, make_plot=False):
    """
    Get a blob example.
    Args:
        is_pos: If the example is positive or negative.
        dificulty: How hard the example should be (0, 100).
    Returns: A numpy array of the coordinates for the fsampled points that
        classify the example. i.e. [x_0, y_0, x_1, y_1, ...]
    """
    coords = _get_cursor_coords(is_pos, dificulty)
    params1 = _interp_params(PROTOS[0, :], PROTOS[1, :], coords[0] / 100)
    params2 = _interp_params(PROTOS[2, :], PROTOS[3, :], coords[0] / 100)
    params = _interp_params(params1, params2, coords[1] / 100)
    pts = gen_blob(params, num_samps, make_plot)
    pts = pts.ravel()
    return pts

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
    amp, offset = params[0], params[0]
    max_mag = 0
    points = np.zeros((TOTAL_POINTS, 2))
    for i in xrange(TOTAL_POINTS):
        mag = 0
        for j in xrange(int(NUM_TERMS / 2)):
            mag += amp * (1 + np.cos(offset + params[j + 2] * ANGLES[i]))
        for j in xrange(int(NUM_TERMS / 2), NUM_TERMS):
            mag += amp * (1 + np.sin(offset + params[j + 2] * ANGLES[i]))
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

def _interp_params(param1, param2, percent):
    """
    Interpolate parameters for the blob.
    Args:
        param1, param2: The two params [mag, offset] to interpolate between.
        percent: The percentage of param1 to have.
    Returns: [magnitude, offset]
    """
    param = np.zeros(param1.shape)
    for i in xrange(len(param1)):
        param[i] = param1[i] * percent + param2[i] * (1 - percent)
    return param

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

def plot_in_between():
    plt.figure(1)
    i = 1
    for xdim in xrange(0, 101, 25):
        for ydim in xrange(0, 101, 25):
            coords = [xdim, ydim]
            params1 = _interp_params(PROTOS[0, :], PROTOS[1, :], coords[0] / 100)
            params2 = _interp_params(PROTOS[2, :], PROTOS[3, :], coords[0] / 100)
            params = _interp_params(params1, params2, coords[1] / 100)
            plt.subplot(5, 5, i)
            gen_blob(params, 36, True)
            plt.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom='off',      # ticks along the bottom edge are off
                        top='off',         # ticks along the top edge are off
                        labelbottom='off') # labels along the bottom edge are off
            plt.tick_params(
                        axis='y',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        left='off',      # ticks along the bottom edge are off
                        right='off',         # ticks along the top edge are off
                        labelbottom='off') # labels along the bottom edge are off
            i += 1
    plt.show()

if __name__ == '__main__':
    # print get_example(True, 50, make_plot=True)
    # plt.show()
    # plt.figure(1)
    # posns = [221, 222, 223, 224]
    # for plot_num in xrange(4):
    #     plt.subplot(posns[plot_num])
    #     gen_blob(PROTOS[plot_num,:], 36, True)
    # plt.show()
    plot_in_between()
