"""
DEPRECATED
Contains utility function to draw examples of blobs.

Because everything is done as Bezier curves, we classify each blob using its
four control points.

Based on: http://www.indiana.edu/~pcl/rgoldsto/curvemorph/bezier-curve.html
"""
from __future__ import division

import numpy as np

# Constants
MAX_DIF = 100 # Maximum dificulty of example.

NUM_CONTROL_PTS = 4 # Number of control points needed to classify a blob.
INTRABLOB_DIST = 500 # Distance between blobs.
CURSOR_RANGE = (450, 450 + INTRABLOB_DIST) # Valid range for cursor to be in.
OFFSET = 1.7
# Ref points are the control points of the four reference blobs.
X_REF_POINTS = ((142, 255, 362, 137),
                (177.35536251396132, 219.6447313046678, 276.64454366783457, 137),
                (106.6447313046678, 219.64454366765852, 397.35497585721964, 137),
                (142.00009381862913, 184.28927497232633, 311.9995195250542, 137))
Y_REF_POINTS = ((58, 239, 479, 286),
                (93.35531560467788, 274.3554094231825, 443.6449108935654, 286),
                (93.35540942318252, 203.6447782140757, 393.64449675895787, 286),
                (128.7107250278604, 239.0001876372582, 358.28940765252327, 286))

def get_example(is_pos, dif):
    """
    Given a dificulty, draw a random example with that dificulty. Here dificulty
    is measured in L2 distance away from the midpoint of the furthermost edge of
    the square for each classification.
    Args:
        is_pos: Boolean for whether we want a positive or negative example.
        dif: Dificuly ranging from 0 to INTRABLOB_DIST / 2.
    Returns: A numpy array of the coordinates for the four control points that
        classify the example. i.e. [x_0, y_0, x_1, y_1, ...]
    """
    coord = _get_cursor_coords(is_pos, dif)
    return _get_ctrl_pts(coord)

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

def _get_ctrl_pts(cursor):
    """Get the control points of the generated blob given the mouse's position.
    This is mostly adapted from the code found on the website.
    Args:
        cursor: A tuple for the x,y position of the cursor. Must be within
            CURSOR_RANGE.
    Returns: A numpy array of the coordinates for the four control points.
        i.e. [x_0, y_0, x_1, y_1, ...]
    """
    if cursor[0] < CURSOR_RANGE[0] or cursor[1] < CURSOR_RANGE[0]:
        raise AttributeError('Cursor value out of range.')
    if cursor[0] > CURSOR_RANGE[1] or cursor[1] > CURSOR_RANGE[1]:
        raise AttributeError('Cursor value out of range.')
    to_return = []
    for j in xrange(NUM_CONTROL_PTS):
        # Calculate horizontal rectangle.
        hx_m = (X_REF_POINTS[1][j] - X_REF_POINTS[0][j]) / INTRABLOB_DIST
        hx_b = X_REF_POINTS[0][j] - hx_m * INTRABLOB_DIST
        hx_rec = (cursor[0] - 100) * OFFSET * hx_m + hx_b

        hy_m = (Y_REF_POINTS[1][j] - Y_REF_POINTS[0][j]) / INTRABLOB_DIST
        hy_b = Y_REF_POINTS[0][j] - hy_m * INTRABLOB_DIST
        hy_rec = (cursor[0] - 100) * OFFSET * hy_m + hy_b

        vx_m = (X_REF_POINTS[2][j] - X_REF_POINTS[0][j]) / INTRABLOB_DIST
        vx_b = X_REF_POINTS[2][j] - vx_m * INTRABLOB_DIST
        vx_rec = (cursor[1] - 100) * OFFSET * vx_m + vx_b

        vy_m = (Y_REF_POINTS[1][j] - Y_REF_POINTS[0][j]) / INTRABLOB_DIST
        vy_b = Y_REF_POINTS[0][j] - vy_m * INTRABLOB_DIST
        vy_rec = (cursor[1] - 100) * OFFSET * vy_m + vy_b

        to_return.append((hx_rec + vx_rec) / 2)
        to_return.append((hy_rec + vy_rec) / 2)
    to_return = np.sqrt(np.array(to_return))
    return to_return
