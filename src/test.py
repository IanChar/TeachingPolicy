"""
Test out functions to see if they are working correctly.
"""
import numpy as np
import matplotlib.pyplot as plt

import blob_generation

def _test_get_cursor_coords():
    samples = 10000
    for is_pos in [True, False]:
        difs, xs, ys = [], [], []
        for _ in xrange(samples):
            dif = np.random.uniform(0, 100)
            x, y = blob_generation._get_cursor_coords(is_pos, dif)
            difs.append(dif)
            xs.append(x)
            ys.append(y)
        plt.scatter(xs, ys, c=difs)
        plt.xlim((450, 950))
        plt.ylim((450, 950))
    plt.show()

if __name__ == '__main__':
    _test_get_cursor_coords()
