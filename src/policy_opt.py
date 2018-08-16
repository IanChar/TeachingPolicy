"""
Bayesian optimization using
https://github.com/fmfn/BayesianOptimization
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq
import scipy
import scipy.interpolate

from bayes_opt import BayesianOptimization

import plan_eval
from student import Student

TEST_PATH = '100test.txt'
ALPHA_MAX = 50
BETA_MAX = 50
STUDENTS = [Student() for _ in xrange(1)]

def optimize(trials, alpha_0=10, beta_0=10, num_exs=15, students=None,
             test_path=TEST_PATH, make_plot=True):
    """
    Find the optimal teaching plan that maximizes student scores for a set
    number of examples. Prints out optimal [alpha, beta] in terminal.
    Args:
        trials: The number of teaching plans to try out after init_points.
        alpha_0: The first alpha to try.
        beta_0: The first beta to try.
        num_exs: The number of positive and negative examples to show students.
        students: List of student objects to teach where the average score
            among students is considered.
        test_path: The file path for the test to consider.
        make_plot: Whether to make a plot of the results.
    """
    if students is None:
        students = STUDENTS
    test_qs, test_ans = plan_eval.read_test(test_path)
    def target_func(alpha, beta):
        avg, var = plan_eval.evaluate_plan(alpha, beta, students, num_exs,
                                           test_qs, test_ans)
        # Wipe Students memory
        for s in students:
            s.wipe_memory()
        return avg
    # Do Bayes' optimization.
    bo = BayesianOptimization(target_func, {'alpha': (0, ALPHA_MAX),
                                            'beta': (0, BETA_MAX)})
    bo.explore({'alpha': [alpha_0], 'beta': [beta_0]})
    bo.maximize(init_points=5, n_iter=trials, acq='ucb', kappa=2)

    max_val = bo.res['max']
    best = (float(max_val['max_val']),
            float(max_val['max_params']['alpha']),
            float(max_val['max_params']['beta']))
    if make_plot:
        print bo.gp.predict(np.array([[10, 3], [10, 5]]))
        alphas = [float(val['alpha']) for val in bo.res['all']['params']]
        betas = [float(val['beta']) for val in bo.res['all']['params']]
        vals = [float(v) for v in bo.res['all']['values']]
        hist = [(vals[i], alphas[i], betas[i]) for i in xrange(len(alphas))]
        plot_history(best, hist, bo)
    return best

def optimize_fastest(trials, alpha_0=10, beta_0=10, ex_cutoff=25,
                     perf_thresh=0.8,students=None, test_path=TEST_PATH,
                     make_plot=True):
    """
    Find the optimal teaching policy that will reach a threshold of performance
    with the least amount of examples.
    Args:
        trials: The number of teaching plans to try out after init_points.
        alpha_0: The first alpha to try.
        beta_0: The first beta to try.
        ex_cutoff: The maximum number of examples to try.
        perf_thresh: The performance threshold to achieve.
        students: List of student objects to teach where the average score
            among students is considered.
        test_path: The file path for the test to consider.
        make_plot: Whether to make a plot of the results.
    """
    if students is None:
        students = STUDENTS
    test_qs, test_ans = plan_eval.read_test(test_path)
    def target_func(alpha, beta):
        avg, var = plan_eval.teach_until_perfect(alpha, beta, students,
                                                 test_qs, test_ans, ex_cutoff,
                                                 perf_thresh)
        # Wipe Students memory
        for s in students:
            s.wipe_memory()
        return -1 * avg # Multiply by negative 1 to find min
    # Do Bayes' optimization.
    bo = BayesianOptimization(target_func, {'alpha': (0, ALPHA_MAX),
                                            'beta': (0, BETA_MAX)})
    bo.explore({'alpha': [alpha_0], 'beta': [beta_0]})
    bo.maximize(init_points=5, n_iter=trials, acq='ucb', kappa=2)

    max_val = bo.res['max']
    best = (-1 * float(max_val['max_val']),
            float(max_val['max_params']['alpha']),
            float(max_val['max_params']['beta']))
    if make_plot:
        print bo.gp.predict(np.array([[10, 3], [10, 5]]))
        alphas = [float(val['alpha']) for val in bo.res['all']['params']]
        betas = [float(val['beta']) for val in bo.res['all']['params']]
        vals = [-1 * float(v) for v in bo.res['all']['values']]
        hist = [(vals[i], alphas[i], betas[i]) for i in xrange(len(alphas))]
        plot_history(best, hist, bo)
    return best


def plot_history(best, history, bo):
    """
    Make a scatter plot of the points that we tried and the best point.
    Args:
        best: Best point as (score, alpha, beta).
        history: list of points of the aove form that were tried.
    """
    scores = [h[0] for h in history]
    alphas = [h[1] for h in history]
    betas = [h[2] for h in history]
    a = heapq.nsmallest(3, history)
    xi, yi = np.linspace(0, ALPHA_MAX, 500), np.linspace(0, BETA_MAX, 500)
    xi, yi = np.meshgrid(xi, yi)
    pts = np.zeros((10 ** 2, 2))
    ind = 0
    for x in np.linspace(0, ALPHA_MAX, 10):
        for y in np.linspace(0, BETA_MAX, 10):
            pts[ind,:] = [x, y]
            ind += 1
    ao = bo.gp.predict(pts)
    rbf = scipy.interpolate.Rbf(pts[:,0], pts[:,1], ao, function='linear')
    ai = rbf(xi, yi)
    plt.imshow(ai, vmin=min(ao), vmax=max(ao), origin='lower', extent=[0, ALPHA_MAX, 0, BETA_MAX], cmap='inferno')
    plt.scatter(alphas, betas, facecolors='none', edgecolors='green')
    plt.plot(a[0][1], a[0][2], 'y*', markersize=12, color='gold')
    plt.plot(a[1][1], a[1][2], 'y*', markersize=12, color='silver')
    plt.plot(a[2][1], a[2][2], 'y*', markersize=12, color='brown')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.xlim((0, ALPHA_MAX))
    plt.ylim((0, BETA_MAX))
    plt.show()

if __name__ == '__main__':
    optimize(5)
