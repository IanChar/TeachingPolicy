"""
Use Gaussian Processes to find optimum teaching policy.
"""

from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint
import matplotlib.pyplot as plt
import numpy as np
import heapq
import scipy

import plan_eval
from student import Student

TEST_PATH = '100test.txt'
ALPHA_MAX = 50
BETA_MAX = 50
STUDENTS = [Student() for _ in xrange(50)]

def find_fastest_policy(trials, alpha_0=10, beta_0=10, ex_cutoff=250, perf_thresh=0.93,
                        students=None, test_path=TEST_PATH, make_plot=True):
    """
    Find the best alpha/beta for the teaching policy.
    Args:
        trials: Number of teaching plans to try out (including first plan).
        alpha_0: Starting alpha.
        beta_0: Starting beta.
        ex_cutoff: Max number of examples to show.
        perf_thresh: The threshold of what is considered perfect.
        students: The students to teach, will default to STUDENTS if non given.
        test_path: The path of the file with test qs/answers.
        make_plot: Whether to make a scatter plot of the history.
    Returns: The best alpha/beta found.
    """
    if students is None:
        students = STUDENTS
    test_qs, test_ans = plan_eval.read_test(test_path)
    history = []
    eval_policy = _create_perf_evaluator(ex_cutoff, perf_thresh, students, test_qs, test_ans, history)

    experiment = Experiment([[0, ALPHA_MAX], [0, BETA_MAX]])
    # Run the start experiment and evaluate.
    experiment.historical_data.append_sample_points([eval_policy(alpha_0, beta_0)])
    for i in xrange(trials-1):
        print '--------TRIAL %d DONE--------' % (i + 1)
        alpha, beta = gp_next_points(experiment)[0]
        experiment.historical_data.append_sample_points([eval_policy(alpha, beta)])
    best = min(history)
    print len(history)
    print len(history)

    if make_plot:
        plot_history(min(history), history)
    return best

def find_accurate_policy(trials, alpha_0=10, beta_0=10, num_exs=250, students=None,
                         test_path=TEST_PATH, make_plot=False):
    """
    Find the best alpha/beta for the teaching policy.
    Args:
        trials: Number of teaching plans to try out (including first plan).
        alpha_0: Starting alpha.
        beta_0: Starting beta.
        num_exs: The number of examples given to students per teaching plan.
        students: The students to teach, will default to STUDENTS if non given.
        test_path: The path of the file with test qs/answers.
        make_plot: Whether to make a scatter plot of the history.
    Returns: The best alpha/beta found.
    """
    if students is None:
        students = STUDENTS
    test_qs, test_ans = plan_eval.read_test(test_path)
    history = []
    eval_policy = _create_evaluator(num_exs, students, test_qs, test_ans, history)

    experiment = Experiment([[0, ALPHA_MAX], [0, BETA_MAX]])
    # Run the start experiment and evaluate.
    experiment.historical_data.append_sample_points([eval_policy(alpha_0, beta_0)])
    for i in xrange(trials-1):
        print '--------TRIAL %d DONE--------' % (i + 1)
        alpha, beta = gp_next_points(experiment)[0]
        experiment.historical_data.append_sample_points([eval_policy(alpha, beta)])
    best = max(history)
    a = heapq.nlargest(1, el)
    print best
    print a
    print len(history)
    if make_plot:
        plot_history(max(history), history)
    return best

def _create_evaluator(num_exs, students, test_qs, test_ans, history):
    """
    Creates function that will return SamplePoint for given alpha, beta.
    """
    def evaluator(alpha, beta):
        avg, var = plan_eval.evaluate_plan(alpha, beta, students, num_exs,
                                           test_qs, test_ans)
        print alpha, beta, avg, var
        # Wipe Students memory
        for s in students:
            s.wipe_memory()
        # Since minimizes by default say score is 1 - avg
        history.append((avg, alpha, beta))
        score = 1 - avg
        return SamplePoint([alpha, beta], score, var)
    return evaluator

def _create_perf_evaluator(cut_off, perf_thresh, students, test_qs, test_ans, history):
    """
    Creates function that will return SamplePoint for given alpha, beta.
    """
    def evaluator(alpha, beta):
        avg, var = plan_eval.teach_until_perfect(alpha, beta, students,
                                                 test_qs, test_ans, cut_off, perf_thresh)
        print alpha, beta, avg, var
        # Wipe Students memory
        for s in students:
            s.wipe_memory()
        # Since minimizes by default say score is 1 - avg
        history.append((avg, alpha, beta))
        score = avg
        return SamplePoint([alpha, beta], score, var)
    return evaluator

def plot_history(best, history):
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
    # plt.scatter(alphas, betas, c=scores, cmap='inferno')
    # plt.plot(a[0][1], a[0][2], 'y*', markersize=12, color='gold')
    # plt.plot(a[1][1], a[1][2], 'y*', markersize=12, color='silver')
    # plt.plot(a[2][1], a[2][2], 'y*', markersize=12, color='brown')
    # plt.xlabel('Alpha')
    # plt.ylabel('Beta')
    # plt.xlim((0, ALPHA_MAX))
    # plt.ylim((0, BETA_MAX))
    # plt.show()

    xi, yi = np.linspace(0, ALPHA_MAX, 500), np.linspace(0, BETA_MAX, 500)
    xi, yi = np.meshgrid(xi, yi)
    xo=alphas
    yo=betas
    ao=scores
    rbf = scipy.interpolate.Rbf(xo, yo, ao, function='linear')
    ai = rbf(xi, yi)
    plt.imshow(ai, vmin=min(ao), vmax=max(ao), origin='lower', extent=[0, ALPHA_MAX, 0, BETA_MAX], cmap='inferno')
    plt.scatter(xo, yo, facecolors='none', edgecolors='green')
    plt.plot(a[0][1], a[0][2], 'y*', markersize=12, color='gold')
    plt.plot(a[1][1], a[1][2], 'y*', markersize=12, color='silver')
    plt.plot(a[2][1], a[2][2], 'y*', markersize=12, color='brown')
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.xlim((0, ALPHA_MAX))
    plt.ylim((0, BETA_MAX))
    plt.show()


if __name__ == '__main__':
    best = find_accurate_policy(10)
    print '__________BEST ANSWER: ', best, '_____________'
