"""
Use Gaussian Processes to find optimum teaching policy.
"""

from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint

import plan_eval
from student import Student

TEST_PATH = '100test.txt'
STUDENTS = [Student() for _ in xrange(10)]

def find_best_policy(trials, alpha_0=10, beta_0=50, num_exs=25, students=None,
                     test_path=TEST_PATH):
    """
    Find the best alpha/beta for the teaching policy.
    Args:
        trials: Number of teaching plans to try out (including first plan).
        alpha_0: Starting alpha.
        beta_0: Starting beta.
        num_exs: The number of examples given to students per teaching plan.
        students: The students to teach, will default to STUDENTS if non given.
        test_path: The path of the file with test qs/answers.
    Returns: The best alpha/beta found.
    """
    if students is None:
        students = STUDENTS
    test_qs, test_ans = plan_eval.read_test(test_path)
    history = []
    eval_policy = _create_evaluator(num_exs, students, test_qs, test_ans, history)

    experiment = Experiment([[0, 100], [0, 100]])
    # Run the start experiment and evaluate.
    experiment.historical_data.append_sample_points([eval_policy(alpha_0, beta_0)])
    for i in xrange(trials - 1):
        print '--------TRIAL %d DONE--------' % (i + 1)
        alpha, beta = gp_next_points(experiment)[0]
        experiment.historical_data.append_sample_points([eval_policy(alpha, beta)])
    print history
    return max(history)

def _create_evaluator(num_exs, students, test_qs, test_ans, history):
    """
    Creates function that will return SamplePoint for given alpha, beta.
    """
    def evaluator(alpha, beta):
        avg, var = plan_eval.evaluate_plan(alpha, beta, students, num_exs,
                                           test_qs, test_ans)
        # Wipe Students memory
        for s in students:
            s.wipe_memory()
        # Since minimizes by default say score is 1 - avg
        history.append((avg, alpha, beta))
        score = 1 - avg
        return SamplePoint([alpha, beta], score, var)
    return evaluator

if __name__ == '__main__':
    best = find_best_policy(50)
    print '__________BEST ANSWER: ', best, '_____________'
