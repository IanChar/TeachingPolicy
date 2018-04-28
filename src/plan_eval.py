"""
Given a teaching plan (alpha, beta), evaluate how well the plan works on
some given students.
"""

from __future__ import division
import numpy as np
import pickle

import blob_generation

def evaluate_plan(alpha, beta, students, num_examples, test_qs, test_ans):
    """
    First teach the students with a certain number of examples, then test.
    Args:
        alpha, beta: Parameters in alpha x + beta where x is the number of
            correct answers minus incorrect answers.
        students: The list of students to teach.
        num_examples: The number of examples to show each student.
        test_qs: List of questions for the test represented as feature vectors.
        test_ans: List of answers for the test represented as booleans.
    Returns: Tuple of the average score and variance on the final test.
    """
    scores = []
    for s in students:
        teach(s, alpha, beta, num_examples)
        scores.append(s.give_test(test_qs, test_ans))
    avg_score = sum(scores) / len(students)
    var_score = np.var(scores)
    return (avg_score, var_score)

def teach(student, alpha, beta, num_examples):
    """
    Teach a particular student.
    Args:
        alpha, beta: Parameters in alpha x + beta where x is the number of
            correct answers minus incorrect answers.
        students: The list of students to teach.
        num_examples: The number of examples to show each student.
    """
    score = 0
    ex_type = True
    max_dif = blob_generation.MAX_DIF
    hard_qs = (max_dif * 0.95, max_dif)
    easy_qs = (0, 0.05 * max_dif)
    for _ in xrange(num_examples):
        dif = alpha * score + beta
        dif = np.random.uniform(hard_qs[0], hard_qs[1]) if dif > 100 else dif
        dif = np.random.uniform(easy_qs[0], easy_qs[1]) if dif < 0 else dif
        ex = blob_generation.get_example(ex_type, dif)
        correct = student.feed_example(ex, ex_type)
        if correct:
            score += 1
        else:
            score -= 1
        ex_type = not ex_type

def gen_unif_test(num_qs, filepath=None):
    """
    Generate a test where dificulty is picked uniformly at random.
    Args:
        num_qs: Number of positive and negative questions respectively.
        filepath: If specified, writes the test to a file.
    Returns: (test_qs, test_ans) where test_qs is a list of feature vectors and
        test_ans is a list of boolean answers.
    """
    qs, ans = [], []
    for _ in xrange(num_qs):
        ans.append(True)
        dif = np.random.uniform(0, blob_generation.MAX_DIF)
        ex = blob_generation.get_example(True, dif)
        qs.append(ex)
    for _ in xrange(num_qs):
        ans.append(False)
        dif = np.random.uniform(0, blob_generation.MAX_DIF)
        ex = blob_generation.get_example(False, dif)
        qs.append(ex)
    if filepath is not None:
        with open(filepath, 'wb') as f:
            pickle.dump((qs, ans), f)
    return (qs, ans)

def read_test(filepath):
    """
    Read in a test given the filepath name.
    Args:
        filepath: The path to the file to read in.
    Returns: (test_qs, test_ans) where test_qs is a list of feature vectors and
        test_ans is a list of boolean answers.
    """
    with open(filepath, 'rb') as f:
        read = pickle.load(f)
        return read

if __name__ == '__main__':
    gen_unif_test(100, '100test.txt')
