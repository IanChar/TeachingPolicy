"""
Test out functions to see if they are working correctly.
"""
import numpy as np
import matplotlib.pyplot as plt

import blob_generation
from student import Student
import plan_eval

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

def _test_student():
    stu = Student()
    ex1 = np.array([0 for _ in xrange(8)])
    correct = stu.feed_example(ex1, False)
    print correct, stu.examples, stu.record, stu.time
    ex2 = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    correct = stu.feed_example(ex2, True)
    print correct, stu.examples, stu.record, stu.time
    ex3 = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    correct = stu.feed_example(ex3, True)
    print correct, stu.examples, stu.record, stu.time
    ex4 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    correct = stu.feed_example(ex4, False)
    print correct, stu.examples, stu.record, stu.time

def _test_evaluate_plan():
    students = [Student()]
    test_qs, test_ans = plan_eval.gen_unif_test(1000)
    print plan_eval.evaluate_plan(5, 50, students, 25, test_qs, test_ans)

def _test_gen_read_test():
    plan_eval.gen_unif_test(10, 'test.txt')
    print plan_eval.read_test('test.txt')

if __name__ == '__main__':
    # _test_get_cursor_coords()
    # _test_student()
    # _test_evaluate_plan()
    _test_gen_read_test()
