"""
Code for the student AI simulated via a k-nearest-neighbor-like algorithm.
"""

from __future__ import division
import numpy as np
import heapq

# Constants
K = 3
SUCCESS_PROB = 0.4
FAIL_PROB = 1
POWER_DECREMENT = lambda p, t: p * t ** -0.5

class Student(object):

    def __init__(self, k=K, success_prob=SUCCESS_PROB, fail_prob=FAIL_PROB,
                 decrement=POWER_DECREMENT):
        self.time = 0
        # Tuple of whether positive or negative example and feature vector.
        self.examples = []
        # List of whether the student got corresponding examples correct.
        self.record = []
        # k in the k-nearest neighbors.
        self.k = k
        # Initial probability when example classified succesfuly.
        self.success_prob = success_prob
        # Initial probability when example classified incorrectly.
        self.fail_prob = fail_prob
        # How to decrement the probability an example is used as time goes by.
        self.decrement = decrement

    def feed_example(self, ex, is_pos):
        """
        Show the student an example, the student will first try to guess the
        answer. If answered correctly to example will have less weight.
        Args:
            ex: The feature vector for the example.
            is_pos: Whether the example is positive or not.
        Returns: Boolean whether the student got the example right.
        """
        # See if the student can answer the question right.
        guess = self._guess(ex)
        correct = guess is is_pos
        # Add example.
        self.examples.append((is_pos, ex))
        self.record.append(correct)
        self.time += 1
        return correct

    def give_test(self, exs, are_pos):
        """
        Give the student a final test. These points will not be remembered by
        the student.
        Args:
            exs: List of feature vectors representing the test questions.
            are_pos: List of booleans for if corresponding example is positive.
        Returns: Float from 0 - 1 for the percentage that the student answered
            correctly.
        """
        correct = 0
        total = len(exs)
        probs = self._get_remember_probs()
        for i in xrange(total):
            guess = self._guess(exs[i], probs)
            if guess is are_pos[i]:
                correct += 1
        return correct / total

    def _guess(self, f_vec, probs=None):
        """
        Have the student make a guess at the question.
        Args:
            f_vec: The example represented as a feature vector to classify.
            probs: Optional probability vector of whether previous examples
                will be considered or not.
        Returns: Boolean for whether the student this it is positive or not.
        """
        if probs is None:
            probs = self._get_remember_probs()
        num_exs = len(self.examples)
        # Construct a min heap.
        heap = []
        for i in xrange(num_exs):
            u = np.random.uniform()
            if u <= probs[i]:
                is_pos, ex = self.examples[i]
                dist = np.linalg.norm(ex - f_vec, 2)
                heapq.heappush(heap, (dist, is_pos))
        pos_votes, neg_votes = 0, 0
        for _ in xrange(self.k):
            if heapq.heappop(heap)[1]:
                pos_votes += 1
            else:
                neg_votes += 1
        return pos_votes >= neg_votes

    def _get_remember_probs(self):
        """
        Calculate probabilities the student will remember each example.
        Returns: List of probabilities.
        """
        probs = []
        for t, correct in enumerate(self.record):
            init_p = self.success_prob if correct else self.fail_prob
            probs.append(self.decrement(init_p, self.time - t))
        return probs
