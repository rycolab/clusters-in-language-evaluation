"""
A categorical model with laplace smoothing
"""

import math
import numpy as np


class Categorical:
    probs = None

    def __init__(self, num_classes, laplace_smoothing=1):
        self.num_classes = num_classes
        self.laplace_smoothing = laplace_smoothing

    def train(self, labels):
        q_bins, _ = np.histogram(labels, bins=self.num_classes,
                                 range=[0, self.num_classes], density=False)
        q_bins = q_bins + self.laplace_smoothing
        self.probs = q_bins / q_bins.sum()

    def score_instance(self, label):
        assert self.probs is not None, \
            'Categorical model needs to be trained before used.'
        if self.probs[label] == 0:
            return float('inf')

        return math.log(self.probs[label])
