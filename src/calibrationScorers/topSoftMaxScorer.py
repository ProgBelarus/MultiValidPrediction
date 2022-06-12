from calibrationScorers import calibrationScorer, predictionSet
import numpy as np

import pdb

class TopSoftMaxScorePredictionSet(predictionSet.PredictionSet):
    def __init__(self, ys):
        self.ys = ys

    def cover(self, y):
        return y in self.ys


class TopSoftMaxScorer(calibrationScorer.CalibrationScorer):
    def __init__(self, use_randomization=True):
        self.f_pred = None
        self.use_randomization = use_randomization

    def calc_score(self, x, y):
        y = np.array(y).reshape(-1, 1)
        x = np.array(x).reshape(len(y), -1)
        soft_maxes = np.array(self.f_pred(x)).reshape(len(y), -1)

        pi = (-soft_maxes).argsort(1)
        soft_maxes_sorted = np.take_along_axis(soft_maxes, pi, 1)
        soft_maxes_cumsum = soft_maxes_sorted.cumsum(1)
        y_to_cumulative_sum = np.take_along_axis(soft_maxes_cumsum, pi, 1)

        scores = np.take_along_axis(y_to_cumulative_sum, y, -1).ravel()

        return scores

    def get_prediction_set(self, x, calibration_threshold):
        x = x.reshape(1, -1)

        soft_maxes = np.array(self.f_pred(x))

        pi = (-soft_maxes).argsort()
        soft_maxes_sorted = np.take_along_axis(soft_maxes, pi, -1)

        soft_maxes_cumsum = soft_maxes_sorted.cumsum()

        size = np.sum(soft_maxes_cumsum < calibration_threshold, -1) + 1

        # pdb.set_trace()
        if self.use_randomization:
            prob_exceeds = (soft_maxes_cumsum[size-1] - calibration_threshold) / soft_maxes[0][pi[0][size-1]]
            if np.random.random_sample() <= prob_exceeds:
                size = size - 1

        return TopSoftMaxScorePredictionSet(pi[0][:size])

    def update(self, f_pred):
        self.f_pred = f_pred