from calibrationScorers import calibrationScorer, predictionSet
import numpy as np


class ResidualCalibrationPredictionSet(predictionSet.PredictionSet):
    def __init__(self, iterval_center, interval_width):
        self.interval_center = iterval_center
        self.interval_width = interval_width

    def cover(self, y):
        return self.interval_center - self.interval_width <= y < self.interval_center + self.interval_width


class ResidualCalibrationScorer(calibrationScorer.CalibrationScorer):
    def __init__(self):
        self.f_pred = None

    def calc_score(self, x, y):
        return np.abs(y - self.f_pred(x))

    def get_prediction_set(self, x, calibration_threshold):
        interval_center = self.f_pred(x)
        interval_with = calibration_threshold
        new_prediction_set = ResidualCalibrationPredictionSet(interval_center, interval_with)
        return new_prediction_set

    def update(self, f_pred):
        self.f_pred = f_pred
