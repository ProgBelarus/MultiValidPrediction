import sys

sys.path.insert(0, '../')

from calibrationScorers import calibrationScorer, predictionSet

import numpy as np
from utils_raps import validate, get_logits_targets, sort_sum
from scipy.special import softmax
import conformal

import pdb


class RapsPredictionSet(predictionSet.PredictionSet):
    def __init__(self, ys):
        self.ys = ys

    def cover(self, y):
        return y in self.ys


class RapsScorer(calibrationScorer.CalibrationScorer):
    def __init__(self, model, num_classes, delta, lamda, kreg, T=1.3, randomized=True, allow_zero_sets=True):
        self.model = model
        self.delta = delta

        self.num_classes = num_classes

        self.penalties = np.zeros((1, self.num_classes))
        self.penalties[:, int(kreg):] += lamda

        self.T = T
        self.randomized = randomized
        self.allow_zero_sets = allow_zero_sets

    # Returns the softmax score and conformal score
    def calc_score(self, x, y):
        logit = self.model(x)
        logit_numpy = logit.detach().cpu().numpy()
        score = softmax(logit_numpy / self.T, axis=1)
        I, ordered, cumsum = sort_sum(score)

        E = conformal.giq(score, y, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties,
                     randomized=self.randomized, allow_zero_sets=self.allow_zero_sets)

        return score, E

    def get_prediction_set_from_softmax_score(self, soft_max_score, q):
        score = soft_max_score
        I, ordered, cumsum = sort_sum(score)

        S = conformal.gcq(score, q, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties, randomized=self.randomized,
                allow_zero_sets=self.allow_zero_sets)

        return RapsPredictionSet(S[0])

    def get_prediction_set(self, x, q):
        logit = self.model(x)
        logit_numpy = logit.detach().cpu().numpy()
        score = softmax(logit_numpy / self.T, axis=1)
        I, ordered, cumsum = sort_sum(score)

        S = conformal.gcq(score, q, I=I, ordered=ordered, cumsum=cumsum, penalties=self.penalties, randomized=self.randomized,
                allow_zero_sets=self.allow_zero_sets)

        return RapsPredictionSet(S)


