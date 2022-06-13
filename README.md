# MVP: Practical Adversarial Multivalid Conformal Prediction


This repository contains code associated with the paper [Practical Adversarial Multivalid Conformal Prediction](https://arxiv.org/abs/2206.01067) by O. Bastani, V. Gupta, C. Jung, G. Noarov, R. Ramalingam, and A. Roth. 

We propose MVP (MultiValid Prediction) --- a conformal prediction method for sequential adversarial data that produces prediction sets with valid, stronger-than-marginal empirical coverage that is:
- *Threshold-calibrated:* The coverage is valid conditional on the threshold used to form the prediction set from the conformal score.
- *Group-conditional:* The coverage is valid on each of an arbitrary (e.g. intersecting) user-specified collection of subsets of the feature space.

## Contents

 - `src/` Implementation of MVP (class `MultiValidPrediction` contained in `src/MultiValidPrediction.py`), 
 along with some useful utilities (in particular, a collection of conformal scorers in `src/calibrationScorers/`).
 - `experiments/` Jupyter notebooks for the experiments in the corresponding section of the paper.
