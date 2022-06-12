import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

class MultiValidPrediction:
    def __init__(self, delta=0.1, n_buckets=50, groups=[(lambda x : True)], eta=0.5, r=1000, normalize_by_counts=True):
        # coverage parameter, want to be 1 - delta covered
        self.delta = delta
        # how many buckets do you want?
        self.n_buckets = n_buckets
        # groups, given as a collection of True/False outputting functions
        self.groups = groups
        self.n_groups = len(groups)
        # eta, should be set externally based on n_groups, n_buckets, and T
        self.eta = eta
        # nuisance parameter
        self.r = r
        # do you normalize computation by bucket-group counts? 
        self.normalize_by_counts = normalize_by_counts

        # actual thresholds played
        self.thresholds = []
        # scores encountered
        self.scores = []
        # feature vectors encountered
        self.xs = []

        # for each round: 1 = miscovered, 0 = covered
        self.err_seq = []
        # vvals[i][g] = v_value on bucket i and group g so far
        self.vvals = np.zeros((self.n_buckets, self.n_groups), dtype=np.float64)
        # bg_miscoverage[i][g] = how many times was (i, g) miscovered so far?
        self.bg_miscoverage = np.zeros((self.n_buckets, self.n_groups), dtype=int)
        # bg_counts[i][g] = how many times did (i, g) come up so far?
        self.bg_counts = np.zeros((self.n_buckets, self.n_groups), dtype=int)


    def predict(self, x):
        curr_groups = [i for i in range(self.n_groups) if (self.groups[i])(x)]
        if len(curr_groups) == 0: # arbitrarily return threshold 0 for points with zero groups
          return 0

        all_c_neg = True # are all c values nonpositive?
        cmps_prev = 0.0
        cmps_curr = 0.0
        overcalibr_log_prev = 0.0
        overcalibr_log_curr = 0.0

        for i in range(self.n_buckets):
            # compute normalized bucket-group counts
            norm_fn = lambda x: np.sqrt((x+1)*(np.log2(x+2)**2))
            bg_counts_norm = 1./norm_fn(self.bg_counts[i, curr_groups])
            
            # compute sign of cvalue for bucket i
            a = self.eta * self.vvals[i, curr_groups]
            if self.normalize_by_counts:
                a *= bg_counts_norm
            mx = np.max(a)
            mn = np.min(a)

            if self.normalize_by_counts:
                overcalibr_log_curr  =  mx + logsumexp( a - mx, b=bg_counts_norm)
                undercalibr_log_curr = -mn + logsumexp(-a + mn, b=bg_counts_norm)
            else:
                overcalibr_log_curr  =  mx + logsumexp( a - mx)
                undercalibr_log_curr = -mn + logsumexp(-a + mn)
            cmps_curr = overcalibr_log_curr - undercalibr_log_curr

            if cmps_curr > 0:
                all_c_neg = False
            
            if (i != 0) and ((cmps_curr >= 0 and cmps_prev <= 0) or (cmps_curr <= 0 and cmps_prev >= 0)):
                cvalue_prev = np.exp(overcalibr_log_prev) - np.exp(undercalibr_log_prev)
                cvalue_curr = np.exp(overcalibr_log_curr) - np.exp(undercalibr_log_curr)

                Z = np.abs(cvalue_prev) + np.abs(cvalue_curr)
                prob_prev = 1 if Z == 0 else np.abs(cvalue_curr)/Z
                if np.random.random_sample() <= prob_prev:
                    return (1.0 * i) / self.n_buckets - 1.0 /(self.r * self.n_buckets)
                else:
                    return 1.0 * i / self.n_buckets

            cmps_prev = cmps_curr
            overcalibr_log_prev = overcalibr_log_curr
            undercalibr_log_prev = undercalibr_log_curr

        return (1.0 if all_c_neg else 0.0)

    def update(self, x, threshold, score):
        curr_groups = [i for i in range(self.n_groups) if (self.groups[i])(x)]
        if len(curr_groups) == 0: # don't update on points with zero groups
          return

        self.thresholds.append(threshold)
        self.scores.append(score)
        self.xs.append(x)

        bucket = min(int(threshold * self.n_buckets + 0.5/self.r), self.n_buckets - 1)
        err_ind = int(score > threshold)
      
        # update vvals
        self.vvals[bucket, curr_groups] += self.delta - err_ind # (1-err_ind) - (1-delta)
        # update stats
        self.bg_counts[bucket, curr_groups] += 1
        self.bg_miscoverage[bucket, curr_groups] += err_ind
        self.err_seq.append(err_ind)

    def coverage_stats(self, plot_thresholds=True, print_per_group_stats=True):
        if plot_thresholds:
          plt.plot(self.thresholds)
          plt.title('Thresholds')
          plt.show()

        if print_per_group_stats:
          print('Per-Group Coverage Statistics:')
          for group in range(self.n_groups):
              miscoverages = np.sum(self.bg_miscoverage[:, group])
              counts = np.sum(self.bg_counts[:, group])
              miscoverage_rate = miscoverages/counts

              spacing = int(np.ceil(np.log10(len(self.thresholds))))
              group_spacing = int(np.ceil(np.log10(self.n_groups)))
              print(  'Group ',         '{:{x}d}'.format(group, x=group_spacing), 
                    ': Count: ',        '{:{x}d}'.format(counts, x=spacing), 
                    ', Miscoverages: ', '{:{x}d}'.format(miscoverages, x=spacing), 
                    ', Rate: ',         f'{miscoverage_rate:.6f}')
        
        return self.thresholds, self.bg_miscoverage, self.bg_counts
