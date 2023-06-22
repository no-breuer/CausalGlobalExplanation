import numpy as np
import warnings
from sage import utils
from statsmodels.stats.correlation_tools import cov_nearest


class Imputer:
    '''Imputer base class.'''

    def __init__(self, model):
        self.model = utils.model_conversion(model)

    def __call__(self, x, S):
        raise NotImplementedError


class DefaultImputer(Imputer):
    '''Replace features with default values.'''

    def __init__(self, model, values):
        super().__init__(model)
        if values.ndim == 1:
            values = values[np.newaxis]
        elif values[0] != 1:
            raise ValueError('values shape must be (dim,) or (1, dim)')
        self.values = values
        self.values_repeat = values
        self.num_groups = values.shape[1]

    def __call__(self, x, S):
        # Prepare x.
        if len(x) != len(self.values_repeat):
            self.values_repeat = self.values.repeat(len(x), 0)

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.values_repeat[~S]

        # Make predictions.
        return self.model(x_)


class MarginalImputer(Imputer):
    '''Marginalizing out removed features with their marginal distribution.'''

    def __init__(self, model, data):
        super().__init__(model)
        self.data = data
        self.data_repeat = data
        self.samples = len(data)
        print(self.samples)
        self.num_groups = data.shape[1]

        if len(data) > 1024:
            warnings.warn('using {} background samples may lead to slow '
                          'runtime, consider using <= 1024'.format(
                len(data)), RuntimeWarning)

    def __call__(self, x, S):
        # Prepare x and S.
        n = len(x)
        x = x.repeat(self.samples, 0)
        S = S.repeat(self.samples, 0)

        # Prepare samples.
        if len(self.data_repeat) != self.samples * n:
            self.data_repeat = np.tile(self.data, (n, 1))

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.data_repeat[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


class CausalImputer(Imputer):
    """Sample features based on causal relationship"""

    def __init__(self, model, data, ordering=None, confounding=None):
        super().__init__(model)
        self.data = data
        self.num_samples = len(data)
        self.num_groups = data.shape[1]
        self.mu = data.mean(0)
        self.cov = np.cov(data.T)
        self.ordering = ordering
        self.confounding = confounding

        ## check if covariance is positive definite
        if np.any(np.linalg.eigvals(self.cov) > 0):
            self.cov = cov_nearest(self.cov, method="nearest")

        # check if ordering and confounding are of same size
        if ordering and (len(ordering) != len(confounding)):
            raise Exception("Ordering and confounding must be of same size")

        # all features must be mentioned in causal ordering
        if ordering:
            flat_ordering = [item for sublist in ordering for item in sublist]
            if not np.array_equal(np.arange(len(data[0])), np.sort(np.array(flat_ordering))):
                raise Exception("Wrong or uncomplete ordering")

    def __call__(self, x, S):

        # x: data points 512 batch size
        # inds: new added features to S
        # S: added features
        samples = []

        for i in range(len(x)):
            temp = np.zeros((self.num_samples, len(x[i])))

            x_ = np.tile(x[i], (self.num_samples, 1))
            S_ = np.tile(S[i], (self.num_samples, 1))

            # get indices of coalition and set them fixed. (do(X_S = x_S))
            coalition = np.where(S[i] == True)[0]
            temp[:, coalition] = x_[:, coalition]
            features = np.arange(len(x[0]))

            for j in range(len(self.ordering)):
                # determine which features need to be sampled on which to condition
                dependent = np.array(list(set(features) - set(coalition)))
                to_sample = np.array(list(set(self.ordering[j]) & set(dependent)))

                #print('coa, dep, samp:')
                #print(coalition)
                #print(dependent)
                #print(to_sample)

                if len(to_sample) != 0:
                    to_cond = np.asarray([item for sublist in self.ordering[0:j] for item in sublist])

                    if not self.confounding[j]:
                        to_cond = np.array(list(set(to_cond).union((set(self.ordering[j]) & set(coalition)))))

                    if len(to_cond) == 0:
                        new_samples = np.random.multivariate_normal(mean=self.mu[to_sample],
                                                                    cov=self.cov[np.ix_(to_sample, to_sample)],
                                                                    size=self.num_samples)

                    else:
                        # compute sigma covariance matrix for cond. sampling
                        ss_sigma = self.cov[np.ix_(to_sample, to_sample)]
                        sc_sigma = self.cov[np.ix_(to_sample, to_cond)]
                        cc_sigma = self.cov[np.ix_(to_cond, to_cond)]
                        cc_sigma_inv = np.linalg.inv(cc_sigma)
                        t = np.matmul(sc_sigma, cc_sigma_inv)
                        sampling_sigma = ss_sigma - np.matmul(t, sc_sigma.T)

                        # get mu and combine
                        mu_s = self.mu[to_sample]
                        mu_c = self.mu[to_cond]
                        sampling_mu = mu_s + np.matmul(t, (x[i][to_cond] - mu_c).T)

                        new_samples = np.random.multivariate_normal(mean=sampling_mu, cov=sampling_sigma,
                                                                    size=self.num_samples)
                        # print(new_samples)
                    temp[:, to_sample] = new_samples
            samples.append(temp)
        samples = np.asarray(samples)

        # reshape to fit model specs
        x_star = samples.reshape(-1, samples.shape[-1])

        # make prediction with causal dataset
        pred = self.model(x_star)
        pred = pred.reshape(-1, self.num_samples, *pred.shape[1:])
        return np.mean(pred, axis=1)
