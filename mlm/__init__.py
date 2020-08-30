from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import Bunch
from scipy.optimize import root
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import  KDTree


__all__ = ['MinimalLearningMachine',
           'MinimalLearningMachineClassifier']

__author__  = "Saulo Oliveira <saulo.freitas.oliveira@gmail.com>"
__status__  = "production"
__version__ = "1.2.0"
__date__    = "07 July 2020"


def prepare_data(X, y, lb=None):
    X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

    y = np.ravel(y).reshape(len(y), -1)

    if lb != None:
        y = lb.fit_transform(y)

    return X, y

def flat(x):
    return np.ravel(x).reshape(1, -1).T

def randperm(n, k):
    permutation = np.random.permutation(n)
    return permutation[:k]

def J(t):
    return lambda y, dy: norm(cdist(y, t) - dy)

def multilateration(Dy, t):

    y0 = np.mean(t)

    _cost_function = J(t)

    result = [root(method='lm', fun=lambda y: _cost_function(y, dy), x0=y0) for dy in Dy]

    return np.asarray([_.x for _ in result])

def predict_class(lb, t, Dy):

    # class as arrays
    C = lb.transform(lb.classes_)

    # projection on t

    dyhat = cdist(C, t)

    costs = cdist(dyhat, Dy)

    result = C[np.argmin(costs, axis=0)]

    return lb.inverse_transform(result)

class MinimalLearningMachine(BaseEstimator, RegressorMixin):

    def __init__(self, estimator_type='regressor', l=0):
        self.M, self.t = list(), list()
        self._sparsity_scores = (0, np.inf) # sparsity and norm frobenius
        self._estimator_type = estimator_type
        self.l = l

    def fit(self, X, y):
        X, y = prepare_data(X, y)
        
        self.M, self.t = X, y

        assert (len(self.M) != 0), "No reference point was yielded by the selector"

        Dx = cdist(X, self.M)
        Dy = cdist(y, self.t)

        self.B_ = np.linalg.inv(Dx) @ Dy

        self._sparsity_scores = (1 - len(self.M) / len(X), norm(self.B_, ord='fro'))

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "B_")
        except AttributeError:
            raise RuntimeError("You must train the model prior predicting data!")

        dyhat = cdist(X, self.M) @ self.B_

        return multilateration(dyhat, self.t)

    def sparsity(self):
        try:
            getattr(self, "B_")
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")

        s = np.round(self._sparsity_scores, 2)

        return s[0], s[1]

class MinimalLearningMachineClassifier(MinimalLearningMachine, ClassifierMixin):

    def __init__(self):
        MinimalLearningMachine.__init__(self, estimator_type='classifier')
        self.lb = LabelBinarizer()

    def fit(self, X, y=None):
        X, y = prepare_data(X, y)

        y = self.lb.fit_transform(y)
        
        self.M, self.t = X, y

        assert (len(self.M) != 0), "No reference point was yielded by the selector"

        dx = cdist(X, self.M)
        dy = cdist(y, self.t)

        self.B_ = np.linalg.pinv(dx) @ dy

        self._sparsity_scores = (1 - len(self.M) / len(X), norm(self.B_, ord='fro'))

        return self


    def predict(self, X, y=None):
        
        try:
            getattr(self, "B_")
        except AttributeError:
            raise RuntimeError("You must train the model prior predicting data!")

        Dy = cdist(X, self.M) @ self.B_

        return predict_class(self.lb, self.t, Dy)

    def score(self, X, y, sample_weight=None):
        return ClassifierMixin.score(self, X, y, sample_weight)

def random_selector(X, y, factor=0.5):

    n = len(X)
    k = round(n * factor)

    idx = randperm(n, k)

    return X[idx], y[idx]

class RandomMinimalLearningMachine(MinimalLearningMachineClassifier):

    def __init__(self, factor=0.5):
        self.factor = factor
        MinimalLearningMachineClassifier.__init__(self)


    def fit(self, X, y):
        X, y = prepare_data(X, y)

        y = self.lb.fit_transform(y)
        
        self.M, self.t = random_selector(X, y, self.factor)

        assert (len(self.M) != 0), "No reference point was yielded by the selector"

        Dx = cdist(X, self.M)
        Dy = cdist(y, self.t)

        self.B_ = np.linalg.pinv(Dx) @ Dy

        self._sparsity_scores = (1 - len(self.M) / len(X), norm(self.B_, ord='fro'))

        return self

class RankMinimalLearningMachineClassifier(MinimalLearningMachineClassifier):

    def __init__(self, C=0):
        self.C = C
        MinimalLearningMachineClassifier.__init__(self)

    def fit(self, X, y):
        X, y = prepare_data(X, y)

        y = self.lb.fit_transform(y)

        self.M, self.t = X, y
        
        assert (len(self.M) != 0), "No reference point was yielded by the selector"

        Dx = cdist(X, self.M)
        Dy = cdist(y, self.t)

        I = np.eye(len(X))

        self.B_ = np.linalg.inv(Dx.T @ Dx + self.C * I) @ Dx.T @ Dy

        self._sparsity_scores = (1 - len(self.M) / len(X), norm(self.B_, ord='fro'))

        return self


# regressors for the state-of-the-mlm-art

def cardano(t, Dy):
    A = len(t)
    B = -3 * np.sum(t)
    C = np.sum( 3 * t ** 2 - Dy ** 2, axis=1)
    D = np.sum(Dy @ t - t ** 3, axis=1)

    return zip(A * np.ones((A, 1)), B * np.ones((A, 1)), C, D)

def root_analysis(roots, t, dy):
    real_roots = list(map(np.isreal, roots))

    cost_function = J(t)

    if np.sum(real_roots) == 3:
        # Rescue the root with the lowest cost associated
        j = [norm(cost_function(root, dy)) for root in np.real(roots)]

        idx = np.argmin(j)
        return np.real(roots[idx])

    else:
        # As True > False, then rescue the first real value
        return np.real(roots[np.argmax(real_roots)])

class CubicEquationMinimalLearningMachine(MinimalLearningMachine):

    def predict(self, X, y=None):
        try:
            getattr(self, "B_")
        except AttributeError:
            raise RuntimeError("You must train the model prior predicting data!")

        Dy = cdist(X, self.M) @ self.B_

        return [np.roots(*params) for params in cardano(self.t, Dy)]

def class_corner_selection(X, y, radius=None, k_neighbors=16, threshold=9, return_indexes=False):

    kdtree = KDTree(X)

    if radius == None:
        dist, _ = kdtree.query(X, k=2)
        radius = np.max(dist, axis=None)

    dist, ind = kdtree.query(X, k=k_neighbors + 1)

    #import pdb; pdb.set_trace()
    
    L = lambda d, i, lab: np.sum( \
        np.logical_and(flat(d <= radius), \
                        flat(cdist(y[i], lab) > 0)).astype(int))
        
    costs = [L(d, i, lab.reshape(1, -1)) for d, i, lab in zip(dist, ind, y)]
    costs = np.asarray(costs)

    

    # the query point is also into the cost list, 
    # thus we add 1 to the threshold

    proto_idx, corner_idx = costs > 1, costs > threshold + 1

    if return_indexes:
        return proto_idx, corner_idx

    
    Prototypes = X[proto_idx], y[proto_idx]
    Corners = X[corner_idx], y[corner_idx]

    return Prototypes, Corners


def ks2_sample_p(X, y, k=None, t=None, l=1):
    from scipy.stats import zscore, ks_2samp
    from sklearn.neighbors import KDTree

    kdtree = KDTree(X)

    if k == None or k < 2:
        k = np.log10(5 * len(X))
    
    Dx, idx = kdtree.query(X, k=k + 1)

    # drop itself since we recover it from each query above
    Dx = Dx[:, 1:]
    Dy = cdist(y, y[idx])

    Dx = zscore(Dx, axis=1, nan_policy='omit')
    Dy = zscore(Dy, axis=1, nan_policy='omit')

    pvals = np.asarray([ks_2samp(dx, dy) for dx, dy in zip(Dx, Dy)])

    diag_values = pvals + l * (pvals <= t).astype(int)

    P = np.eye(len(X))
    np.fill_diagonal(P, diag_values)

    return P

def random_p(X, y):
    P = np.eye(len(X))
    diag_values = np.random.normal(size=(1, len(X)))
    np.fill_diagonal(P, diag_values)
    return P

class LightWeightedMinimalLearningMachineClassifier(MinimalLearningMachineClassifier):

    def __init__(self, P=random_p):
        MinimalLearningMachineClassifier.__init__(self)
        self.P = random_p if P == None else P

    def fit(self, X, y=None):
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        y = self.lb.fit_transform(y)
        
        self.M, self.t = X, y

        assert (len(self.M) != 0), "No reference point was yielded by the selector"

        
        Dx = cdist(X, self.M)
        Dy = cdist(y, self.t)

        P = self.P(X, y) 

        self.B_ = np.linalg.inv(Dx.T @ Dx + P.T @ P) @ Dx.T @ Dy

        return self

    
class ClassCornerLightWeightedMinimalLearningMachineClassifier(LightWeightedMinimalLearningMachineClassifier):

    def __init__(self, selector=None):
        LightWeightedMinimalLearningMachineClassifier.__init__(self)

    def fit(self, X, y=None):
    
        candidates, corners_idx = class_corner_selection(X, self.lb.fit_transform(y), return_indexes=True)
        
        def _internal_p(_X, y):
            if sum(candidates) == 0:
                return np.eye(len(X))

            kdtree = KDTree(X[candidates])

            dist, _ = kdtree.query(_X, k=1)

            P = np.eye(len(_X))
            diag_values = np.max(dist) - np.asarray(dist)# ** 2
            
            np.fill_diagonal(P, diag_values)
            return P
            
        self.P = _internal_p


        LightWeightedMinimalLearningMachineClassifier.fit(self, X, y)

        self._sparsity_scores = (1 - len(self.M) / len(X), norm(self.B_, ord='fro'))
