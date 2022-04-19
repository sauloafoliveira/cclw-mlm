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
__date__    = "17 April 2022"


def prepare_data(X, y, lb=None):
    X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
    return X, y

def randperm(n, k):
    permutation = np.random.permutation(n)
    return permutation[:k]

def J(t):
    return lambda y, dy: norm(cdist(np.atleast_2d(y), np.atleast_2d(t)) - np.atleast_2d(dy))

def multilateration(Dy, t):

    y0 = np.mean(t)

    _cost_function = J(t)

    result = [root(method='lm', \
        fun=lambda y: _cost_function(y, dy), x0=y0) \
            for dy in Dy]

    return np.asarray([_.x for _ in result])


'''
SOUZA JUNIOR, Amauri Holanda et al. 
Minimal learning machine: a novel supervised distance-based approach for regression and classification. 
Neurocomputing, v. 164, p. 34-44, 2015.
https://doi.org/10.1016/j.neucom.2014.11.073
'''
class MinimalLearningMachine(BaseEstimator, RegressorMixin):

    def __init__(self, estimator_type='regressor'):
        self.M, self.t = None, None  # reference points
        self._sparsity_scores = (0, np.inf) # sparsity and norm frobenius
        self._estimator_type = estimator_type
        
    def _set_reference_points(self, X, y):
        
        self.M = X
        self.t = y if len(np.shape(y)) == 2 else np.reshape(y, (len(y), -1))

    def _compute_dx_dy(self, X, y):
        assert (len(self.M) != 0), "No reference point was yielded by the selector"

        yy = y if len(np.shape(y)) == 2 else np.reshape(y, (len(y), -1))
        
        self.Dx = cdist(X, self.M)
        self.Dy = cdist(yy, self.t)
        return self     

    def _estimate_b(self):
        self.B_ = np.linalg.inv(self.Dx) @ self.Dy
        self.Dx, self.Dy = None, None
        return self

    def fit(self, X, y):
        X, y = prepare_data(X, y)

        self._set_reference_points(X, y)

        self._compute_dx_dy(X, y)

        self._estimate_b()
        
        self._sparsity_scores = (1 - len(self.M) / len(X), norm(self.B_, ord='fro'))

        self.Dx, self.Dy = None, None # release memory

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

        return (s[0], s[1])



def fast_mlat_nn(lb, t, Dy):
    # classes as one_hot
    one_hot_classes = lb.transform(lb.classes_)

    dy = cdist(one_hot_classes, t)

    costs = cdist(Dy, dy)

    minimun_costs_idx = np.argmin(costs, axis=1)
    
    actual_output = lb.classes_[minimun_costs_idx]

    return actual_output    
    
'''
SOUZA JUNIOR, Amauri Holanda et al. 
Minimal learning machine: a novel supervised distance-based approach for regression and classification. 
Neurocomputing, v. 164, p. 34-44, 2015.
https://doi.org/10.1016/j.neucom.2014.11.073
'''
class MinimalLearningMachineClassifier(MinimalLearningMachine, ClassifierMixin):

    def __init__(self, estimator_type='classifier'):
        super().__init__(estimator_type=estimator_type)
        self.lb = LabelBinarizer()

    def fit(self, X, y=None):
        X, y = prepare_data(X, y)
        # y must be one-hot-encoding

        one_hot_y = self.lb.fit_transform(y)

        return super().fit(X, one_hot_y)

    def _estimate_b(self):
        self.B_ = np.linalg.pinv(self.Dx) @ self.Dy
        self.Dx, self.Dy = None, None
        return self

    def predict(self, X, y=None):
        
        X = np.atleast_2d(X)

        try:
            getattr(self, "B_")
        except AttributeError:
            raise RuntimeError("You must train the model prior predicting data!")

        
        Dy = cdist(X, self.M) @ self.B_

        return fast_mlat_nn(self.lb, self.t, Dy)

    def score(self, X, y, sample_weight=None):
        return ClassifierMixin.score(self, X, y, sample_weight)

def random_selection(X, y, factor=0.5):

    n = len(X)
    k = round(n * factor)

    idx = randperm(n, k)

    return X[idx], y[idx]

'''
SOUZA JUNIOR, Amauri Holanda et al. 
Minimal learning machine: a novel supervised distance-based approach for regression and classification. 
Neurocomputing, v. 164, p. 34-44, 2015.
https://doi.org/10.1016/j.neucom.2014.11.073
'''
class RandomMLMClassifier(MinimalLearningMachineClassifier):

    def __init__(self, factor=0.5):
        super().__init__()
        self.factor = factor


    def _set_reference_points(self, X, y):
        self.M, self.t = random_selection(X, y, self.factor)
        return self

    def _estimate_b(self):
        self.B_ = np.linalg.pinv(self.Dx) @ self.Dy
        return self

'''
ALENCAR, Alisson SC et al. 
Mlm-rank: A ranking algorithm based on the minimal learning machine. 
In: 2015 Brazilian Conference on Intelligent Systems (BRACIS). IEEE, 2015. p. 305-309.
https://doi.org/10.1109/BRACIS.2015.39
'''
class RankMLMClassifier(MinimalLearningMachineClassifier):

    def __init__(self, c=0):
        super().__init__()
        self.c = 1e-5 if c == 0 else c

    def _estimate_b(self):
        n = self.Dx.shape[0]
        cI = np.eye(n) * self.c

        inv_Dx = np.linalg.inv(self.Dx.T @ self.Dx + cI)
        self.B_ = inv_Dx @ self.Dx.T @ self.Dy
        
        return self


# regressors for the state-of-the-mlm-art
def cardano(t, Dy):
    A = len(t)
    B = -3 * np.sum(t)
    C = np.sum( 3 * t ** 2 - Dy ** 2, axis=1)
    D = np.sum(Dy @ t - t ** 3, axis=1)

    return zip(A * np.ones((A, 1)), B * np.ones((A, 1)), C, D)

def root_analysis(roots, t, dy):
    real_roots = np.isreal(roots, dtype=complex)
    cost_function = J(t)

    if np.sum(real_roots) == 3:
        # Rescue the root with the lowest cost associated
        costs = [norm(cost_function(_root, dy)) for _root in real_roots]

        idx = np.argmin(costs)

        return real_roots[idx]

    else:
        # As True > False, then rescue the first real value
        idx = np.argmax(real_roots)
        return real_roots[idx]

'''
MESQUITA, Diego PP; GOMES, Joao PP; SOUZA JUNIOR, Amauri H. 
Ensemble of efficient minimal learning machines for classification and regression. 
Neural Processing Letters, v. 46, n. 3, p. 751-766, 2017.
https://doi.org/10.1007/s11063-017-9587-5
'''
class CubicEquationMLM(MinimalLearningMachine):

    def predict(self, X, y=None):
        try:
            getattr(self, "B_")
        except AttributeError:
            raise RuntimeError("You must train the model prior predicting data!")

        Dy = cdist(X, self.M) @ self.B_

        return [root_analysis(*params) for params in cardano(self.t, Dy)]


'''
FLORENCIO, Jose AV et al. 
A new perspective for minimal learning machines: A lightweight approach. 
Neurocomputing, v. 401, p. 308-319, 2020.
https://doi.org/10.1016/j.neucom.2020.03.088
'''
class LightWeightedMLM(CubicEquationMLM):

    def __init__(self, estimator_type='regressor'):
        super().__init__(estimator_type)
        self.P = None

    def _compute_p(self, X, y): # random
        P = np.eye(len(X))
        diag_values = np.random.normal(size=(1, len(X)))
        np.fill_diagonal(P, diag_values)
        return P
    
    def _set_reference_points(self, X, y):
        super()._set_reference_points(X, y)
        self.P = self._compute_p(X, y)
    
    def _estimate_b(self):

        Dx, Dy, P = self.Dx, self.Dy, self.P
        
        inv_Dx_P = Dx.T @ Dx + P.T @ P

        self.B_ = np.linalg.inv(inv_Dx_P) @ Dx.T @ Dy
        
        self.P = None

        return self



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

'''
FLORENCIO, Jose AV et al. 
A new perspective for minimal learning machines: A lightweight approach. 
Neurocomputing, v. 401, p. 308-319, 2020.
https://doi.org/10.1016/j.neucom.2020.03.088
'''
class K2S_LWMLM(LightWeightedMLM):

    def __init__(self, k_neighbors=None, threshold=None, l=1):
        super().__init__()
        self.k = k_neighbors
        self.threshold = threshold
        self.l = l

    def _compute_p(self, X, y):
        return ks2_sample_p(X, y, self.k_neighbors, self.threshold, self.l)


def class_corner_selection(X, y, radius=None, k_neighbors=16, threshold=9, return_indexes=False, return_costs=False):

    kdtree = KDTree(X)

    if threshold >= k_neighbors:
        raise RuntimeError("You must have a threshold < k_neighbors!")

    if radius == None: # fast setup
        dist, _ = kdtree.query(X, k=2)
        radius = np.max(dist, axis=None)

    dist, ind = kdtree.query(X, k=k_neighbors + 1)
   
    L = lambda d, i, lab: np.sum( \
        np.logical_and(flat(d <= radius), \
                        flat(cdist(y[i], lab) > 0)).astype(int))
        
    costs = [L(d, i, lab.reshape(1, -1)) for d, i, lab in zip(dist, ind, y)]
    costs = np.asarray(costs)

    # the query point is also into the cost list, 
    # thus we add 1 to the threshold

    proto_idx, corner_idx = costs > 1, costs > threshold + 1

    if return_indexes:
        if return_costs:
            return proto_idx, corner_idx, costs

        return proto_idx, corner_idx

    
    Prototypes = X[proto_idx], y[proto_idx]
    Corners = X[corner_idx], y[corner_idx]

    if return_costs:
        Prototypes, Corners, costs
    return Prototypes, Corners


class LightWeightedMLMClassifier(LightWeightedMLM, MinimalLearningMachineClassifier):

    def __init__(self):
        super().__init__(estimator_type='classifier')

    def fit(self, X, y=None):
        X, y = prepare_data(X, y)
        # y must be one-hot-encoding
        return LightWeightedMLM.fit(self, X, self.lb.fit_transform(y))


    def predict(self, X, y=None):
        return MinimalLearningMachineClassifier.predict(self, X, y)


def ncd(X, Ps):
    kdtree = KDTree(Ps)
    nx, _ = kdtree.query(X, k=1)
    return np.max(nx) - nx


'''
Dealing with Heteroscedasticity in Minimal Learning Machine Framework

'''
class ClassCornerLWMLMClassifier(LightWeightedMLMClassifier):

    def __init__(self, radius=None, k_neighbors=16, threshold=9):
        super().__init__(self)

        self.radius = radius
        self.k_neighbors = k_neighbors
        self.threshold = threshold

    def _compute_p(self, X, y):
        candidates, _ = class_corner_selection(X, self.lb.fit_transform(y), return_indexes=True)
        
        P = np.eye(len(X))

        if np.any(candidates):
            Ps = X[candidates]
            np.fill_diagonal(P, ncd(X, Ps))
        return P


'''
GOMES, Joao Paulo P., SOUZA, Amauri H., CORONA, Francesco, et al. 
A cost sensitive minimal learning machine for pattern classification. 
In : International Conference on Neural Information Processing. 
Springer, Cham, 2015. p. 557-564.
https://10.1007/978-3-319-26532-2_61
'''
class wMLM(MinimalLearningMachineClassifier):

    def __init__(self):
        super().__init__()

    def _compute_w(self, X, y):
        cf = 1 / np.bincount(y) # inverse of class frequence

        w = np.zeros(y.shape)

        for i, u in enumerate(np.unique(y)):
            w += np.where(u == y, cf[i], 0)

        return np.diag(w)

    def _estimate_b(self):
        W = self._compute_w()

        inv_Dx = self.Dx.T @ W @ self.Dx
        self.B_ = inv_Dx @ W @ self.Dy

        return self