import numpy as np
from numpy.linalg import inv, det, eigvals, pinv, cholesky
from scipy.stats import multivariate_normal, matrix_normal, chi2
import seaborn as sns
import pandas as pd

eps = 1e-12
eps2 = 1e-6


def is_pd(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def project_onto_pd_cone(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < eps] = eps

    return eigvec @ np.diag(eigval) @ eigvec.T


# TODO: Is this a good idea?
def numerically_ok_covariance(E):

    assert E.shape[0] == E.shape[1]

    if is_pd(E):
        return E

    E = E + np.eye(E.shape[0]) * eps

    if is_pd(E):
        return E

    E = E + np.eye(E.shape[0]) * eps2

    if is_pd(E):
        return E

    return project_onto_pd_cone(E)


def nkp(A, Bshape):
    """Nearest Kronecker product to a matrix.

    Adapted from https://gist.github.com/mattjj/854ea42eaf7c6b637ca84d8ca0c8310e

    Given a matrix A and a shape, solves the problem
    min || A - kron(B, C) ||_{Fro}^2
    where the minimization is over B with (the specified shape) and C.
    The size of the SVD computed in this implementation is the size of the input
    argument A, and so to compare to nkp_sum if the output is two N x N matrices
    the complexity scales like O((N^2)^3) = O(N^6).
    Args:
    A: m x n p.s.d. matrix
    Bshape: pair of ints (a, b) where a divides m and b divides n
    Returns:
    Approximating factors (B, C) bot p.s.d. matrices
    """

    A = np.linalg.cholesky(A)

    blocks = map(lambda blockcol:
                 np.split(blockcol, Bshape[0], 0),
                 np.split(A,        Bshape[1], 1))

    Atilde = np.vstack([block.ravel()
                        for blockcol in blocks
                        for block in blockcol])

    U, s, V = np.linalg.svd(Atilde)

    Cshape = A.shape[0] // Bshape[0], A.shape[1] // Bshape[1]

    idx = np.argmax(s)
    B = np.sqrt(s[idx]) * U[:, idx].reshape(Bshape).T
    C = np.sqrt(s[idx]) * V[idx, :].reshape(Cshape)

    return B @ B.T, C @ C.T


class MN(object):

    eps = 1e-12

    def __init__(self, M, U, V):

        assert(M.ndim == 2)
        assert(U.ndim == 2)
        assert(V.ndim == 2)

        n, p = M.shape

        assert(U.shape == (n, n))
        assert(V.shape == (p, p))

        self.M = M
        self.U = U
        self.V = V

        self.m = M.flatten().reshape(-1, 1)
        self.E = np.kron(U, V)

        self.E = numerically_ok_covariance(self.E)
        self.E_inv = inv(self.E)

        self.dim = (n, p)

    @staticmethod
    def from_MND(m, E, dim):
        M = m.reshape(dim)

        E = numerically_ok_covariance(E)
        U, V = nkp(E, (dim[0], dim[0]))

        return MN(M, U, V)


    def add(self, A):
        assert (isinstance(A, MN))
        assert (A.dim == self.dim)

        return self.from_MND(self.m + A.m, self.E + A.E, self.dim)


    def multiply(self, T):
        assert (T.shape[0] == self.dim[1])

        return MN(self.M @ T, self.U, T.T @ self.V @ T)

    def add_mean(self, M):

        assert (M.shape == self.dim)

        return MN(self.M + M, self.U, self.V)

    def pdf(self, X):

        M = self.M
        U = self.U
        V = self.V

        n, p = M.shape

        nom = np.exp(-0.5 * np.trace(inv(V) @ (X - M).T @ inv(U) @ (X - M)))
        denom = (2 * np.pi) ** (n*p / 2) * det(V) ** (n/2) * det(U) ** (p/2)

        p = nom / denom

        return p

    def sample(self, n=1):

        return matrix_normal.rvs(mean=self.M, rowcov=self.U, colcov=self.V, size=n)


    def pdf(self, x):

        x = x.reshape(-1)
        return multivariate_normal.pdf(x, self.m.reshape(-1), self.E)

    # def sample(self, n=1):
    #     x = multivariate_normal.rvs(self.m.reshape(-1), self.E, size=n)
    #     return x.reshape((n, *self.dim))

    def sample_vec(self, n=1):

        return multivariate_normal.rvs(self.m.reshape(-1), self.E, size=n)


    def conf_elipsoid(self, c):
        l = eigvals(self.E)
        n = self.m.shape[0]

        rv = chi2(n)

        ppf = rv.ppf(c)

        axis_length = 2 * np.sqrt(ppf * l)

    def get_ellipse_bound(self, c=0.95):

        n = self.m.shape[0]
        rv = chi2(n)
        bound = rv.ppf(c)

        return bound

    def is_inside_conf(self, x, c=0.95, bound=None):

        assert c is None or bound is None

        if bound is None:
            bound = self.get_ellipse_bound(c)

        return (x - self.m).T @ self.E_inv @ (x - self.m) <= bound

    def sample_truncated(self, n, c):

        accepted_samples = list()
        i = 0

        while len(accepted_samples) < n:

            expected_draws = int(((n - len(accepted_samples)) * (1/c)))
            samples = self.sample_vec(min(max(10, expected_draws), int(1e6)))

            bound = self.get_ellipse_bound(c)

            for sample in samples:
                if len(accepted_samples) < n and self.is_inside_conf(sample.reshape(-1, 1), c=None, bound=bound):
                    accepted_samples.append(sample.reshape(self.dim))
                    i = i + 1

        return accepted_samples

    def plot(self, n_samples=100, visualize_stable=False):

        samples = self.sample_truncated(n_samples, 0.9)

        sample_frame = pd.DataFrame(np.array(samples).reshape(n_samples, -1))
        stable = list()
        for sample in samples:
            if visualize_stable:
                stable.append(np.all(np.abs(np.linalg.eigvals(sample)) < 1))
            else:
                stable.append(True)

        sample_frame['stable'] = stable
        figure = sns.pairplot(sample_frame, hue='stable')

        return figure

    def to_dict(self):
        data = dict()

        data['M'] = self.M.copy()
        data['U'] = self.U.copy()
        data['V'] = self.V.copy()

        return data



if __name__ == "__main__":

    np.random.seed(1)
    A = np.array([
        [1, 0.2],
        [0., 1.]
    ])

    std = 1.

    U = np.random.rand(2,2)
    U = U @ U.T

    U = np.eye(2) * std

    V = np.random.rand(2,2)
    V = V @ V.T

    V = np.eye(2) * std

    A = MN(A, U, V)

    B = np.array([[0],
                  [.7]])

    U = np.random.rand(2, 2)
    U = U @ U.T
    U = np.eye(2) * std

    V = np.random.rand(1, 2)
    V = V @ V.T
    V = np.eye(1) * std


    B = MN(B, U, V)

    from scipy.linalg import solve_discrete_are

    S_ = np.eye(2)
    R_ = np.eye(1) * 0.001

    P = np.array(np.array(solve_discrete_are(A.M, B.M, S_, R_)))
    K_nom = - np.linalg.inv(R_ + B.M.T @ P @ B.M) @ (B.M.T @ P @ A.M)

    A_cl = A.add(B.multiply(K_nom))

    n_samples = 100

    import matplotlib.pyplot as plt

    fig = A_cl.plot(1000)
    plt.show()

    A.plot(1000)
    plt.show()