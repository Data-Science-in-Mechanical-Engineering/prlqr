import numpy as np
from numpy.linalg import eigvals
import logging

from prlqr.matrix_normal_distribution import MN


def controllable(A, B, tol=None):
    n = np.shape(A)[0]
    C = B
    for i in range(1, n):
        C = np.hstack((C, (A ** i) @ B))

    return np.linalg.matrix_rank(C, tol=tol) == n


class UncertainStateSpaceModel:

    eps = 0.

    def __init__(self, M, dim, omega_var=None):

        # This is basically for documentation purposes...
        assert isinstance(M, MN)

        self.M = M

        self.dim = dim
        n_states = self.dim[0]
        n_inputs = self.dim[1]

        if omega_var is not None:
            self.omega_var = np.diag(omega_var) + np.identity(n_states) * UncertainStateSpaceModel.eps
        else:
            self.omega_var = np.identity(n_states) * UncertainStateSpaceModel.eps

    def sample(self, n, c=.95, controllability_tol=1e-3):
        n_states = self.dim[0]
        n_inputs = self.dim[1]

        A_c = list()
        B_c = list()
        i = 0
        max_iter = 10
        while len(A_c) < n and i < max_iter:
            i += 1

            samples = self.M.sample_truncated(n=n-len(A_c), c=c)
            n_states = self.dim[0]
            n_inputs = self.dim[1]

            As = map(lambda A: A[:, 0:n_states], samples)
            Bs = map(lambda B: B[:, n_states:n_states + n_inputs], samples)

            As = list(As)
            Bs = list(Bs)

            for j in range(len(As)):
                A = As[j]
                B = Bs[j]

                if controllable(A, B, tol=controllability_tol):
                    A_c.append(A)
                    B_c.append(B)

        logging.info('Sampled {0} systems in {1} iterations. Wanted {2}'.format(len(A_c), i, n))

        return A_c, B_c

    def mean(self):

        n_states = self.dim[0]
        n_inputs = self.dim[1]

        return self.M.M[:, 0:n_states], self.M.M[:, n_states:n_states+n_inputs]

    def variances(self):

        n_states = self.dim[0]
        n_inputs = self.dim[1]

        var = np.diag(self.M.E).reshape((n_states, n_states+n_inputs))

        return var[:, 0:n_states], var[:, n_states:n_states + n_inputs]

    def print(self):
        A, B = self.mean()
        A_var, B_var = self.variances()

        logging.info('-------------------------------------')
        logging.info('Uncertain state space system:')
        np.set_printoptions(precision=4)
        logging.info('A = {0},\n B={1}'.format(A, B))

        logging.info('A:')
        logging.info('\n'.join(['\t'.join(['{:4.4f} +- {:4.4f}'.format(float(A[i,j]), 2.*np.sqrt(A_var[i,j])) for j in range(A.shape[1])]) for i in range(A.shape[0])]))

        logging.info('B:')
        logging.info('\n'.join(['\t'.join(['{:4.4f} +- {:4.4f}'.format(float(B[i,j]), 2.*np.sqrt(B_var[i,j])) for j in range(B.shape[1])]) for i in range(B.shape[0])]))

        logging.info('Noise variance {}'.format(self.omega_var))
        logging.info('-------------------------------------')
    def to_dict(self):
        data = dict()

        data['dim'] = self.dim
        data['omega_var'] = self.omega_var
        data['matrix_normal'] = self.M.to_dict()

        return data

