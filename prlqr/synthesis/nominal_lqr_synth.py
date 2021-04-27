import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_discrete_are
import logging
from prlqr.matrix_normal_distribution import MN
from prlqr.synthesis.syntheziser import LQRSyntheziser, NoControllerFound


class NLQRSyntheziser(LQRSyntheziser):

    def __init__(self, uncertainStateSpaceModel, Q, R, settings):

        super().__init__(uncertainStateSpaceModel, Q, R, settings)

        self.picos_eps = 1e-9

    def synthesize(self):

        logging.info("Start computing nominal controller.")

        K = None
        try:

            A, B = self.ussm.mean()

            P = np.array(np.array(solve_discrete_are(A, B, self.Q, self.R)))
            K = - np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)

        except Exception as e:
            logging.info(e)
            logging.info('Failed computing nominal controller.')

        logging.info('Successful computing nominal controller.')

        return K


if __name__ == "__main__":

    np.random.seed(1)
    A = np.array([
        [1, 0.2],
        [0., 1.]
    ])

    B = np.array([[0],
                  [.7]])

    M = np.hstack((A,B))

    std = 0.1

    U = np.random.rand(2,2) * std
    U = U @ U.T

    V = np.random.rand(3,3) * std
    V = V @ V.T

    M = MN(M, U, V)

    Q = np.eye(2)
    R = np.eye(1)

    from prlqr.uncertain_state_space_model import UncertainStateSpaceModel
    ussm = UncertainStateSpaceModel(M, (2, 1))

    settings = {
        'lmi_settings': {
            'posterior_samples': 20,
            'confidence_interval': .95,
        },
    }

    synth = NLQRSyntheziser(ussm, Q, R, settings)
    K = synth.synthesize()

    print(K)
