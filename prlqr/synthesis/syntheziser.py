import numpy as np

class NoControllerFound(Exception):
    """Raised when there is no common controller for the sampled systems"""
    pass

class NumericalProblem(Exception):
    pass

class LQRSyntheziser:

    def __init__(self, uncertainStateSpaceModel, Q, R, settings):
        self.ussm = uncertainStateSpaceModel

        dim = self.ussm.dim
        n_states = dim[0]
        n_inputs = dim[1]

        assert (Q.shape == (n_states, n_states))
        assert (R.shape == (n_inputs, n_inputs))


        # Normalize for numerics..
        Q_norm = np.linalg.norm(Q, ord=2, keepdims=True)
        R_norm = np.linalg.norm(R, ord=2, keepdims=True)

        # avg_norm = (Q_norm + R_norm) / 2
        avg_norm = Q_norm
        self.Q = Q / avg_norm
        self.R = R / avg_norm

        self.confidence_interval = settings['confidence_interval']

        self.verbosity = 0
