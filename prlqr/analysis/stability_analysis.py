import numpy as np


def spectral_radius(A):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    return np.max(np.abs(np.linalg.eigvals(A)))


def check_stability(A, eps=1e-6):
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    return spectral_radius(A) < 1 - eps


# P(empirical - E(unstable) >= eps) <= exp(-2 n eps**2) <= alpha
# n >= (1/(2eps**2)) * log(1/alpha) with alpha in (0,1), eps > 0
def one_sided_hoeffding_sample_bound(alpha=0.001, eps=0.005):
    return int(np.ceil(1./(2 * eps**2) * np.log(1/alpha)))


class LinearStabilityAnalysis:

    def __init__(self, uncertainStateSpaceModel, K, settings):
        self.K = K
        self.ussm = uncertainStateSpaceModel
        self.V = self.ussm.omega_var

        self.controllability_tol = settings.synthesis_settings['controllability_tol']

        self.confidence_interval = settings.synthesis_settings['confidence_interval']

        self.As = list()
        self.Bs = list()

    # Checks the probability of being stable w.r.t. the distribution of the state space model uncertainty
    # P(empirical - E(unstable) >= eps) <= exp(-2 n eps**2) <= alpha
    def p_stability(self, alpha=0.001, eps=0.01):

        sample_list = list()

        samples = one_sided_hoeffding_sample_bound(alpha=alpha, eps=eps)
        print('Check stability based on {} samples'.format(samples))

        if len(self.As) < samples:
            As, Bs = self.ussm.sample(n=samples,
                                      c=self.confidence_interval,
                                      controllability_tol=self.controllability_tol)
            self.As, self.Bs = As, Bs

        for A, B in zip(self.As, self.Bs):

            A_cl = A + B @ self.K
            stable = check_stability(A_cl)
            sample_list.append(stable)

        return np.mean(sample_list)


if __name__ == "__main__":

    from prlqr.uncertain_state_space_model import UncertainStateSpaceModel
    from prlqr.systems.linear_system import DoubleIntegrator
    from prlqr.systems.dynamical_system import NormalRandomControlLaw, StateFeedbackLaw
    from prlqr.matrix_normal_distribution import MN

    noise = 0.1

    controller = NormalRandomControlLaw(variance=.01)

    system = DoubleIntegrator(controller, {'process_noise': noise})
    A = system.A
    B = system.B

    M = np.hstack((A, B))
    U = np.eye(2) * 0.1
    V = np.eye(3) * 0.1

    M = MN(M, U, V)

    Q = np.eye(2)
    R = np.eye(1)

    ussm = UncertainStateSpaceModel(M, (2, 1), omega_var=np.array([noise, noise]))

    controller = StateFeedbackLaw(K=system.optimal_controller(Q, R))
    n = 10

    system = DoubleIntegrator(controller, {'process_noise': noise})
    from collections import namedtuple

    TestSettings = namedtuple('TestSettings', ['synthesis_settings'])
    settings = TestSettings(synthesis_settings={'confidence_interval': 0.95})

    a = LinearStabilityAnalysis(ussm, controller.K, settings)
    v = a.p_stability()
    print(v)
