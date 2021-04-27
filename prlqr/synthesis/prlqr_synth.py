import numpy as np
import picos as pic
from numpy.linalg import inv, cholesky
from scipy.linalg import pinv2
import logging

from prlqr.matrix_normal_distribution import MN
from prlqr.synthesis.syntheziser import LQRSyntheziser, NoControllerFound, NumericalProblem


#     'synthesis_settings': {
#         # Sets the confidence interval on the posterior system distribution which we consider for controller synthesis
#         # This applies to the prob. robust and the robust setting
#         'confidence_interval': .95,
#         # For the probabilistic robust synthesis
#         'stability_prob': .95,  # Probability of a system in the model posterior to be stable
#         'synthesis_prob': .95,  # in this fraction of cases (synthesis runs)
#         'a_posteriori_eps': .01,  # Epsilon the a posteriori stability analysis deviates from the (model) truth
#         'a_posteriori_prob': 0.999  # in this fraction of cases
#     },

def min_sample(eps, beta, n):

    N = (2/eps) * np.log((1/beta)) + n
    return np.ceil(N)

class PRLQRSyntheziser(LQRSyntheziser):

    def __init__(self, uncertainStateSpaceModel, Q, R, settings):

        super().__init__(uncertainStateSpaceModel, Q, R, settings)

        self.stability_prob = settings['stability_prob']
        self.synthesis_prob = settings['synthesis_prob']

        self.controllability_tol = settings['controllability_tol']

        self.min_iter = 5
        self.max_iter = settings['max_iter']

        # Number of optimization variables
        n_opt = self.ussm.dim[0] * self.ussm.dim[1] + 2 * (self.ussm.dim[0]) ** 2

        self.n_scenarios = min_sample(eps=1-self.stability_prob, beta=1-self.synthesis_prob, n=n_opt)
        self.n_scenarios = max(self.n_scenarios, 500)

        self.picos_eps = 1e-5
        self.picos_c = 1e5

        self.abs_tol = None
        self.rel_tol = None

        # Trying to solve numerical problems...
        self.cost_scale = 1.
        self.objective_scale = 1.

    def synthesize(self, retries=5):

        logging.info("Start computing probabilistically robust controller. Retries left {0}".format(retries))

        if retries <= 0:
            return None

        systems = self.sample(n=self.n_scenarios, conf=self.confidence_interval)

        try:
            logging.info("Start finding common Lyapunov solution.")

            K, Y, L, cost = self.solve_initialization(systems)

        except NumericalProblem as e:
            logging.info(e)
            logging.info("Numerical problem with confidence of {0:.2f} retry.".format(self.confidence_interval))
            return self.synthesize(retries=retries-1)

        except NoControllerFound as e:
            logging.info(e)
            logging.info("Failed with confidence of {0:.2f}.".format(self.confidence_interval))
            return None

        logging.info("Successful with confidence of {0:.2f}.".format(self.confidence_interval))

        X = [pinv2(Y) for _ in range(len(systems))]

        converged = False
        i = 0
        max_iter = self.max_iter
        while not converged and i < max_iter:
            i += + 1
            K_old = K
            cost_old = cost
            try:
                # logging.info("Iteration {0} of majorize-minimize step.".format(i))
                K, X, cost = self.solve_improvement(systems, X)

            except NumericalProblem as error:
                logging.info(error)
                if i < self.min_iter:
                    logging.info("Numerical problem in improvement. Retry.")
                    return self.synthesize(retries=retries-1)
                else:
                    logging.info("Numerical problem in improvement. But we have an improved controller. Return")
                    return K_old
            except NoControllerFound as error:
                logging.info(error)
                logging.info("For some reason the improvement failed. Return no controller.")
                return None

            converged = np.allclose(cost, cost_old, atol=1e-5)
            logging.info(
                'Improving initial controller i={0}, convergence = {1}, cost improvement = {2}.'
                    .format(i,
                            converged,
                            cost_old - cost
                            )
            )
            # logging.info(K)
        return K

    def sample(self,n, conf=2):
        """"Samples n stabilizable systems from the uncertain state space model"""

        systems = list()
        i = 0
        max_iter = 10
        while len(systems) < n and i < max_iter:
            i += 1

            As, Bs = self.ussm.sample(n=n-len(systems), c=conf, controllability_tol=self.controllability_tol)

            for j in range(len(As)):
                A = As[j]
                B = Bs[j]

                #if controllable(A, B):
                systems.append({'A': A, 'B': B})

            logging.info("Sampled {0} systems for synthesis. Wanted {1}, i = {2}".format(len(systems), n, i))

        return systems

    def solve_initialization(self, systems):

        # Solve
        K, Y, L, status, cost = self.solve_common_lyapunov_relaxation(systems, self.cost_scale, self.objective_scale)

        if status != 'optimal':
            logging.info('Status of the solution: {}'.format(status))
            logging.info('No optimal solution found for initialization.')
            if status == 'unknown':
                raise NumericalProblem
            raise NoControllerFound

        return K, Y, L, cost

    def solve_common_lyapunov_relaxation(self, systems, cost_scale=1., objective_scale=1.):

        dim = self.ussm.dim
        n_states = dim[0]
        n_inputs = dim[1]

        Q = self.Q * cost_scale
        R = self.R * cost_scale

        Q12 = np.sqrt(Q)
        Rinv = np.linalg.inv(R)

        # Noise
        N = self.ussm.omega_var
        # Normalize
        # N = N / np.linalg.norm(N, ord=2, keepdims=True)

        G = cholesky(N)
        zeros_A = np.zeros((n_states, n_states))
        zeros_B = np.zeros((n_states, n_inputs))

        ones = np.eye(n_states)

        F = pic.Problem()

        AA = [pic.Constant("A[{}]".format(i), value=systems[i]['A']) for i in range(len(systems))]
        BB = [pic.Constant("B[{}]".format(i), value=systems[i]['B']) for i in range(len(systems))]


        Q = pic.Constant("Q", Q)
        Q12 = pic.Constant("Q12", Q12)
        R = pic.Constant("R", R)
        Rinv = pic.Constant("Rinv", Rinv)
        G = pic.Constant("G", G)

        Y = pic.SymmetricVariable('Y', shape=(n_states, n_states))

        Z = pic.SymmetricVariable('Z', shape=(n_states, n_states))
        L = pic.RealVariable('L', shape=(n_inputs, n_states))

        F.set_objective('min', pic.trace(Z))

        # Constraint for numerics, we need to invert this
        #F.add_constraint(pic.SpectralNorm(Y) < 20)

        F.add_constraint(((Z & G) //
                          (G.T & Y)) >> self.picos_eps)
        #[F.add_constraint(((Z & G) //
         #                  (G.T & Y)) << self.picos_c) for Z in ZZ]

        for i in range(len(systems)):

            expr = ((Y & Y * AA[i].T + L.T * BB[i].T & Y * Q12 & L.T) //
                    (AA[i] * Y + BB[i] * L & Y & zeros_A & zeros_B) //
                    (Q12 * Y & zeros_A & ones & zeros_B) //
                    (L & zeros_B.T & zeros_B.T & Rinv))

            F.add_constraint(expr >> self.picos_eps * np.eye(expr.shape[0]))

            # F.add_constraint(expr << self.picos_c * np.eye(expr.shape[0]))

        F.add_constraint(Y >> self.picos_eps * np.eye(Y.shape[0]))
        F.add_constraint(Y << self.picos_c * np.eye(Y.shape[0]))


        F.options["abs_*_tol"] = self.abs_tol
        F.options["rel_*_tol"] = self.rel_tol
        F.options['mosek_params'] = {'MSK_IPAR_NUM_THREADS': 1}

        F.solve(verbosity=self.verbosity, primals=None)

        if F.status != 'optimal':
            return None, None, None, F.status, np.NaN

        L = np.atleast_2d(np.array(L.value))
        Y = np.atleast_2d(np.array(Y.value))

        K = L @ inv(Y)

        return K, Y, L, F.status, F.value

    def solve_improvement(self, systems, X_dash):

        # Solve
        K, X, status, cost = self.improve_upper_bound(systems,
                                                X_dash=X_dash,
                                                cost_scale=self.cost_scale,
                                                objective_scale=self.objective_scale)

        if status != 'optimal':
            logging.info('Status of the solution: {}'.format(status))
            logging.info('No optimal solution found for improvement.')
            if status == 'unknown':
                raise NumericalProblem
            raise NoControllerFound

        return K, X, cost

    def improve_upper_bound(self, systems, X_dash, cost_scale=1., objective_scale=1.):

        dim = self.ussm.dim
        n_states = dim[0]
        n_inputs = dim[1]

        # N = np.identity(n_states)
        N = self.ussm.omega_var

        # Normalize
        # N = N / np.linalg.norm(N, ord=2, keepdims=True)

        zeros_A = np.zeros((n_states, n_states))
        zeros_B = np.zeros((n_states, n_inputs))

        F = pic.Problem()

        AA = [pic.Constant("A[{}]".format(i), value=systems[i]['A']) for i in range(len(systems))]
        BB = [pic.Constant("B[{}]".format(i), value=systems[i]['B']) for i in range(len(systems))]

        Q = self.Q * cost_scale
        R = self.R * cost_scale

        Q = pic.Constant("Q", Q)
        Rinv = np.linalg.inv(R)
        Rinv = pic.Constant("Rinv", Rinv)

        XXinv = [pic.Constant("Xinv[{}]".format(i), value=pinv2(X_dash[i])) for i in range(len(systems))]

        XX = [pic.SymmetricVariable('X[{}]'.format(i), shape=(n_states, n_states)) for i in range(len(systems))]
        K = pic.RealVariable('K', shape=(n_inputs, n_states))

        scale = objective_scale / len(systems)
        cost = pic.sum([scale * pic.trace(XX[i] * N) for i in range(len(systems))])
        F.set_objective('min', cost)

        #  [F.add_constraint(pic.SpectralNorm(XX[i]) < 500) for i in range(len(systems))]

        for i in range(len(systems)):

            expr = ((XX[i] - Q & (AA[i] + BB[i] * K).T & K.T) //
                    ((AA[i] + BB[i] * K) & XXinv[i] - XXinv[i] * (XX[i] - X_dash[i]) * XXinv[i] & zeros_B) //
                    (K & zeros_B.T & Rinv))

            F.add_constraint(expr >> self.picos_eps * np.eye(expr.shape[0]))

            F.add_constraint(XX[i] << self.picos_c * np.eye(n_states))
            F.add_constraint(XX[i] >> 1e-4 * np.eye(n_states))

            # F.add_constraint(expr << self.picos_c * np.eye(expr.shape[0]))



        [F.add_constraint(X >> self.picos_eps) for X in XX]
        #[F.add_constraint(X << self.picos_c) for X in XX]

        # For numerical issues do not let this get to small, we need to invert it.
        #[F.add_constraint(pic.trace(X) > 1e-6) for X in XX]
        #[F.add_constraint(pic.trace(X) < 1e6) for X in XX]

        F.options["abs_*_tol"] = self.abs_tol
        F.options["rel_*_tol"] = self.rel_tol

        F.options['mosek_params'] = {'MSK_IPAR_NUM_THREADS': 1}

        F.solve(verbosity=self.verbosity, primals=None)

        if F.status != 'optimal':
            return None, None, F.status, np.NaN

        K = np.atleast_2d(np.array(K.value))
        X = [np.atleast_2d(np.array(X.value)) for X in XX]

        return K, X, F.status, F.value


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
    ussm = UncertainStateSpaceModel(M, (2,1))

    settings = {
        'lmi_settings': {
            'posterior_samples': 20,
            'confidence_interval': .95,
        },
    }

    synth = PRLQRSyntheziser(ussm, Q, R, settings)

    K = synth.synthesize()

    print(K)
