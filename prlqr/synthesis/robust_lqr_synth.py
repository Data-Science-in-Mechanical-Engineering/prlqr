import numpy as np
import picos as pic
from numpy.linalg import inv, cholesky
from scipy.stats import norm
import logging

from prlqr.matrix_normal_distribution import MN
from prlqr.synthesis.syntheziser import LQRSyntheziser, NoControllerFound


class RLQRSyntheziser(LQRSyntheziser):

    def __init__(self, uncertainStateSpaceModel, Q, R, settings):

        super().__init__(uncertainStateSpaceModel, Q, R, settings)

        eps = 1 - settings['stability_prob']
        eps_post = settings['a_posteriori_eps']
        confidence_interval = settings['confidence_interval']

        confidence_interval = max(0., confidence_interval - (eps + eps_post))

        self.confidence_interval = confidence_interval

        self.picos_eps = 1e-6
        self.abs_tol = 1e-7
        self.rel_tol = 1e-6

    def synthesize(self):

        logging.info("Start computing robust controller.")

        K, R, Q = None, None, None

        try:
            K, R, Q = self.solve_lqr_lmi_hinf(conf=self.confidence_interval)

        except Exception as e:
            logging.info(e)

            logging.info("Failed with confidence of {0:.2f}.".format(self.confidence_interval))
            return None

        logging.info("Successful with confidence of {0:.2f}.".format(self.confidence_interval))

        return K

    def solve_lqr_lmi_hinf(self, alpha=1., conf=.95):

        Q_ = self.Q
        R_ = self.R

        dim = self.ussm.dim
        n = dim[0]  # #of states
        m = dim[1]  # #of inputs
        f = n**2 + n*m
        q = n + m

        A, B = self.ussm.mean()

        delta = alpha**(-2)

        Q12 = np.sqrt(Q_)
        R12 = np.sqrt(R_)

        zeros_A = np.zeros((n, n))
        zeros_B = np.zeros((n, m))

        # Bw = pic.new_param("Bw", np.ones((n, q)))
        N = self.ussm.omega_var
        G = cholesky(N)
        Bw = np.zeros((n, q))
        Bw[0:n, 0:n] = G
        # omega_var = np.diag(self.ussm.omega_var)
        # np.fill_diagonal(Bw, omega_var)

        A_var, B_var = self.ussm.variances()

        ppf = norm.ppf(1-(1-conf)/2)

        A_u = ppf * np.sqrt(np.array(A_var))
        Cq = [np.diag(A_u[i, :]) for i in range(n)]
        Cq = np.vstack(Cq)
        Cq = np.vstack([Cq, np.zeros((n*m, n))])
        Cq = pic.Constant("C_q", Cq)

        B_u = ppf * np.sqrt(np.array(B_var))
        Dq = [np.diag(B_u[i, :]) for i in range(n)]
        Dq = np.vstack(Dq)
        Dq = np.vstack([np.zeros((n*n, m)), Dq])

        Dq = pic.Constant("D_q", Dq)

        Bp = np.hstack([np.kron(np.eye(n), np.ones((1, n))), np.kron(np.eye(n), np.ones((1, m)))])

        F = pic.Problem()
        W = pic.SymmetricVariable('W', shape=(q, q))
        Q = pic.SymmetricVariable('Q', shape=(n, n))
        R = pic.RealVariable('R', shape=(m, n))
        t = pic.RealVariable('T', shape=(f, 1))
        T = pic.diag(t)
        g = pic.RealVariable('g', shape=(1, 1))

        F.set_objective('min', g)

        Q12 = pic.Constant("Q12", Q12)
        R12 = pic.Constant("R12", R12)

        Cz = pic.Constant("Cz", ((Q12) // (zeros_B.T)))
        Dz = pic.Constant("Dz", ((zeros_B) // (R12)))
        Bw = pic.Constant("Bw", Bw)
        Bp = pic.Constant("Bp", Bp)
        A = pic.Constant("A", A)
        B = pic.Constant("B", B)


        F.add_constraint(pic.trace(W) <= g)

        expr0 = ((W & (Cz * Q + Dz * R)) //
                 ((Cz * Q + Dz * R).T & Q))

        F.add_constraint(expr0 == expr0.T)
        F.add_constraint(expr0 >> self.picos_eps)


        I_q = np.eye(q)
        zeros_qn = pic.Constant("zeros_qn", np.zeros((q, n)))
        zeros_fn = pic.Constant("zeros_fn", np.zeros((f, n)))
        zeros_fq = pic.Constant("zeros_fq", np.zeros((f, q)))
        zeros_ff = pic.Constant("zeros_ff", np.zeros((f, f)))

        expr1 = A * Q + B * R
        expr2 = Cq * Q + Dq * R
        expr3 = Bp * T

        expr4 = ((Q        & zeros_qn.T    & zeros_fn.T    & expr1.T   & expr2.T       ) //
                 (zeros_qn & I_q           & zeros_fq.T    & Bw.T      & zeros_fq.T    ) //
                 (zeros_fn & zeros_fq      & T             & expr3.T   & zeros_ff.T    ) //
                 (expr1    & Bw            & expr3         & Q         & zeros_fn.T    ) //
                 (expr2    & zeros_fq      & zeros_ff      & zeros_fn  & delta*T       ))

        F.add_constraint(expr4 >> self.picos_eps)

        F.add_constraint(expr4 == expr4.T)

        F.options["abs_*_tol"] = self.abs_tol
        F.options["rel_*_tol"] = self.rel_tol

        F.solve(verbosity=self.verbosity, primals=True)
        # logging.info(F)

        if F.status != 'optimal':
            logging.info('Status of the solution: {}'.format(F.status))
            logging.info('No optimal solution found.')

            if not F.check_current_value_feasibility():
                logging.info('No feasible solution found.')
                raise NoControllerFound

        R = np.atleast_2d(np.array(R.value))
        Q = np.atleast_2d(np.array(Q.value))

        K = R @ inv(Q)

        return K, R, Q

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

    synth = RLQRSyntheziser(ussm, Q, R, settings)
    K = synth.synthesize()

    print(K)
