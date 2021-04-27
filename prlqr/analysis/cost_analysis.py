import numpy as np
import picos as pic

from prlqr.analysis.stability_analysis import check_stability


# TODO: Can we apply Hoeffding here to know how often we need to sample?
class LinearQuadraticCostAnalysis:

    def __init__(self, uncertainStateSpaceModel, K, Q, R):
        self.K = K
        self.ussm = uncertainStateSpaceModel
        self.Q = Q
        self.R = R
        self.V = self.ussm.omega_var

        self.stab_eps = 1e-6


    # Steady state cost with n time steps (after reaching the steady state)
    # From: https://arxiv.org/abs/1910.07732v1 eq. (11)
    def lqr_expected_cost(self, A, B, n=1):

        A_cl = A + B @ self.K
        if not check_stability(A_cl, eps=self.stab_eps):
            return np.Inf

        Q_cl = self.Q + self.K.T @ self.R @ self.K

        d = A.shape
        X_Q = pic.SymmetricVariable('X_Q', shape=d)

        F = pic.Problem()

        F.set_objective('min', pic.trace(X_Q))

        F.add_constraint(A_cl.T * X_Q * A_cl - X_Q + Q_cl == 0)
        F.add_constraint(X_Q >> 0)

        F.solve(verbosity=0, primals=None)

        # Unstable, so expected cost is infinite
        if F.status != 'optimal':
            return np.Inf

        X_Q = np.atleast_2d(X_Q.value)

        return np.trace(n * self.V @ X_Q)

    def expected_cost(self, n=100, samples=100, c=.95):

        lqr_costs = list()

        As, Bs = self.ussm.sample(n=samples, c=c)

        for A, B in zip(As, Bs):
            mean_cost = self.lqr_expected_cost(A, B, n)
            lqr_costs.append(mean_cost)

        return lqr_costs


class EmpiricalQuadraticCostAnalysis:

    def __init__(self, system, Q, R, settings):

        self.system = system
        self.Q = Q
        self.R = R

        # TODO This works for a static reference only. Which is fine for now
        self.x0 = system.current_reference_state
        self.u0 = system.current_reference_input

    def lqr_sum(self, n=100, samples=100):
        cost_sum = list()

        for _ in range(samples):

            offset = 10
            data = self.system.create_trajectory(self.x0, n=n+offset)

            x = data['x']
            u = data['u']
            cost = list()
            for i in range(offset, u.shape[1]):

                x_i = x[:,[i]] - self.x0
                u_i = u[:, [i]] - self.u0

                if self.system.empirically_unstable(x_i + self.x0, u_i + self.u0):
                   return np.Inf, np.Inf

                c_i = x_i.T @ self.Q @ x_i + u_i.T @ self.R @ u_i
                cost.append(c_i)

            cost = np.array(cost)
            cost_sum.append(cost.sum())

        cost_sum = np.array(cost_sum)
        return cost_sum.mean(), cost_sum.std()


if __name__ == "__main__":

    from prlqr.uncertain_state_space_model import UncertainStateSpaceModel
    from prlqr.systems.linear_system import DoubleIntegrator
    from prlqr.systems.dynamical_system import NormalRandomControlLaw, StateFeedbackLaw
    from prlqr.matrix_normal_distribution import MN

    noise = 0.001

    controller = NormalRandomControlLaw(variance=.1)

    system = DoubleIntegrator(controller, {'process_noise': noise})
    A = system.A
    B = system.B

    M = np.hstack((A, B))
    U = np.eye(2) * 0.025
    V = np.eye(3) * 0.025

    M = MN(M, U, V)

    Q = np.eye(2)
    R = np.eye(1)

    ussm = UncertainStateSpaceModel(M, (2, 1), omega_var=np.array([noise, noise]))

    controller = StateFeedbackLaw(K=system.optimal_controller(Q, R))
    n = 200

    system = DoubleIntegrator(controller, {'process_noise': noise})
    emp = EmpiricalQuadraticCostAnalysis(system, Q, R, dict())
    cost_emp, cost_emp_v = emp.lqr_sum(n=n, samples=100)
    print('Emp. cost for the given system {0}+-{1}'.format(cost_emp, cost_emp_v))

    test = LinearQuadraticCostAnalysis(uncertainStateSpaceModel=ussm, K=controller.K, Q=Q, R=R)
    cost_lmi = test.lqr_expected_cost(system.A, system.B, n=n)

    print('Anl. cost for the given system {0}'.format(cost_lmi))

    cost_exp = test.expected_cost(n, samples=10)

    print('Exp. cost over uncertain system {0}+-{1}'.format(np.ma.masked_invalid(cost_exp).mean(),
                                                            np.ma.masked_invalid(cost_exp).std()))
