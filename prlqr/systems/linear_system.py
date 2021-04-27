import numpy as np
from prlqr.systems.dynamical_system import DiscreteTimeDynamicalSystem, StateFeedbackLaw, NormalRandomControlLaw
from prlqr.analysis.stability_analysis import check_stability

class LinearSystem(DiscreteTimeDynamicalSystem):

    def __init__(self, A, B, controller, settings):
        self.A = A
        self.B = B

        super().__init__(state_dimension=B.shape[0], input_dimension=B.shape[1], controller=controller, settings=settings)

    def x_next(self, u):
        noise = np.random.randn(self.state_dimension, 1) * np.sqrt(self.process_noise)
        return self.A @ self.current_state + self.B @ u + noise

    def empirically_unstable(self, x, u):
        if isinstance(self.controller, StateFeedbackLaw):
            A_cl = self.A + self.B @ self.controller.K
            return not check_stability(A_cl)

        # Check bounds that a stable controller should not pass... Overwrite this if necessary for your system
        else:
            return np.any(x > 1e3) or np.any(u > 1e3)

    def linearize(self, q):
        return self.A, self.B

    def optimal_controller(self, Q, R):
        from scipy.linalg import solve_discrete_are

        P = np.array(solve_discrete_are(self.A, self.B, Q, R))
        K_opt = - np.linalg.inv(R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

        return K_opt


class DoubleIntegrator(LinearSystem):

    def __init__(self, controller, settings):
        A = np.array([
            [1, 0.2],
            [0, 1]
        ])

        B = np.array([
            [0],
            [.7]
        ])
        super().__init__(A, B, controller, settings)


class GraphLaplacian3D(LinearSystem):

    def __init__(self, controller, settings):
        A = np.array([
            [1.01, 0.01, 0.00],
            [0.01, 1.01, 0.01],
            [0.00, 0.01, 1.01],
        ])

        B = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        super().__init__(A, B, controller, settings)

    def x_next(self, u):
        noise = np.random.randn(self.state_dimension, 1) * np.sqrt(self.process_noise)

        return self.A @ self.current_state + self.B @ u + noise


class GraphLaplacian3DNonLin(GraphLaplacian3D):

    def __init__(self, controller, settings):
        super().__init__(controller, settings)

    def x_next(self, u):
        noise = np.random.randn(self.state_dimension, 1) * np.sqrt(self.process_noise)

        state = self.current_state
        non_lin = (0.3 * np.tril(np.ones((3, 3))) @ state) ** 3
        return np.clip(self.A @ self.current_state + self.B @ u + non_lin + noise, -1e3, 1e3)

    def empirically_unstable(self, x, u):
        if isinstance(self.controller, StateFeedbackLaw):
            A_cl = self.A + self.B @ self.controller.K
            return not check_stability(A_cl) or np.any(x > 9e2)

        else:
            return np.any(x > 9e2)


if __name__ == "__main__":

    control_var = 0.5

    controller = NormalRandomControlLaw(variance=control_var)

    system = GraphLaplacian3D(controller, {'process_noise': 0.001})

    Q = np.eye(system.state_dimension)
    R = np.eye(system.input_dimension) * 1

    controller = StateFeedbackLaw(K=system.optimal_controller(Q, R))

    #system = GraphLaplacian3D(controller, {'process_noise': 0.001})

    x0 = np.ones((system.state_dimension, 1)) * 0.0
    u0 = np.ones((system.input_dimension, 1)) * 0.0

    data = system.create_trajectory(x0, n=15)

    x = data['x']
    u = data['u']

    import matplotlib.pyplot as plt

    plt.plot(x[0,:], label="x_0")
    plt.plot(x[1, :], label="x_1")
    plt.plot(u[0, :], label="u")
    plt.legend()
    plt.show()

    def lse(training_data):

        X0 = training_data['xtrain']
        X1 = training_data['ytrain']

        BA = X1 @ np.linalg.pinv(X0)

        B = BA[:, 0:3]
        A = BA[:, 3:]
        return A, B

    # training_data = system.training_data(10,10, x0=x0)
    # A, B = lse(training_data)
    #
    # print(np.sum(np.abs(A - system.A)))
    # print(np.sum(np.abs(B - system.B)))
    #

    controller = NormalRandomControlLaw(variance=control_var)
    system = GraphLaplacian3DNonLin(controller, {'process_noise': 0.001})

    from scipy.linalg import solve_discrete_are
    from prlqr.systems.pendulum import numerical_jacobian

    def f(x):

        state = x[0:3, :]
        input = x[3:, :]
        non_lin = (0.3 * np.tril(np.ones((3, 3))) @ state)**3
        return np.clip(system.A @ state + system.B @ input + non_lin, -1e3, 1e3)

    j = numerical_jacobian(f, np.array([[0.], [0.], [0.], [0.], [0.], [0.]]), 3)

    A, B = j[0:3, 0:3], j[:, 3:]

    training_data = system.training_data(40, 6, x0=x0, u0=u0)
    A, B = lse(training_data)

    print(np.sum(np.abs(A - system.A)))
    print(np.sum(np.abs(B - system.B)))
    from scipy.linalg import solve_discrete_are

    P = np.array(np.array(solve_discrete_are(A, B, Q, R)))
    K = - np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    print('Nom. controller stable:')
    print(np.all(np.linalg.eigvals(system.A+system.B@system.optimal_controller(Q,R)) < 1))

    print('LSE Controller stable:')
    print(np.all(np.linalg.eigvals(system.A+system.B@K) < 1))

    controller = StateFeedbackLaw(K=K)
    system.controller = controller
    data = system.create_trajectory(x0, n=1000)

    x = data['x']
    u = data['u']

    import matplotlib.pyplot as plt

    plt.plot(x[0,:], label="x_0")
    plt.plot(x[1, :], label="x_1")
    plt.title('LSE')
    # plt.plot(u[0, :], label="u")
    plt.legend()
    plt.show()

    from prlqr.analysis.cost_analysis import EmpiricalQuadraticCostAnalysis

    a = EmpiricalQuadraticCostAnalysis(system, Q, R, None)

    #print(a.lqr_sum(n=200,samples=200))

    controller = StateFeedbackLaw(K=system.optimal_controller(Q,R))
    system.controller = controller
    data = system.create_trajectory(x0, n=1000)

    x = data['x']
    u = data['u']

    plt.plot(x[0,:], label="x_0")
    plt.plot(x[1, :], label="x_1")
    # plt.plot(u[0, :], label="u")
    plt.title('Lin')
    plt.legend()
    plt.show()


    a = EmpiricalQuadraticCostAnalysis(system, Q, R, None)

    #print(a.lqr_sum(n=200,samples=200))

