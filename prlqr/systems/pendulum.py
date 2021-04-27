import numpy as np
from prlqr.systems.dynamical_system import DiscreteTimeDynamicalSystem, StateFeedbackLaw, NormalRandomControlLaw

def numerical_jacobian(f, x, m, dx=10**-10):
    n = x.shape[0]
    jac = np.zeros((m, n))
    for j in range(n):
        x_plus = np.array([(xi if k != j else xi + dx) for k, xi in enumerate(x)])
        x_minus = np.array([(xi if k != j else xi - dx) for k, xi in enumerate(x)])

        jac[:, [j]] = (f(x_plus) - f(x_minus)) / (2 * dx)
    return jac

class Rk4DiscretizedSystem(DiscreteTimeDynamicalSystem):

    def __init__(self, internal_state_dimension, input_dimension, controller, settings):
        super().__init__(internal_state_dimension, input_dimension, controller, settings)
        self.delta_t = 0.02

    # General method for solving ODE of dynamics with RK4
    def rk4(self, u, x):
        k1 = self.dynamics(x, u) * self.delta_t
        xk = x + k1 / 2
        k2 = self.dynamics(xk, u) * self.delta_t
        xk = x + k2 / 2
        k3 = self.dynamics(xk, u) * self.delta_t
        xk = x + k3
        k4 = self.dynamics(xk, u) * self.delta_t
        xnext = x + (k1 + 2 * (k2 + k3) + k4) / 6

        return xnext

    def x_next(self, u):
        noise = np.random.randn(self.state_dimension, 1) * np.sqrt(self.process_noise)
        # Avoid 'unstability' in empirical tests
        # noise = np.clip(a = noise, a_min=-2 * np.sqrt(self.process_noise), a_max=2 * np.sqrt(self.process_noise))
        x = self.current_state
        return self.rk4(u, x) + noise
    
    def dynamics(self, x, u):
        raise NotImplementedError


class Pendulum(Rk4DiscretizedSystem):

    def __init__(self, controller, settings):
        super().__init__(internal_state_dimension=2, input_dimension=1, controller=controller, settings=settings)

        self.g = 9.8
        self.length = 1.
        self.m = .1
        self.k = 0.05
        self.gain = 1

        self.current_reference_state = np.array([
            [np.pi],
            [0]
        ])

    def prior_model(self, x, u):
        return x

    def prior_jacobian(self, q_star):
        A = np.eye(self.state_dimension)
        B = np.zeros((self.state_dimension, self.input_dimension))
        return A, B

    def dynamics(self, x, u):

        theta = x[[0], :]
        thetadot = x[[1], :]
        theta_next = thetadot
        thetadot_next = - ((self.g / self.length) * np.sin(theta)) \
                        - ((self.k / (self.m * self.length**2)) * thetadot) \
                        - ((self.gain/self.m) * u)

        xnext = np.concatenate((theta_next, thetadot_next), axis=0)

        return xnext

    # The pendulum is unstable if the angle is bigger than 90 degrees
    def empirically_unstable(self, x, u):
        return np.abs(x[0]) < np.pi * (1 / 2) or np.abs(x[0]) > np.pi * (3 / 2)

    def linearize(self, q):

        def f(x):
            return self.rk4(x=x[0:2, :], u=x[2, :])

        j = numerical_jacobian(f, q, self.state_dimension)

        A, B = j[0:2,0:2], j[:, 2:]

        return A, B

    def optimal_controller(self, Q, R):
        from scipy.linalg import solve_discrete_are

        A, B = self.linearize(np.array([[np.pi], [0.], [0.]]))

        P = np.array(solve_discrete_are(A, B, Q, R))
        K_opt = - np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        return K_opt


if __name__ == "__main__":

    # controller = RandomControlLaw(variance=.01)

    K = np.array([[3.40111202, 2.43789606]])
    controller = StateFeedbackLaw(K=K)


    system = Pendulum(controller, {'process_noise': 0.002})

    Q = np.eye(2)
    R = np.eye(1)
    K = system.optimal_controller(Q, R)
    controller = StateFeedbackLaw(K=K)
    controller = NormalRandomControlLaw(variance=.01)
    system.controller = controller
    x0 = np.array([
        [np.pi],
        [0.0]
    ])

    data = system.create_trajectory(x0, n=80)

    x = data['x']
    u = data['u']

    for i in range(u.shape[1]):
        unstable = system.empirically_unstable(x[:, i], u[:, i])
        if unstable:
            print('Unstable {}'.format(i))
            break

    import matplotlib.pyplot as plt

    plt.plot(x[0,:] / np.pi, label="x_0")
    plt.plot(x[1, :], label="x_1")
    plt.plot(u[0, :], label="u")
    plt.legend()
    plt.show()

    print(x.mean(axis=1))