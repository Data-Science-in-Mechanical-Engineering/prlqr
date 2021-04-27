from prlqr.systems.dynamical_system import DiscreteTimeDynamicalSystem, StateFeedbackLaw
from prlqr.analysis.stability_analysis import check_stability
from prlqr.systems.pendulum import numerical_jacobian

import numpy as np
from scipy.integrate import solve_ivp
import math


class FurutaPendulum(DiscreteTimeDynamicalSystem):

    def __init__(self, controller, settings):
        self.simulator = FurutaSimulator()
        self.prior_simulator = FurutaSimulator()

        self.prior_simulator.mp = self.prior_simulator.mp*1.2
        self.prior_simulator.Lp = self.prior_simulator.Lp*1.2

        self.prior_simulator.r = self.prior_simulator.r*1.2
        self.prior_simulator.mr = self.prior_simulator.mr*1.2

        def f(x):
            return self.prior_simulator.x_next(x[0:4, :], x[4, :])

        j = numerical_jacobian(f, np.array([[0.], [0.], [0.], [0.], [0.]]), 4)

        A, B = j[0:4,0:4], j[:, 4:]

        self.prior_A = np.eye(4)
        self.prior_B = np.zeros_like(B)

        super().__init__(state_dimension=4, input_dimension=1, controller=controller, settings=settings)

    def x_next(self, u):
        noise = np.random.randn(self.state_dimension, 1) * np.sqrt(self.process_noise)
        return self.simulator.x_next(self.current_state, u) + noise

    def empirically_unstable(self, x, u):
        return np.abs(x[1]) > np.pi / 4

    def linearize(self, q):

        def f(x):
            return self.simulator.x_next(x[0:4, :], x[4, :])

        j = numerical_jacobian(f, q, self.state_dimension)

        A, B = j[0:4,0:4], j[:, 4:]

        return A, B

    def optimal_controller(self, Q, R):
        from scipy.linalg import solve_discrete_are

        A, B = self.linearize(np.array([[0.], [0.], [0.], [0.], [0.]]))

        P = np.array(solve_discrete_are(A, B, Q, R))
        K_opt = - np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        return K_opt

    def prior_model(self, x, u):
        A, B = self.prior_jacobian(q_star=np.array([[0.], [0.], [0.], [0.], [0.]]))
        return A @ x + B @ u

    def prior_jacobian(self, q_star):
        return self.prior_A, self.prior_B


class FurutaSimulator(object):

    def __init__(self, mr=0.095, r=.085, mp=0.024, Lp=0.129):

        # Motor
        # Resistance
        self.Rm = 8.4
        # Current-torque (N-m/A)
        self.kt = 0.042
        # Back-emf constant (V-s/rad)
        self.km = 0.042

        # Rotary Arm
        # Mass (kg)
        self.mr = mr  # 0.095
        # Total length (m)
        self.r = r  # 0.085
        # Moment of inertia about pivot (kg-m^2)
        self.Jr = self.mr * self.r ** 2 / 3  # Jr = Mr*r^2/12
        # Equivalent Viscous Damping Coefficient (N-m-s/rad)
        self.br = -0.00005  # damping tuned heuristically to match QUBE-Sero 2 response

        # Pendulum Link
        # Mass (kg)
        self.mp = mp  # 0.024
        # Total length (m)
        self.Lp = Lp  # 0.129
        # Pendulum center of mass (m)
        self.l = self.Lp / 2
        # Moment of inertia about pivot (kg-m^2)
        self.Jp = self.mp * self.Lp ** 2 / 3  # Jp = mp*Lp^2/12;
        # Equivalent Viscous Damping Coefficient (N-m-s/rad)
        self.bp = -0.00003  # damping tuned heuristically to match QUBE-Sero 2 response
        # Gravity Constant
        self.g = 9.81

    def diff_forward_model(self, t, state, action):
        theta = state[0]
        alpha = state[1]
        theta_dot = state[2]
        alpha_dot = state[3]
        tau = (self.km * (-action - self.km * theta_dot)) / self.Rm  # torque

        cosalpha = math.cos(alpha)
        sinalpha = math.sin(alpha)
        Jeq = self.mp ** 2 * self.r ** 2 * self.l ** 2 * cosalpha ** 2 - self.Jp ** 2 * sinalpha ** 2 - self.Jr * self.Jp

        # EQUATIONS with alpha = 0 when pendulum up
        theta_dot_dot = 1/Jeq * (
                -cosalpha**2*sinalpha*self.l*self.r*self.Jp*self.mp*theta_dot**2 +
                (2*cosalpha*sinalpha*self.Jp**2*alpha_dot+cosalpha*self.l*self.r*self.br*self.mp) *
                theta_dot+self.mp*self.r*self.l*sinalpha*self.Jp*alpha_dot**2+self.Jp*self.bp*alpha_dot -
                self.Jp*tau-cosalpha*sinalpha*self.g*self.l**2*self.r*self.mp**2
        )
        #
        alpha_dot_dot = 1/Jeq * (
                (-cosalpha*sinalpha**3*self.Jp**2-cosalpha*sinalpha*self.Jp*self.Jr) * theta_dot**2 +
                (
                        2*cosalpha**2*sinalpha*self.l*self.r*self.Jp*self.mp*alpha_dot
                        + sinalpha**2*self.Jp*self.br+self.Jr*self.br
                ) * theta_dot +
                cosalpha*sinalpha*self.l**2*self.r**2*self.mp**2*alpha_dot**2 +
                cosalpha*self.l*self.r*self.bp*self.mp*alpha_dot -
                cosalpha*self.l*self.r*self.mp*tau -
                sinalpha**3*self.g*self.l*self.Jp*self.mp -
                sinalpha*self.g*self.l*self.Jr*self.mp
        )

        return [theta_dot, alpha_dot, theta_dot_dot, alpha_dot_dot]


    def forward_model(self, theta, alpha, theta_dot, alpha_dot, Vm, dt):

        # Calculate dynamics with transformed alpha
        # Alpha transformation just if alpha=0 when down equations
        #alpha += np.pi
        state = np.array([theta, alpha, theta_dot, alpha_dot])
        next_state = np.array(
            solve_ivp(lambda t, state : self.diff_forward_model(t, state, Vm), [0, dt], state, method='RK45').y)[:, -1]
        theta = next_state[0]
        alpha = next_state[1]
        theta_dot = next_state[2]
        alpha_dot = next_state[3]

        # Normalize to range of -pi to pi
        # theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        # alpha = ((alpha + np.pi) % (2 * np.pi)) - np.pi

        return theta, alpha, theta_dot, alpha_dot


    def x_next(self, x, u):
        dt = 1./50.

        theta = x[0, 0]
        alpha = x[1, 0]
        theta_dot = x[2, 0]
        alpha_dot = x[3, 0]

        Vm = u[0]

        theta, alpha, theta_dot, alpha_dot = self.forward_model(theta, alpha, theta_dot, alpha_dot, Vm, dt)
        x_p = np.array([
            [theta],
            [alpha],
            [theta_dot],
            [alpha_dot]
        ])

        return x_p


if __name__ == "__main__":

    control_var = 0.5

    from prlqr.systems.dynamical_system import FeedbackWithNormalRandomControlLaw
    controller = FeedbackWithNormalRandomControlLaw(variance=control_var, K=None)

    system = FurutaPendulum(controller, {'process_noise': 0.001})

    Q = np.eye(system.state_dimension)
    R = np.eye(system.input_dimension) * 1

    controller.K = system.optimal_controller(Q, R)
    #system = GraphLaplacian3D(controller, {'process_noise': 0.001})

    x0 = np.ones((system.state_dimension, 1)) * 0.0
    u0 = np.ones((system.input_dimension, 1)) * 0.0

    data = system.create_trajectory(x0, n=200)

    x = data['x']
    u = data['u']

    import matplotlib.pyplot as plt

    plt.plot(x[0,:], label="Theta")

    plt.plot(u[0, :], label="u")
    plt.legend()
    plt.show()