import numpy as np


class DiscreteTimeDynamicalSystem:

    def __init__(self, state_dimension, input_dimension, controller, settings):

        self.__state_dimension = state_dimension
        self.__input_dimension = input_dimension

        self.current_trajectory = list()

        self.__current_state = np.zeros((self.state_dimension, 1))

        self.current_reference_state = np.zeros((self.state_dimension, 1))
        self.current_reference_input = np.zeros((self.input_dimension, 1))

        self.controller = controller

        self.process_noise = settings.get('process_noise', 0.00)

    @property
    def state_dimension(self):
        return self.__state_dimension

    @property
    def input_dimension(self):
        return self.__input_dimension

    @property
    def current_state(self):
        return self.__current_state

    @current_state.setter
    def current_state(self, x):

        assert x.shape == (self.state_dimension, 1)

        self.current_trajectory.append(x)
        self.__current_state = x

    def reset_trajectory(self, x0):

        assert x0.shape == (self.state_dimension, 1)

        self.current_trajectory = list()
        self.__current_state = x0

    def control_law(self):
        return self.controller(self.current_state, self)

    def x_next(self, u):
        raise NotImplementedError

    def empirically_unstable(self, x, u):
        raise NotImplementedError

    # Model assumptions
    def prior_model(self, x, u):
        return np.zeros_like(x)

    def prior_jacobian(self, q_star):
        A = np.zeros((self.state_dimension, self.state_dimension))
        B = np.zeros((self.state_dimension, self.input_dimension))
        return A, B

    def create_trajectory(self, x0, n=20):

        assert x0.shape == (self.state_dimension, 1)
        assert self.controller is not None

        x = np.zeros((self.state_dimension, n+1))
        u = np.zeros((self.input_dimension, n))

        x_model = np.zeros((self.state_dimension, n+1))

        # For bookkeeping
        self.reset_trajectory(x0)

        x[:, [0]] = self.current_state
        x_model[:, [0]] = self.current_state


        for i in range(1, n+1):

            idx_prev = [i-1]
            idx = [i]

            u[:, idx_prev] = self.control_law()

            self.current_state = self.x_next(u=u[:, idx_prev])

            x[:, idx] = self.current_state

            x_model[:, idx] = self.prior_model(x[:, idx_prev], u[:, idx_prev])

        ret = {
            'x': x,
            'u': u,
            'x_model': x_model
        }
        return ret

    def training_data(self, n_trajectories, n_samples, x0, u0, jitter=0.01):

        assert x0.shape == (self.state_dimension, 1)

        Xtrain = list()
        ytrain = list()
        ymodel = list()

        # Add known operating point to the dataset
        xu = np.vstack((x0, u0))
        y = x0
        y_model = self.prior_model(x0, u0)  # In case the model doesn't the operating point (or we don't use a model :)

        assert xu.shape[1] == y.shape[1]

        y_diff = y - y_model

        Xtrain.append(xu)
        ytrain.append(y_diff)
        ymodel.append(y_model)

        for i in range(0, n_trajectories):

            random_offset = (jitter - -jitter) * np.random.random_sample(x0.shape) + -jitter

            trajectory = self.create_trajectory(x0=x0+random_offset, n=n_samples)

            xu = np.vstack((trajectory['x'][:, 0:-1], trajectory['u'][:, 0:]))
            y = trajectory['x'][:, 1:]
            y_model = trajectory['x_model'][:, 1:]

            y_diff = y - y_model

            assert xu.shape[1] == y.shape[1]

            Xtrain.append(xu)
            ytrain.append(y_diff)
            ymodel.append(y_model)

        ret = {
            'xtrain': np.hstack(Xtrain),
            'ytrain': np.hstack(ytrain),
            'y_model': np.hstack(ymodel)
        }

        # import matplotlib.pyplot as plt

        # plt.plot(ret['xtrain'].T)
        # plt.show()
        # plt.plot(ret['xtrain'].T[:,0])
        # plt.show()
        # plt.plot(ret['xtrain'].T[:,1])
        # plt.show()

        return ret

    def to_dict(self):
        data = dict()
        data['name'] = self.__class__.__name__
        return data


class ControlLaw:

    def __call__(self, x, system):
        raise NotImplementedError

    def to_dict(self):
        data = self.__dict__.copy()
        data['name'] = self.__class__.__name__
        return data


class NormalRandomControlLaw(ControlLaw):

    def __init__(self, variance):
        self.variance = variance

    def __call__(self, x, system):

        delta_u = np.random.randn(system.input_dimension, 1) * np.sqrt(self.variance)
        u = system.current_reference_input + delta_u
        return u

class FeedbackWithNormalRandomControlLaw(ControlLaw):

    def __init__(self, variance, K):
        self.variance = variance
        self.K = K

    def __call__(self, x, system):

        delta_u = np.random.randn(system.input_dimension, 1) * np.sqrt(self.variance)
        u = self.K @ (x - system.current_reference_state) + system.current_reference_input
        return u # np.clip(u + delta_u, -1., 1.)

class StateFeedbackLaw(ControlLaw):

    def __init__(self, K):
        self.K = K

    def __call__(self, x, system):
        return self.K @ (x - system.current_reference_state) + system.current_reference_input
