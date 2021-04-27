import numpy as np
import torch
import gpytorch

from prlqr.model.gp_derivative_model import DerivativeExactGPSEModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Normalizer:

    def __init__(self, input_dim, output_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.data_mean = np.zeros(output_dim)
        self.__data_std = np.ones(output_dim)
        self.input_data_range = np.ones(input_dim)
        self.input_data_min = np.zeros(input_dim)

        self.on = True

    @property
    def data_std(self):
        if self.on:
            return self.__data_std
        else:
            return np.ones(self.output_dim)

    def update(self, x, y):
        self.data_mean = np.mean(y, axis=1)
        self.__data_std = np.std(y, axis=1)
        self.input_data_range = np.max(x, axis=1) - np.min(x, axis=1)
        self.input_data_min = np.min(x, axis=1)

    def scale_input(self, x):
        if self.on:
            return (x - self.input_data_min[:,None]) / self.input_data_range[:,None]

        return x

    def scale_output(self, y):
        if self.on:
            return (1/self.data_std[:,None]) * (y - self.data_mean[:,None])

        return y

    # TODO: This will only work for a single point
    def descale_derivative(self, dmu, ds2):
        if self.on:
            factor = self.data_std / self.input_data_range
            return np.diag(factor) @ dmu,  np.diag(factor) @ ds2 @ np.diag(factor).T
        return dmu, ds2

    def descale_output(self, mu, s2):
        if self.on:
            return (self.data_std[:, None] * mu) + self.data_mean[:,None], s2 * self.data_std[:, None]**2

        return mu, s2

    def descale_sample(self, f):
        if self.on:
            return (self.data_std[:,None] * f) + self.data_mean[:,None]

        return f


class GaussianProcess:

    def __init__(self, input_dim, index, settings):
        self.x = None
        self.y = None

        self.model = None

        self.input_dim = input_dim

        # The index of the state this GP is used for
        self.index = index
        self.settings = settings

        self.normalizer = Normalizer(input_dim, output_dim=1)

        self.parameter_dict = dict()

        self.__normalize = True

        self.n = 0

    def __getNormalize(self):
        return self.normalizer.on

    def __setNormalize(self, on):
        self.normalizer.on = on

    normalize = property(__getNormalize, __setNormalize)

    @property
    def noise_variance(self):
        return self.model.likelihood.noise.item()

    def update(self, x, y):

        self.normalizer.update(x, np.atleast_2d(y))

        self.x = self.normalizer.scale_input(x)
        self.y = self.normalizer.scale_output(y)

        self.n = x.shape[1]

        train_x = torch.tensor(self.x.T).float()
        train_y = torch.tensor(self.y.reshape(-1,)).float()

        # TODO Implement The gamma prior if needed

        def param_to_uniform_prior(parameter_name, index, scale):
            params = self.settings[parameter_name][index] * scale
            lower = torch.tensor([param[0] for param in params]).squeeze()
            upper = torch.tensor([param[1] for param in params]).squeeze()

            hyperprior = gpytorch.priors.UniformPrior(lower, upper)
            constraint = gpytorch.constraints.Interval(lower, upper)
            return hyperprior, constraint

        data_var = self.normalizer.data_std ** 2
        input_data_range = self.normalizer.input_data_range

        lengthscale_hyperprior, lengthscale_constraint = param_to_uniform_prior('lengthscale_priors',
                                                                                self.index,
                                                                                scale=1/input_data_range[:,None])
        outputscale_hyperprior, outputscale_constraint = param_to_uniform_prior('signalvariance_priors',
                                                                                self.index,
                                                                                scale=1/data_var)
        noise_hyperprior, noise_constraint = param_to_uniform_prior('noisevariance_priors',
                                                                    self.index,
                                                                    scale=1/data_var)


        self.model = DerivativeExactGPSEModel(
            train_x=train_x,
            train_y=train_y,
            lengthscale_constraint=lengthscale_constraint,
            lengthscale_hyperprior=lengthscale_hyperprior,
            outputscale_constraint=outputscale_constraint,
            outputscale_hyperprior=outputscale_hyperprior,
            noise_constraint=noise_constraint,
            noise_hyperprior=noise_hyperprior,
            prior_mean=0.,
            ard_num_dims=x.shape[0]
        )
        self.model.likelihood.noise = noise_hyperprior.mean
        self.model.covar_module.base_kernel.lengthscale = lengthscale_hyperprior.mean
        self.model.covar_module.outputscale = outputscale_hyperprior.mean


    def predict(self, x_s, descale=True):

        n = x_s.shape[1]

        x_s = self.normalizer.scale_input(x_s)

        self.model.eval()
        self.model.likelihood.eval()

        x_s = torch.tensor(x_s.T).float()
        mvnd = self.model(x_s)

        mu = mvnd.mean.detach().numpy()
        s2 = mvnd.variance.detach().numpy()

        mu = mu.reshape((-1, n))
        s2 = s2.reshape((-1, n))

        if descale:
            mu, s2 = self.normalizer.descale_output(mu, s2)

        return mu, s2

    def linearize(self, x_star):

        x_star = np.array(x_star).reshape(-1, 1)


        # This already scales it's input
        f_x_star, _ = self.predict(x_star)

        x_star = self.normalizer.scale_input(x_star)

        self.model.eval()
        self.model.likelihood.eval()

        x_star = torch.tensor(x_star.T).float()

        dmu, ds2 = self.model.posterior_derivative(x_star)

        dmu = dmu.detach().numpy()
        ds2 = ds2.detach().numpy()

        dmu = dmu.reshape((-1, 1))

        ds2 = np.atleast_2d(ds2.squeeze())

        dmu, ds2 = self.normalizer.descale_derivative(dmu, ds2)

        return dmu, ds2, f_x_star

    def optimize_hyperparameters(self, restarts=1):

        d = self.x.shape[0]

        self.model.train()
        self.model.likelihood.train()

        best_hypers = {
            'likelihood.noise_covar.noise': self.model.likelihood.noise_covar.noise.detach(),
            'covar_module.base_kernel.lengthscale': self.model.covar_module.base_kernel.lengthscale.detach(),
            'covar_module.outputscale': self.model.covar_module.outputscale.detach(),
        }

        best_loss = np.Inf
        for _ in range(restarts):

            # Use the adam optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=.1)  # Includes GaussianLikelihood parameters


            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            training_iter = 200  # * d
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self.model(self.model.train_inputs[0])
                # Calc loss and backprop gradients
                loss = -mll(output, self.model.train_targets)
                loss.backward()
                # print('Iter {0}/{1} - Loss: {2}  noise: {3}, scale: {4}, length: {5}'.format(
                #     i + 1,
                #     training_iter,
                #     loss.item(),
                #     self.model.likelihood.noise.item(),
                #     self.model.covar_module.outputscale.detach().item(),
                #     self.model.covar_module.base_kernel.lengthscale.detach().numpy()
                # ))
                optimizer.step()

            output = self.model(self.model.train_inputs[0])
            nmll = -mll(output, self.model.train_targets).detach().numpy()
            # print('NMLL after optimization is {}'.format(nmll))

            if nmll < best_loss:
                # print(self.model.covar_module.base_kernel.lengthscale.detach().numpy())
                best_loss = nmll
                best_hypers = {
                    'likelihood.noise_covar.noise': self.model.likelihood.noise_covar.noise.detach(),
                    'covar_module.base_kernel.lengthscale': self.model.covar_module.base_kernel.lengthscale.detach(),
                    'covar_module.outputscale': self.model.covar_module.outputscale.detach(),
                }

            lengthscales = self.model.covar_module.base_kernel.lengthscale_prior.sample()
            outputscales = self.model.covar_module.outputscale_prior.sample()
            noise = self.model.likelihood.noise_covar.noise_prior.sample()
            hypers = {
                'likelihood.noise_covar.noise': noise,
                'covar_module.base_kernel.lengthscale': lengthscales,
                'covar_module.outputscale': outputscales,
            }
            self.model.initialize(**hypers)

        self.parameter_dict_torch = best_hypers

        # Unscaled version
        data_var = self.normalizer.data_std ** 2
        input_data_range = self.normalizer.input_data_range

        self.parameter_dict = {
                    'noise': best_hypers['likelihood.noise_covar.noise'].numpy() * data_var,
                    'lengthscale': best_hypers['covar_module.base_kernel.lengthscale'].numpy() * input_data_range,
                    'outputscale': best_hypers['covar_module.outputscale'].numpy() * data_var,
                }

        self.model.initialize(**best_hypers)

        self.model.eval()
        self.model.likelihood.eval()


if __name__ == "__main__":
    import numpy as np

    np.random.seed(0)

    def objective(x):
        return np.array([0.2, -.4]) @ x + np.random.randn(x.shape[1]) * 0.1


    Xtrain = np.random.randn(2, 100) * 10.
    ytrain = objective(Xtrain)

    gp = GaussianProcess(.01, input_dim=2)

    gp.normalize = True

    gp.update(Xtrain, ytrain)
    gp.optimize_hyperparameters()

    x_0 = np.array([[0., 0.]])

    dmu, ds2, y_0 = gp.linearize(x_0)

    print(y_0)
    print(dmu)
    print(np.sqrt(ds2))
