import torch
import gpytorch

# Based on an implementation by Sarah Müller (smueller@is.mpg.de)

class ExactGPSEModel(gpytorch.models.ExactGP):
    """An exact Gaussian process (GP) model with a squared exponential (SE) kernel.

    ExactGP: The base class of gpytorch for any Gaussian process latent function to be
        used in conjunction with exact inference.
    GPyTorchModel: The easiest way to use a GPyTorch model in BoTorch.
        This adds all the api calls that botorch expects in its various modules.

    Attributes:
        train_x: (size N x D) The training features X.
        train_y: (size N x 1) The training targets y.
    """

    def __init__(
            self,
            train_x,
            train_y,
            lengthscale_constraint=None,
            lengthscale_hyperprior=None,
            outputscale_constraint=None,
            outputscale_hyperprior=None,
            noise_constraint=None,
            noise_hyperprior=None,
            ard_num_dims=None,
            prior_mean=0,
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint, noise_prior=noise_hyperprior
        )
        super(ExactGPSEModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        if prior_mean != 0:
            self.mean_module.initialize(constant=prior_mean)
            self.mean_module.constant.requires_grad = False

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_hyperprior,
                lengthscale_constraint=lengthscale_constraint,
            ),
            outputscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint,
        )
        # Initialize lengthscale and outputscale to mean of priors
        if lengthscale_hyperprior is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale_hyperprior.mean
        if outputscale_hyperprior is not None:
            self.covar_module.outputscale = outputscale_hyperprior.mean

    def forward(self, x):
        """Compute the prior latent distribution on a given input.

        Typically, this will involve a mean and kernel function. The result must be a
        MultivariateNormal. Calling this model will return the posterior of the latent
        Gaussian process when conditioned on the training data. The output will be a
        MultivariateNormal.

        Args:
            x: (size n x D) The test points.

        Returns:
            A MultivariateNormal.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DerivativeExactGPSEModel(ExactGPSEModel):
    """Derivative of the ExactGPSEModel w.r.t. the test points x.

    Since differentiation is a linear operator this is again a Gaussian process.

    Attributes:
        train_x: (N x D) The training features X.
        train_y: (N x 1) The training targets y.
    """

    def __init__(
            self,
            train_x,
            train_y,
            lengthscale_constraint=None,
            lengthscale_hyperprior=None,
            outputscale_constraint=None,
            outputscale_hyperprior=None,
            noise_constraint=None,
            noise_hyperprior=None,
            ard_num_dims=None,
            prior_mean=0.0,
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        super(DerivativeExactGPSEModel, self).__init__(
            train_x,
            train_y,
            lengthscale_constraint,
            lengthscale_hyperprior,
            outputscale_constraint,
            outputscale_hyperprior,
            noise_constraint,
            noise_hyperprior,
            ard_num_dims,
            prior_mean,
        )
        self.N, self.D = self.train_inputs[0].shape

    def append_train_data(self, train_x, train_y):
        self.set_train_data(
            inputs=torch.cat([self.train_inputs[0], train_x]),
            targets=torch.cat([self.train_targets, train_y]),
            strict=False,
        )
        self.posterior(
            train_x
        )  # Call this to update prediction strategy of GPyTorch (get_L_lower, get_K_XX_inv).
        self.N += train_x.shape[0]

    def get_L_lower(self):
        return (
            self.prediction_strategy.lik_train_train_covar.root_decomposition()
                .root.evaluate()
                .detach()
        )

    def get_KXX_inv(self):
        L_inv_upper = self.prediction_strategy.covar_cache.detach()
        KXX_inv = L_inv_upper @ L_inv_upper.transpose(0, 1)

        # test = (KXX_inv == KXX_inv.transpose(0, 1)).all()

        return KXX_inv

    def get_KXX_inv_test(self):
        X = self.train_inputs[0]
        sigma_n = self.likelihood.noise_covar.noise.detach()
        return torch.inverse(
            self.covar_module(X).evaluate() + torch.eye(X.shape[0]) * sigma_n
        )

    def _get_KxX_dx(self, x):
        """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
        X = self.train_inputs[0]
        n = x.shape[0]
        K_xX = self.covar_module(x, X).evaluate()
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        return (
                -torch.eye(self.D)
                / lengthscale ** 2
                @ (
                        (x.view(n, 1, self.D) - X.view(1, self.N, self.D))
                        * K_xX.view(n, self.N, 1)
                ).transpose(1, 2)
        )

    def _get_Kxx_dx2(self):
        """Computes the analytic second derivative of the kernel K(x,x) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D x D) The second derivative of K(x,x) w.r.t. x.
        """
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        sigma_f = self.covar_module.outputscale.detach()
        return (torch.eye(self.D) / lengthscale ** 2) * sigma_f

    def posterior_derivative(self, x):
        """Computes the posterior of the derivative of the GP w.r.t. the given test
        points x.

        Args:
            x: (n x D) Test points.

        Returns:
            A GPyTorchPosterior.
        """
        K_xX_dx = self._get_KxX_dx(x)
        mean_d = K_xX_dx @ self.get_KXX_inv() @ self.train_targets
        variance_d = self._get_Kxx_dx2() - K_xX_dx @ self.get_KXX_inv() @ K_xX_dx.transpose(1, 2)
        variance_d = variance_d.clamp_min(1e-9)

        # K_xX_dx = K_xX_dx.detach().numpy().squeeze().astype(float)
        # K = self.get_KXX_inv().detach().numpy().squeeze().astype(float)
        # kk = K_xX_dx @ K @ K_xX_dx.T
        # test = np.allclose(kk, kk.T)

        # TODO: There are some numerical (?) issues here that make the matrix sometimes non-symmetric.
        #  This even happens when KXX_inv is symmetric.
        #  This is an ugly hack to fix that.
        #  Possible solution: Do all of these calculations with double precision (in numpy?)
        variance_d = (variance_d + variance_d.transpose(1, 2))/2
        # test = (variance_d == variance_d.transpose(1, 2)).all()

        return mean_d, variance_d


if __name__ == "__main__":
    import numpy as np

    def objective(x):
        return np.array([0.2, -.4]) @ x + np.random.randn(x.shape[1]) * 0.1


    Xtrain = np.random.randn(2, 100)
    ytrain = objective(Xtrain)

    train_x = torch.tensor(Xtrain.T).float()
    train_y = torch.tensor(ytrain).float()


    gp = DerivativeExactGPSEModel(train_x=train_x, train_y=train_y, ard_num_dims=2)

    gp.train()
    gp.likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    training_iter = 100
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = gp(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            gp.likelihood.noise.item()
        ))
        optimizer.step()

    gp.eval()
    gp.likelihood.eval()

    x_0 = torch.tensor(np.array([[0., 0.]])).float()

    gp(x_0)

    lin = gp.posterior_derivative(x_0)
