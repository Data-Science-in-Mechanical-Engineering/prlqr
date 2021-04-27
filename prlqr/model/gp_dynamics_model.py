import numpy as np
from prlqr.model.gp_wrapper import GaussianProcess
from prlqr.uncertain_state_space_model import UncertainStateSpaceModel
from prlqr.matrix_normal_distribution import MN

import logging


class DynamicsModel:

    def __init__(self, system, training_data, settings):
        self.system = system
        self.Xtrain = training_data['xtrain']
        self.ytrain = training_data['ytrain']
        self.normalize = settings.training_data['normalize_data']

        self.state_dim = system.state_dimension
        self.input_dim = system.input_dimension

        self.gps = list()

        self.settings = settings


    def trainModel(self):

        logging.info('Train GP on {0} data points'.format(self.Xtrain.shape[1]))
        logging.info('Raw data has {0} dimensions'.format(self.Xtrain.shape[0]))


        gps = list()
        for i in range(0, self.state_dim):

            Xtrain = self.Xtrain[:, :]
            ytrain = self.ytrain[[i], :]

            gp = GaussianProcess(
                input_dim=self.state_dim + self.input_dim,
                index=i,
                settings=self.settings.gp_settings)
            gp.normalize = self.normalize

            gp.update(Xtrain, ytrain)

            gp.optimize_hyperparameters(restarts=self.settings.gp_settings['restarts'])

            gps.append(gp)
            logging.info(gp.parameter_dict)

        self.gps = gps

    def getLinearizedModel(self, operating_point):

        # This vector contains the mean for all states followed by all inputs
        dim = self.state_dim*self.state_dim + self.input_dim*self.state_dim
        m = np.zeros((dim, 1))

        # The corresponding covariances
        E = np.zeros((dim, dim))

        omega_var = np.zeros((self.state_dim,))

        for i in range(0, self.state_dim):

            dmu, ds2, f_x_star = self.gps[i].linearize(operating_point[:, :].T)
            k = i * (self.state_dim + self.input_dim)
            l = (i+1) * (self.state_dim + self.input_dim)

            E[k:l, k:l] = ds2
            m[k:l, :] = dmu

            noise_scale = self.gps[i].normalizer.data_std
            omega_var[i] = self.gps[i].noise_variance * noise_scale**2

        dist = MN.from_MND(m=m, E=E, dim=(self.state_dim, self.state_dim+self.input_dim))
        ussm = UncertainStateSpaceModel(dist, (self.state_dim, self.input_dim), omega_var=omega_var)

        return ussm
