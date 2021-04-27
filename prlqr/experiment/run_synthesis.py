import numpy as np
from scipy.stats import wishart
from prlqr.analysis.cost_analysis import LinearQuadraticCostAnalysis
from prlqr.systems.linear_system import LinearSystem

from prlqr.experiment.run import Run
from prlqr.uncertain_state_space_model import UncertainStateSpaceModel
from prlqr.matrix_normal_distribution import MN
import logging

from prlqr.run_logger import configure_logger

# TODO: Restructure this code
class RunSynthesis(Run):

    def run(self):

        self.optimal = False
        settings = self.settings

        assert isinstance(settings.system, LinearSystem)

        system = settings.system
        print('{0} experiments to run'.format(len(self.ids)))

        for run_id in self.ids:

            configure_logger(settings, run_id)
            try:
                logging.info('Started synthesis run {0}.'.format(run_id))

                experiment_id = run_id % settings.experiments
                scale = settings.scales[run_id // settings.experiments]
                # print('{} {}'.format(experiment_id, scale))

                results = settings.to_dict()
                results['scale'] = scale
                results['experiment_id'] = experiment_id
                results['run_id'] = run_id

                logging.info('Choosen scale {0}.'.format(scale))

                seed = settings.seed + experiment_id
                np.random.seed(seed)
                logging.info('Experiment seed is {0}.'.format(seed))

                dx, du = system.state_dimension, system.input_dimension

                n_parameters = dx * dx + du * dx

                M = np.hstack((system.A, system.B))

                s = (np.eye(n_parameters) + np.ones((n_parameters, n_parameters))) / 2

                E = wishart.rvs(df=n_parameters, scale=s, size=1, random_state=seed)

                E =  E * scale
                MND = MN.from_MND(m=M, E=E, dim=(dx, du+dx))

                ussm = UncertainStateSpaceModel(M=MND,
                                                dim=(system.state_dimension, system.input_dimension),
                                                omega_var=[system.process_noise for _ in range(dx)]
                                                )

                controller = self.synthesis_all_controller(ussm)

                results['controller'] = controller
                results['ussm'] = ussm.to_dict()

                logging.info('Synthesis done')
                logging.info(controller)

                Q = settings.cost['Q']
                R = settings.cost['R']
                cost_analysis = LinearQuadraticCostAnalysis(ussm, K=None, Q=Q, R=R)

                all_successful = True

                for K_name, K in controller.items():

                    if K is None:
                        cost = np.NaN
                        all_successful = False
                    else:

                        cost_analysis.K = K
                        mean_cost = cost_analysis.expected_cost(
                            n=settings.cost['horizon'],
                            samples=settings.cost['samples'],
                            c=.9999)

                        mean_cost = np.array(mean_cost)
                        results['cost'][K_name + '_mean_cost'] = mean_cost

                        mean = np.mean(mean_cost[mean_cost < np.Inf])
                        std = np.std(mean_cost[mean_cost < np.Inf])
                        count = np.sum(mean_cost >= np.Inf)

                        logging.info('Cost for {0} is {1} +- {2}'.format(K_name, mean, std))
                        logging.info('Unstable for {0} are {1}'.format(K_name, count))

                results['cost']['all_successful'] = all_successful

                logging.info('All synthesis operations were successful: {}'.format(all_successful))

                # Save results. big TODO: use something else than pickle
                self.save_results(experiment_id=run_id, results=results)

                logging.info('Run {0} done and saved.'.format(run_id))

            except Exception as e:
                logging.error('Run {0} failed.'.format(run_id))
                logging.error(e)


