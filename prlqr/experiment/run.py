import numpy as np

from prlqr.analysis.cost_analysis import EmpiricalQuadraticCostAnalysis
from prlqr.systems.dynamical_system import StateFeedbackLaw
from prlqr.model.gp_dynamics_model import DynamicsModel
from prlqr.synthesis.prlqr_synth import PRLQRSyntheziser
from prlqr.synthesis.nominal_lqr_synth import NLQRSyntheziser
from prlqr.synthesis.robust_lqr_synth import RLQRSyntheziser
from prlqr.analysis.stability_analysis import LinearStabilityAnalysis

import logging
from prlqr.run_logger import configure_logger

class Run:

    def __init__(self, ids, settings):
        # assert(max(ids) < experiment.n_runs * len(experiment.n_trajectory))
        assert(min(ids) >= 0)

        self.ids = ids
        self.settings = settings
        self.optimal = True

    def run(self):

        settings = self.settings
        logging.info('{0} experiments to run'.format(len(self.ids)))

        for run_id in self.ids:

            configure_logger(settings, run_id)

            try:

                experiment_id = run_id % settings.runs

                n_trajectories = settings.training_data['size']['trajectories'][run_id // settings.runs]

                results = settings.to_dict()
                results['n_trajectories'] = n_trajectories
                results['experiment_id'] = experiment_id
                results['run_id'] = run_id


                seed = settings.seed + experiment_id
                np.random.seed(seed)
                logging.info('Started GP experiment {0} in run {1} with seed {2}.'.format(experiment_id, run_id, seed))

                training_data = self.create_training_data(n_trajectories)

                ussm = self.learn_uncertain_state_space_model(training_data)
                ussm.print()

                x0 = settings.q_star['x']
                u0 = settings.q_star['u']
                q_star = np.vstack((x0, u0))

                logging.info('True linearization')
                A_t, B_t = self.settings.system.linearize(q_star)
                logging.info('A {0}, B {1}'.format(A_t, B_t))

                controller = self.synthesis_all_controller(ussm)

                results['controller'] = controller
                results['ussm'] = ussm.to_dict()

                logging.info('Synthesis done')
                logging.info(controller)

                Q = settings.cost['Q']
                R = settings.cost['R']
                cost_analysis = EmpiricalQuadraticCostAnalysis(system=settings.system, Q=Q, R=R, settings=None)

                all_successful = True
                all_stable = True

                for K_name, K in controller.items():

                    if K is None:
                        cost = np.NaN
                        all_successful = False
                    else:
                        feedback = StateFeedbackLaw(K=K)
                        settings.system.controller = feedback

                        cost, _ = cost_analysis.lqr_sum(n=settings.cost['horizon'], samples=settings.cost['samples'])

                        if cost is np.NaN:
                            cost = np.Inf
                        if cost >= np.Inf:
                            all_stable = False

                    logging.info('Cost for {0} is {1}'.format(K_name, cost))

                    results['cost'][K_name + '_cost'] = cost

                results['cost']['all_successful'] = all_successful
                results['cost']['all_stable'] = all_stable

                logging.info('All synthesis operations were successful: {}'.format(all_successful))

                # Save results. big TODO: use something else than pickle
                self.save_results(experiment_id=run_id, results=results)

                logging.info('Run {0} done and saved.'.format(run_id))

            except Exception as e:
                raise e
                logging.info('Experiment {0} failed.'.format(run_id))
                logging.info(e)

    def create_training_data(self, n_trajectories):

        settings = self.settings

        settings.system.controller = settings.training_data['controller']
        x0 = settings.q_star['x']
        u0 = settings.q_star['u']
        training_data = settings.system.training_data(
            n_trajectories,
            settings.training_data['size']['samples'],
            x0=x0,
            u0=u0,
            jitter=settings.training_data['jitter']
        )
        return training_data

    def learn_uncertain_state_space_model(self, training_data):

        settings = self.settings
        system = settings.system

        model = DynamicsModel(system, training_data, settings)

        model.trainModel()

        x0 = settings.q_star['x']
        u0 = settings.q_star['u']

        q_star = np.vstack((x0, u0))

        ussm = model.getLinearizedModel(q_star)

        A_prior, B_prior = system.prior_jacobian(q_star)
        prior_mean = np.hstack((A_prior, B_prior))
        ussm.M = ussm.M.add_mean(prior_mean)

        return ussm

    def synthesis_all_controller(self, ussm):

        settings = self.settings
        system = settings.system

        Q = settings.cost['Q']
        R = settings.cost['R']

        try:
            synth = PRLQRSyntheziser(ussm, Q=Q, R=R, settings=settings.synthesis_settings)
            stability_analysis = LinearStabilityAnalysis(uncertainStateSpaceModel=ussm, K=None, settings=settings)

            synth_success = False
            i = 0
            while not synth_success and i < 5:
                i += 1
                K_pr = synth.synthesize()

                if K_pr is None:
                    break

                stability_analysis.K = K_pr

                eps = settings.synthesis_settings['a_posteriori_eps']
                alpha = 1 - settings.synthesis_settings['a_posteriori_prob']
                p_stability = stability_analysis.p_stability(alpha=alpha, eps=eps)

                synth_success = p_stability >= settings.synthesis_settings['stability_prob']
                logging.info('PR synthesis a posteriori analysis: {}'.format(synth_success))
                logging.info('Empirical stability {0}, required stability: {1}'.format(
                    p_stability,
                    settings.synthesis_settings['stability_prob']))


        except Exception as e:
            logging.info(e)
            logging.info("No probabilistic robust controller found return None")
            K_pr = None

        try:
            synth = RLQRSyntheziser(ussm, Q=Q, R=R, settings=settings.synthesis_settings)
            K_rob = synth.synthesize()
        except Exception as e:
            logging.info(e)
            logging.info("No robust controller found return None")
            K_rob = None

        try:
            synth = NLQRSyntheziser(ussm, Q=Q, R=R, settings=settings.synthesis_settings)
            K_nom = synth.synthesize()
        except Exception as e:
            logging.info(e)
            logging.info("No nominal controller found return None")
            K_nom = None

        # TODO: Interface?
        if hasattr(system, 'optimal_controller') and self.optimal:
            K_opt = system.optimal_controller(Q=Q, R=R)
        else:
            K_opt = None

        controller = {'K_pr': K_pr, 'K_nom': K_nom, 'K_rob': K_rob, 'K_opt': K_opt}
        return controller

    def save_results(self, experiment_id, results):
            import os
            import prlqr
            import pickle
            from datetime import datetime
            from pathlib import Path

            settings = self.settings
            system_name = results['system']['name']

            module_path = os.path.dirname(prlqr.__file__)

            save_path = module_path + '/../results/'
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = save_path + '/' + system_name + '/' + settings.name + '/'
            file_name = str(experiment_id) + '_' + timestamp + '.pickle'

            file = Path(path)
            file.mkdir(parents=True, exist_ok=True)

            full_name = path + file_name

            with open(full_name, 'wb') as handle:
                pickle.dump(results, handle)

