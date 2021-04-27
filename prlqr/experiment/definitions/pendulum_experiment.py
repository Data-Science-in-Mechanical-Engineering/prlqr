import numpy as np
from prlqr.systems.pendulum import Pendulum
from prlqr.systems.dynamical_system import NormalRandomControlLaw


system_settings = {
    'process_noise': 0.002
}
settings = {
    'name': 'pendulum_experiment',
    'seed': 896,  # Determined by random.org ;)
    'system_settings': system_settings,
    'system': Pendulum(controller=None, settings=system_settings),
    # The time-invariant operating point
    'q_star': {
        'x': np.array([[np.pi],
                       [0.]]),
        'u': np.array([[0.]])
    },
    'cost': {
        'Q': np.eye(2) * 1,
        'R': np.eye(1) * 1,
        # For the empirical evaluation
        'horizon': 200,
        'samples': 200
    },
    'runs': 25,  # This is per trajectories size
    'training_data': {
        'size': {
            'trajectories': [8, 13, 21],
            'samples': 30,  # 50,  # Samples per trajectory
        },
        'jitter': 0.01,  # Jitter on the operating point in the data i.e. we cannot set the system exactly to it
        'normalize_data': True,
        'controller': NormalRandomControlLaw(variance=0.01),
        'path_to_data': None,  # For systems with pre-recorded data we might ignore some settings (e.g. the controller)
    },
    'synthesis_settings': {
        # Sets the confidence interval on the posterior system distribution which we consider for controller synthesis
        # This applies to the prob. robust and the robust setting
        'confidence_interval': .98,
        # For the probabilistic robust synthesis
        'stability_prob': .93,  # Probability of a system in the model posterior to be stable (1-eps in the paper)
        'synthesis_prob': .80,  # in this fraction of cases (synthesis runs)
        'a_posteriori_eps': .01,  # Epsilon the a posteriori stability analysis deviates from the (model) truth
        'a_posteriori_prob': 0.999,  # in this fraction of cases
        'controllability_tol': 1e-4,
        'max_iter': 50,  # Iterations of the MM-Algorithm
    },
    'gp_settings': {
        # For uniform distribution
        # 'lengthscale_priors': [
        #     [(.01, 60.0), (.01, 60.0), (.01, 60.0)],
        #     [(.01, 60.0), (.01, 60.0), (.01, 60.0)]
        # ],
        # 'signalvariance_priors': [
        #     [(0.001, 10.0)],
        #     [(0.001, 10.)]
        # ],
        # 'noisevariance_priors': [
        #     [(0.001, 0.005)],
        #     [(0.001, 0.005)]
        # ],
        'lengthscale_priors': [
            [(40.0, 60.0), (5.0, 15.0), (5.0, 15.0)],
            [(1., 5.0), (40.0, 60.0), (3.0, 10.0)]
        ],
        'signalvariance_priors': [
            [(0.001, 0.05)],
            [(0.01, 0.5)]
        ],
        'noisevariance_priors': [
            [(0.001, 0.005)],
            [(0.001, 0.005)]
        ],
        'restarts': 0
    }
}


if __name__ == "__main__":
    from prlqr.experiment.definitions.definition import Settings
    from prlqr.experiment.run import Run

    settings = Settings(**settings)
    ids = range(0, settings.runs * len(settings.training_data['size']['trajectories']))

    from joblib import Parallel, delayed

    def parallel_call(run_id, settings):
        ids = [run_id]

        run = Run(ids, settings)

        run.run()

    #Parallel(n_jobs=1)(delayed(parallel_call)(run_id, settings) for run_id in ids)
    Parallel(n_jobs=-3)(delayed(parallel_call)(run_id, settings) for run_id in ids)