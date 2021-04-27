import numpy as np
from prlqr.systems.linear_system import GraphLaplacian3DNonLin
from prlqr.systems.dynamical_system import NormalRandomControlLaw

system_settings = {
    'process_noise': 0.001
}
settings = {
    'name': 'dean_experiment_paper',
    'seed': 896,  # Determined by random.org ;)
    'system_settings': system_settings,
    'system': GraphLaplacian3DNonLin(controller=None, settings=system_settings),
    # The time-invariant operating point
    'q_star': {
        'x': np.array([[0.],
                       [0.],
                       [0.]]),
        'u': np.array([[0.],
                       [0.],
                       [0.]])
    },
    'cost': {
        'Q': np.eye(3) * 1,
        'R': np.eye(3) * 1.,
        # For the empirical evaluation
        'horizon': 200,
        'samples': 200
    },
    'runs': 25,
    'training_data': {
        'size': {
            'trajectories': [3, 5, 8],
            'samples': 6,  # Samples per trajectory
        },
        'jitter': 0.02,  # Jitter on the operating point in the data i.e. we cannot set the system exactly to it
        'normalize_data': True,
        'controller': NormalRandomControlLaw(variance=.25),
        'path_to_data': None,  # For systems with pre-recorded data we might ignore some settings (e.g. the controller)
    },
    'synthesis_settings': {
        # Sets the confidence interval on the posterior system distribution which we consider for controller synthesis
        # This applies to the prob. robust and the robust setting
        'confidence_interval': .98,
        # For the probabilistic robust synthesis
        'stability_prob': .98,  # Probability of a system in the model posterior to be stable
        'synthesis_prob': .80,  # in this fraction of cases (synthesis runs)
        'a_posteriori_eps': .01,  # Epsilon the a posteriori stability analysis deviates from the (model) truth
        'a_posteriori_prob': 0.999,  # in this fraction of cases
        'controllability_tol': 5e-4,
        'max_iter': 50,  # Iterations of the MM-Algorithm
    },
    'gp_settings': {
        'lengthscale_priors': [
            [(0.30, 10.0), (0.30, 10.0), (0.30, 10.0), (0.30, 10.0), (0.30, 10.0), (0.30, 10.0)],
            [(0.30, 10.0), (0.30, 10.0), (0.30, 10.0), (0.30, 10.0), (0.30, 10.0), (0.30, 10.0)],
            [(0.30, 10.0), (0.30, 10.0), (0.30, 10.0), (0.30, 10.0), (0.30, 10.0), (0.30, 10.0)]
        ],
        'signalvariance_priors': [
            [(45., 50.)],
            [(45., 50.)],
            [(45., 50.)],
        ],
        'noisevariance_priors': [
            [(0.0005, 0.0015)],
            [(0.0005, 0.0015)],
            [(0.0005, 0.0015)],
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


