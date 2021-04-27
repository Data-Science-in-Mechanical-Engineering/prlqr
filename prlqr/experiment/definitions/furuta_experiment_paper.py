import numpy as np
from prlqr.systems.furuta_pendulum import FurutaPendulum
from prlqr.systems.dynamical_system import FeedbackWithNormalRandomControlLaw, NormalRandomControlLaw

system_settings = {
    'process_noise': 0.002
}
settings = {
    'name': 'furuta_experiment_paper',
    'seed': 896,  # Determined by random.org ;)
    'system_settings': system_settings,
    'system': FurutaPendulum(controller=None, settings=system_settings),
    # The time-invariant operating point
    'q_star': {
        'x': np.array([[0.],
                       [0.],
                       [0.],
                       [0.]]),
        'u': np.array([[0.]])
    },
    'cost': {
        'Q': np.diag([1, 1, 1, 1]) * 1.0,  # Multiplier solves numerical issues for LMI solver, sometimes...
        'R': np.diag([1]) * 1.0,
        # For the empirical evaluation
        'horizon': 200,
        'samples': 200
    },
    'runs': 25,  # This is per trajectories size
    'training_data': {
        'size': {
            'trajectories': [30, 50, 80],
            'samples': 30,  # Samples per trajectory
        },
        'jitter': 0.01,  # Jitter on the operating point in the data i.e. we cannot set the system exactly to it
        'normalize_data': True,
        'controller': FeedbackWithNormalRandomControlLaw(
            variance=0.01,
            K=np.array([[-0.4, 15., -0.5,  1.5]])
        ),
        'path_to_data': None,  # For systems with pre-recorded data we might ignore some settings (e.g. the controller)
    },
    'synthesis_settings': {
        # Sets the confidence interval on the posterior system distribution which we consider for controller synthesis
        # This applies to the prob. robust and the robust setting
        'confidence_interval': .98,
        # For the probabilistic robust synthesis
        'stability_prob': .98,  # Probability of a system in the model posterior to be stable (1-eps in the paper)
        'synthesis_prob': .80,  # in this fraction of cases (synthesis runs)
        'a_posteriori_eps': .01,  # Epsilon the a posteriori stability analysis deviates from the (model) truth
        'a_posteriori_prob': 0.999,  # in this fraction of cases
        'controllability_tol': 1e-5,
        'max_iter': 25,  # Iterations of the MM-Algorithm
    },
    'gp_settings': {
        # For uniform distribution
        'lengthscale_priors': [
            [(45.0, 50.0), (45.0, 50.0), (15.0, 20.0), (45.0, 50.0), (27.0, 31.0)],
            [(45.0, 50.0), (25.0, 30.0), (45.0, 50.0), (7.00, 15.0), (5.00, 10.0)],
            [(30.0, 35.0), (0.20, 0.40), (45.0, 50.0), (45.0, 50.0), (1.80, 2.30)],
            [(13.0, 15.0), (0.20, 0.40), (25.0, 30.0), (7.00, 10.0), (2.00, 2.50)]
        ],
        'signalvariance_priors': [
            [(0.01, .025)],
            [(0.01, .025)],
            [(0.01, .025)],
            [(0.01, .025)]
        ],
        'noisevariance_priors': [
            [(0.01, 0.03)],
            [(0.01, 0.03)],
            [(0.01, 0.03)],
            [(0.01, 0.03)]
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