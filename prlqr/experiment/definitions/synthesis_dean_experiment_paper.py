import numpy as np
from prlqr.systems.linear_system import GraphLaplacian3D
from prlqr.systems.dynamical_system import NormalRandomControlLaw

system_settings = {
    'process_noise': 0.001
}
settings = {
    'name': 'synthesis_dean_experiment_paper',
    'seed': 896,  # Determined by random.org ;)
    'system_settings': system_settings,
    'system': GraphLaplacian3D(controller=None, settings=system_settings),
    # Scale for the Wishart distribution scale * 0.5 * (I+1)
    'scales': [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
    'cost': {
        'Q': np.eye(3) * 1e-3,
        'R': np.eye(3) * 1.,
        # For the empirical evaluation
        'horizon': 200,
        'samples': 10000,
    },
    'experiments': 5,
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
        'max_iter': 100,  # Iterations of the MM-Algorithm
    },
}


if __name__ == "__main__":
    from prlqr.experiment.definitions.definition import Settings
    from prlqr.experiment.run_synthesis import RunSynthesis

    settings = Settings(**settings)

    n_runs = settings.experiments * len(settings.scales)
    ids = range(0, n_runs)

    from joblib import Parallel, delayed

    def parallel_call(run_id, settings):
        ids = [run_id]

        run = RunSynthesis(ids, settings)

        run.run()

    #Parallel(n_jobs=1)(delayed(parallel_call)(run_id, settings) for run_id in ids)
    Parallel(n_jobs=-2)(delayed(parallel_call)(run_id, settings) for run_id in ids)