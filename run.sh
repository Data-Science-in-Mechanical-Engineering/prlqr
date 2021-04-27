#! /bin/bash

python -m prlqr.experiment.definitions.dean_synthesis_experiment > dean_synthesis_experiment.txt 2> dean_synthesis_experiment.log

# python -m prlqr.experiment.definitions.double_integrator_experiment > double_integrator_experiment.txt 2> errors_linear.log
# python -m prlqr.experiment.definitions.pendulum_experiment > pendulum_experiment.txt 2> errors_pendulum.log
python -m prlqr.experiment.definitions.dean_experiment_paper > dean_experiment.txt 2> errors_dean.log
python -m prlqr.experiment.definitions.furuta_experiment_paper > furuta_experiment.txt 2> errors_furuta.log

