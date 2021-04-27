
# Probabilistic robust linear quadratic regulators with Gaussian processes

This repository is part of the supplementary material
for the paper titled 
**Probabilistic robust linear quadratic regulators with Gaussian processes**
by *Alexander von Rohr*, *Matthias Neumann-Brosig* and *Sebastian Trimpe*.

The paper is going to be presented at the 3rd Annual Learning for Dynamics & Control Conference (L4DC) and will be published electronically in the Proceedings of Machine Learning Research (PMLR).

If you are finding this code useful please let us know and get in contact.

When you are using this or parts of this code in an academic setting please cite the above paper:

> 📋 TODO: Add bibtex after official publication as well as arxiv link

## Abstract

Probabilistic models such as Gaussian processes (GPs) are powerful tools to learn unknown dynamical systems from data for subsequent use in control design.
While learning-based control has the potential to yield superior performance in demanding applications, robustness to uncertainty remains an important challenge.
Since Bayesian methods quantify uncertainty of the learning results, it is natural to incorporate these uncertainties into a robust design. 
In contrast to most state-of-the-art approaches that consider worst-case estimates, we leverage the learning methods' posterior distribution in the controller synthesis. 
The result is a more informed and thus more efficient trade-off between performance and robustness.
We present a novel controller synthesis for linearized GP dynamics that yields robust controllers with respect to a probabilistic stability margin.
The formulation is based on a recently proposed algorithm for linear quadratic control synthesis, which we extend by giving probabilistic robustness guarantees in the form of credibility bounds for the system's stability.
Comparisons to existing methods based on worst-case and certainty-equivalence designs reveal superior performance and robustness properties of the proposed method.

## How to use the supplementary code

### Install dependencies

This project uses pipenv (https://pypi.org/project/pipenv/) to manage dependencies
I recommend using pyenv (https://github.com/pyenv/pyenv) to manage your python version.

When you have pipenv and the correct python version installed run

```
pipenv install
```

You als need to have MOSEK (https://www.mosek.com/) installed for the LMI based synthesis.
We use PICOS (https://pypi.org/project/PICOS/) as interface to the underlying solver. 
That means, in principle, it is possible to replace MOSEK with CVXOPT (https://cvxopt.org/) without many changes, but we ran into trouble using the proposed LMI formulation with CVXOPT.
Do so at your own risk.

Once you have installed all dependencies you can start the python virtual environment:

```
pipenv shell
```

### Reproducing the figures

The data presented in the paper is part of this repository and can be found in the *results* folder.
To reproduce the figures presented in the paper and based on this data you can rerun the script to create the plots.

For the synthetic distribution presented in section 4.1:

```
python -m prlqr.experiment.load_synthesis_results

```

inside the pipenv shell. You will find the resulting plots in the *figures* folder.
We ran the synthesis for 5 different samples of the covariance matrix. The paper contains the third run.


For the distribution resulting from a GP posterior presented in section 4.2:

```
python -m prlqr.experiment.load_gp_results
```

### Reproducing results

Before running the experiments make sure to rename the experiment otherwise the data will get mixed up.
To rename the experiment open the corresponding file in *experiment/definitions/* and change the line

```python
    'name': 'synthesis_dean_experiment_paper',
```

to something like 
```python
    'name': 'synthesis_dean_experiment_reproduce',
```

You can also change any parameters of the algorithm or GP model exposed in the settings file.

Afterwards you can directly run the script.

**Warning:** Running the script may take a few minutes/hours and will use multiple cores on your machine.

For the synthetic distribution (section 4.1) run
```
python -m prlqr.experiment.definitions.synthesis_dean_experiment_paper
```

For the synthetic system (section 4.2.1) run:

```
python -m prlqr.experiment.definitions.dean_experiment_paper
```

For the rotatory pendulum (section 4.2.2) run:

```
python -m prlqr.experiment.definitions.furuta_experiment_paper
```

There are two additional examples implemented, a simple double integrator and a two-dimensional pendulum that have not been presented in the paper. 
