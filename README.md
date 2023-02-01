# fdt_verification
This repository accompanies a paper currently under review (citation will be added when published) that proposes an algorithm for verifying properties of fuzzy decision trees (FDTs).

## Usage
For general use, 
- [fdt_verification.py](fdt_verification.py) contains the class `FDTNode` for representing FDTs, the class `VerificationDomain` for representing convex domains over which to verify properties, the verification algorithm itself, and functions to execute common verification tasks, such as finding minimum adversarial perturbation. To verify your own FDTs using this implementation, you must map your data structure to the provided `FDTNode`. 
- [demo.ipynb](demo.ipynb) provides visual demonstrations and details the use of the various features. 
- [util.py](util.py) contains miscellaneous helper functions for loading, visualization, and experiment execution.


This was last tested in Python 3.9 and requires [cvxpy](https://www.cvxpy.org/) with a solver of choice for verification and [matplotlib](https://matplotlib.org/) for visualization in [demo.ipynb](demo.ipynb).

## Experiments

To reproduce the experiments, use [experiments.ipynb](experiments.ipynb). This additionally requires [Z3](https://pypi.org/project/z3-solver/), an SMT solver used as a baseline. We use [Gurobi](https://www.gurobi.com/) with cvxpy as the solver for our algorithm.
- [fdt_verification_z3.py](fdt_verification_z3.py) contains code for verification using Z3.
- [fdt_verification_reluval.py](fdt_verification_reluval.py) contains code for verification using the baseline with refinement strategy from [ReluVal](https://arxiv.org/abs/1804.10829).
- [saved_models](saved_models) contains pickled `FDTNode` objects for the FDTs used in the experiments.
- [robustness_points](robustness_points) contains 100 pickled test points from each dataset used for the minimum adversarial perturbation experiments.
- [results](results) contains the pickled results from our run of the experiments. 
