# Iterative Linear Quadratic Control Toolbox
The goal of this toolbox is to analyze empirically the optimization performance of several algorithms for nonlinear
 control from an optimization viewpoint on several synthetic benchmarks.

A companion report [ilqc_algos](https://arxiv.org/abs/2207.06362) is available and details the implementation of the algorithms
 in a common differentiable programming framework. A theoretical analysis of the algorithms under appropriate assumptions is presented in the manuscript [ilqc_theory](https://arxiv.org/abs/2204.02322).

If this software is useful for your research, please consider citing it as

```
@article{roulet2022iterative,
  title={Iterative Linear Quadratic Optimization for Nonlinear Control: Differentiable Programming Algorithmic Templates},
  author={Roulet, Vincent and Srinivasa, Siddhartha and Fazel, Maryam and Harchaoui, Zaid},
  journal={arXiv preprint arXiv:2207.06362},
  year={2022}
}
```

## Introduction
Consider controlling a system such as a car to make it perform a task such as racing a track in a finite time. The
 movement of the system is generally determined by a *nonlinear* differential equation driven by a control input. To
  determine an optimal control law for a given task determined by some costs, the system is usually discretized to
   define an optimization problem in a sequence of control variables. The resulting optimization problem is generally
    nonconvex, yet, experiments demonstrate that nonlinear control algorithms tailored for such problems can exhibit fast
    convergence to reasonable or even optimal controllers. The objective of this toolbox is to study such phenomena.

#### Algorithms
Algorithms for discretized nonlinear control problems exploit the dynamical structure of the problem to compute
 candidate improved solutions at each step as detailed in [ilqc_algos](https://arxiv.org/abs/2207.06362). These algorithms can
  be classified depending on the approximations they used on the problem at each iteration: linear/quadratic
   approximations of the dynamics, linear/quadratic approximations of the costs. The algorithms we consider in this
    toolbox are
- a *gradient descent* (linear approx. of the dynamics, linear approx. of the costs),
- a *Gauss-Newton* method and its *Differentiable Dynamic Programming (DDP) variant* (linear approx. of the dynamics
, quadratic approx. of the costs),
- a *Newton method* and its *DDP variant* (quadratic approx. of the dynamics, quadratic approx. of the costs).

Furthermore, we consider several line-search strategies to update the current candidate solution (see [ilqc_algos
](papers/ilqc_algos.pdf) for more details).

In addition to the implementation of the above algorithms for finite-horizon control, we implemented a model
 predictive controller for autonomous car racing on different tracks.

#### Environments
To study these algorithms we consider the task of swinging up a fixed pendulum or a pendulum on a cart and autonomous
car racing with either a simplified model of a car or a bicycle model of a car. The latter is a model developed by
 [Liniger et al, 2017](https://arxiv.org/abs/1711.07300) by carefully studying the real dynamics of a miniature car,
 see e.g. this [video](https://www.youtube.com/watch?v=mXaElWYQKC4). We reimplemented these dynamics in Pytorch to
  further study the numerical performance of optimization algorithms in realistic settings.


## Installation
To install the dependencies, create a conda environment with
``conda env create -f ilqc.yml``
Note that to visualize the environments, a specific version of pyglet needs to be installed, 
due to backward incompatible changes done by pyglet.
Activate the environment, using
``conda activate ilqc``
and install pytorch (see https://pytorch.org/ to find the adequate command line for your OS); for example on a mac, do
``conda install pytorch -c pytorch``.

## Case example

#### Finite horizon control
To optimize for example the racing of a simple model of a car on a simpe track create a python file in the repository
 such as
 ```
import torch
import seaborn as sns
from matplotlib import pyplot as plt
from envs.car import Car
from algorithms.run_min_algo import run_min_algo

torch.set_default_tensor_type(torch.DoubleTensor)

# Create nonlinear control task
env = Car(model='simple', track='simple', cost='exact', reg_bar=0., horizon=50)

# Optimize the task with a DDP algorithm using linear quadratic approximations
cmd_opt, _, metrics = run_min_algo(env, algo='ddp_linquad_reg', max_iter=20)

# Visualize the movement
env.visualize(cmd_opt)

# Plot the costs along the iterations of the algorithm
sns.lineplot(x='iteration', y='cost', data=metrics)
plt.show()

```

A set of default experiments is present in `finite_horizon_control/fhc_example.py`. You can simply run the file from
 the root of the repository using `python finite_horizon_control/fhc_example.py` to observe optimal controllers for
  the tasks decribed above.

#### Model predictive control
To observe a controller computed by a model predicitve control approach run `python model_predictive_control/mpc_example.py`.

## Tutorials
A set of notebooks is avaialble to describe the models considered for modeling autonomous car racing with different
 nonlinear control algorithms in `ilqc.ipynb`. To run them, simply type jupyter notebook from the root of the folder
  after having activated the environment.

## Reproducing experiments
The experiments presented in [ilqc_algos](https://arxiv.org/abs/2207.06362) can be reproduced by running `python
 finite_horizon_control/compa_algos.py`. Output figures are saved in the folder `finite_horizon_control`. For ease of
  test, the results have been saved in advance. If you prefer to rerun all experiments simply erase the folder results.

## Contact
You can report issues and ask questions in the repository's issues page. If you choose to send an email instead, please direct it to Vincent Roulet at vroulet@uw.edu and include [ilqc] in the subject line.


#### Authors
[Vincent Roulet](http://faculty.washington.edu/vroulet/)  
[Siddhartha Srinivasa](https://goodrobot.ai/)  
[Maryam Fazel](https://people.ece.uw.edu/fazel_maryam/)  
[Zaid Harchaoui](http://faculty.washington.edu/zaid/)  


#### License
This code has a GPLv3 license.
