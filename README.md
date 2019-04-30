Iterative Linear Quadratic Control Toolbox
=============================================

This code provides an implementation of regularized discrete control algorithms on standard synthetic benchmarks implemented in PyTorch.

Those methods are developed in the paper:

V. Roulet S. Srinivasa, D. Drusvyatskiy and Z. Harchaoui. Iterative Linearized Control: Stable Algorithms and Complexity Guarantees. Proceedings of the 36th International Conference on Machine Learning-Volume 72. 2019

If you use this code please cite the paper using the bibtex reference below.


```
@article{roulet2019iterative,
  title={Iterative Linearized Control: Stable Algorithms and Complexity Guarantees},
  author={Roulet, Vincent and Srinivasa, Siddharta and Drusvyatskiy, Dmitry and Harchaoui, Zaid},
  journal={Proceedings of the 36th International Conference on Machine Learning-Volume 72},
  year={2019}
}
```

Introduction
-----------------
Given a robot whose dynamics are described by continuous mechanical equations, control problems aim to make the robot move to a given target position in a finite amount of time within a given precision. The user has access to some control variables that define the dynamics of the robot. We consider here the case where the whole trajectory must be controlled in advance i.e. in an open-loop setting.

In this setting, the dynamical continuous equations are discretized along time, which lead to a discrete control problem. The latter is composed of t steps of the dynamics where each step is parametrized by the current state and a control variable. The optimality of the whole set of control variables is measured by a cost on the trajectory it produces and some penalties on the choice of the control variables.

From an optimization point of view, the problem is a composite optimization problem, where first order oracles are given by linearizing the dynamics at each step and solve the resulting subproblems. For more details and a mathematical presentation of the problem see the paper.


Examples
-------
This code implements standard Iterative Linear Quadratic Regulator (ILQR), regularized ILQR and accelerated regularized ILQR on two standard benchmarks: the inverted pendulum and the two-links arm model. Complete formulations can be found in the paper. Each algorithm iteratively linearizes the dynamics along the current trajectory and solve the resulting linear quadratic control. Here the subproblems are solved using their dual formulation with a conjugate gradient that makes calls to the automatic-differentiation procedure of Pytorch.

To run the inverted pendulum experiment for a discretization grid in time of length 100 with the objective of swinging up the pendulum, use from the main folder

`python exp_paper.py --ctrl_setting inverse_pendulum --horizon 100 --target_goal swing_up`

To run the two-links arm experiment for a discretization grid in time of length 100 with the objective of reaching a random target, use from the main folder

`python exp_paper.py --ctrl_setting two_links_arm --horizon 100 --target_goal cartesian_random_target`

A burning-phase of 5 iterations is done for for the regularized and accelerated regularized algorithms to get the best fixed step-sizes used for the remaining iterations. For ILQR a burning-phase is also performed for the initial step-size, a line-search is then performed at each step to get the next point following an Armijo rule.

Installation
-----------------
This code was written in Python 3.6 with PyTorch version 1.0.0. A conda environment file is provided in `ilqc.yml` and can be installed by using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

The primary dependencies are pytorch, matplotlib, seaborn, pandas. The remainder of the dependencies are standard and e.g., come pre-installed with Anaconda. The code has not been tested on Windows operating systems.


Contact
-----------------
You can report issues and ask questions in the repository's issues page. If you choose to send an email instead, please direct it to Vincent Roulet at vroulet@uw.edu and include [ilqc] in the subject line.

Authors
-----------------
[Vincent Roulet](http://faculty.washington.edu/vroulet/)  

[Dmitriy Drusvyatskiy](https://sites.math.washington.edu/~ddrusv/)

[Siddhartha Srinivasa](https://goodrobot.ai/)

[Zaid Harchaoui](http://faculty.washington.edu/zaid/)  


License
-----------------
This code has a GPLv3 license.
