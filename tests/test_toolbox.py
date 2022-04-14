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