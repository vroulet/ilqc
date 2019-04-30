from copy import deepcopy
import torch
from src.model.utils_model import conj_grad

print('Test implementation of conjugate gradients')
torch.set_default_tensor_type(torch.DoubleTensor)
d = 1000
A = torch.rand(d, d)
A = torch.mm(A, A.t())
b = torch.rand(d)
reg = 0.01

max_iter_conj_grad_factor = 3

print('Test solving the quadratic problem in its primal form')
bb = b.view((d, 1))
true_sol, _ = torch.gesv(bb, A + reg*torch.eye(d))


def func(x):
    return 0.5*torch.sum(x*torch.mv(A, x)) - torch.sum(b*x) + 0.5*reg*torch.sum(x*x)


def grad(x):
    return torch.mv(A, x) - b + reg*x


x0 = torch.rand(d)
sol_home = conj_grad(
    x0, grad, max_iter=max_iter_conj_grad_factor*d, accuracy=1e-30)

print('Difference btw solution found by inversion and conj_grad solution')
print('Norm {}'.format(torch.norm(true_sol.view(-1) - sol_home)))
print('All-close ? {}'.format(sol_home.allclose(true_sol.view(-1))))

# Test conj_grad by dual
print('\nTest solving the quadratic problem by computing dual solution')
L = torch.cholesky(A, upper=False)
Linvb = torch.trtrs(b, L, upper=False)[0].view(-1)


def func_dual(z):
    return torch.sum(Linvb*z) + 0.5*torch.sum(z*z) + 0.5/reg * torch.sum(torch.mv(L, z)*torch.mv(L, z))


def grad_dual(z):
    aux = deepcopy(z)
    aux.requires_grad = True
    out = func_dual(aux)
    grad = torch.autograd.grad(out, aux)[0]
    return grad


z0 = torch.rand(d)
dual_sol = conj_grad(
    z0, grad_dual, max_iter=max_iter_conj_grad_factor*d, accuracy=1e-30)

primal_sol = -1/reg*torch.mv(L, dual_sol)

print('Difference btw solution found by inversion and dual conj_grad solution')
print('Norm {}'.format(torch.norm(true_sol.view(-1) - primal_sol)))
print('All-close ? {}'.format(primal_sol.allclose(true_sol.view(-1))))
