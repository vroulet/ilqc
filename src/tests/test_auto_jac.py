import torch
from src.model.utils_model import auto_jac

torch.set_default_tensor_type(torch.DoubleTensor)
d = 5
A = torch.rand(d, d)
A = torch.mm(A, A.t())
b = torch.rand(d)


def func(x):
    return 0.5*x.dot(torch.mv(A, x)) - x.dot(b)


def grad(x):
    return torch.mv(A, x) - b


print('Test computations of the hessian by autograd')
x0 = torch.rand(d, requires_grad=True)
func_x = func(x0)
func_x.backward(create_graph=True)
grad_x = x0.grad
hess = auto_jac(grad_x, x0)

print('Are computations good ? {}'.format(hess.allclose(A)))
