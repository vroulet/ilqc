import torch


def auto_multi_grad(output, input, create_graph=False):
    """
    Compute gradient (i.e. the transpose of the jacobian) of a multidimensional function given by output on input
    """
    # todo change it into autograd that computes classical gradient
    # if the output is a scalar or gradient of a multivariate function if the output is a function
    output_basis = torch.eye(len(output))
    jac = torch.stack([torch.autograd.grad(output, input, output_coord, retain_graph=True, create_graph=create_graph)[0]
                       for output_coord in output_basis])
    return jac.t()


def auto_multi_hess(output, input):
    if output.grad_fn is None:
        hess = torch.zeros(input.shape[0], output.shape[0], output.shape[1])
    else:
        hess = torch.stack([auto_multi_grad(output_coord, input) for output_coord in output])
        hess = torch.transpose(hess, 0, 1)
    return hess


def define_smooth_relu(eps):
    x1 = -eps
    y1 = torch.tensor(0.)
    x2 = eps
    y2 = eps
    k1 = torch.tensor(0.)
    k2 = torch.tensor(1.)
    a = k1*(x2-x1) - (y2-y1)
    b = -k2*(x2-x1) + (y2-y1)

    def smooth_relu(x):
        tx = (x-x1)/(x2-x1)
        out = (x > eps)*x + (x >= -eps)*(x <= eps)*((1 - tx)*y1 + tx*y2 + tx*(1 - tx)*((1 - tx)*a + tx*b))
        return out
    return smooth_relu


smooth_relu = define_smooth_relu(1e-6)


def smooth_min(x, a):
    return -smooth_relu(-x + a) + a



