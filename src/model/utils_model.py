from copy import deepcopy
import torch


def conj_grad(x0, grad, max_iter=100, accuracy=1e-12):
    """ Solve quadratic problem
    min_x 0.5 x'Ax - x'b (1)
    by a conjugate gradient method
    :param x0 (torch.Tensor) initial point of the method
    :param grad (Callable) outputs the gradient of the quadratic function
    :return
        x: (torch.Tensor) solution
    """
    x = x0
    r = -grad(x)
    b = -grad(torch.zeros_like(x))
    p = r
    for i in range(max_iter):
        Ap = grad(p) + b
        alpha = torch.norm(r)**2/(torch.sum(p*Ap))
        x = x + alpha * p
        rnew = r - alpha * Ap
        # print('iteration {0}, residual norm {1}'.format(i, torch.norm(rnew)))
        if torch.norm(rnew) < accuracy:
            # print('accuracy of conj_grad reached')
            break
        beta = torch.norm(rnew)**2/torch.norm(r)**2
        p = rnew + beta*p
        r = rnew
    return x


def chol_problem(final_state, target_cost_func):
    """
    Gets a quadratic model on the target cost as
    h(y) = h(x) + <∇h(x), y-x> + 0.5 <y-x, H (y-x)>
         = cste + 0.5 || L^T y - L^(-1) ∇h(x)||^2
    where, dennoting ∇h^2(x) = U D U^T, we denote H = U |D| U^T (absolute values of the eigenvalues of H are taken)
        and H = LL^T (cholesky decomposition of H)
    :param final_state: (torch.Tensor) last state on which the approximation of the cost is taken (x above)
    :param target_cost_func: (torch.nn.Module) Cost on the last state
    :return:
        chol_hess: (torch.Tensor) L above
        chol_hess_inv_grad: (torch.Tensor) L^(-1) ∇h(x) above
    """
    aux = deepcopy(final_state.data)
    aux.requires_grad = True

    target_cost = target_cost_func(aux)

    grad = torch.autograd.grad(target_cost, aux, create_graph=True)[0]
    hess = auto_jac(grad, aux)

    (lam, U) = torch.eig(hess, eigenvectors=True)
    lam = torch.abs(lam[:, 0])
    hess = torch.mm(U, torch.mm(torch.diag(lam), U.t()))

    chol_hess = torch.cholesky(hess, upper=False)
    chol_hess_inv_grad = torch.trtrs(grad, chol_hess, upper=False)[0].view(-1)
    return chol_hess, chol_hess_inv_grad


def auto_jac(output, input, create_graph=False):
    """
    Compute jacobian of a function given by output on input
    """
    output_basis = torch.eye(len(output))
    jac = torch.stack([torch.autograd.grad(output, input, output_coord, retain_graph=True, create_graph=create_graph)[0]
                       for output_coord in output_basis])
    return jac

# def conj_grad(x0, Ax, b, max_iter=100, accuracy=1e-3):
#     """ Solve quadratic problem
#     min_x 0.5 x'Ax - x'b (1)
#     by a conjugate gradient method
#     :param x0 (torch.Tensor) initial point of the method
#     :param Ax (Callable) function that given x returns Ax in (1)
#     :param b (torch.Tensor) linear term in (1)
#     """
#     x = x0
#     r = b - Ax(x)
#     p = r
#     for i in range(max_iter):
#         alpha = torch.norm(r)**2/(p.dot(Ax(p)))
#         x = x + alpha * p
#         rnew = r - alpha * Ax(p)
#         print('residual norm {}'.format(torch.norm(rnew)))
#         if torch.norm(rnew) < accuracy:
#             break
#         beta = torch.norm(rnew)**2/torch.norm(r)**2
#         p = rnew + beta*p
#         r = rnew
#     return x
