# Script to implement stein sgd which fits a gaussian mixture via particles.


import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns  # for nicer graphics


def stein_sgd(lgrad_x, x):
    """
    :param lgrad_x: local gradient of x (shape (N, 3))
    :param x: input (shape (N, 3))
    """
    N = x.shape[0]
    y = x  # fixed
    # compute pairwise difference between x and y (x is input, y is fixed)
    diff = x[:, None, :] - y[None, :, :] # shape (Nx, Ny, 3)

    # compute pairwise distance between x and y
    d = torch.norm(diff, dim=-1, keepdim=True)**2  # Nx, Ny, 1

    # compute RBF kernel for each distance
    h = torch.median(d)
    h = torch.sqrt(0.5 * h / np.log(N+1)) # median heuristic

    k = torch.exp(-d / h**2 /2) # shape (Nx, Ny, 1)
    # k = torch.exp(-d / h) # shape (Nx, Ny, 1)

    # first term
    smoothed_grad = torch.sum(k * lgrad_x[None, :, :], dim=1) # shape (Nx, 3)

    # second term
    # kernel gradients(analytical derivative of kernel w.r.t. x)
    # k_grad = 2./h * diff * k # shape (Nx, Ny, 3)
    k_grad = 1./h**2 * diff * k # shape (Nx, Ny, 3)
    repulsive_grad = torch.sum(k_grad, dim=1) # shape (Nx, 3)

    # update x
    return (smoothed_grad + repulsive_grad) / N


# Example gaussian mixture
def gaussian_pdf(x, mu, sigma):
    return 1. / (np.sqrt(2. * torch.pi) * sigma) * torch.exp(-(x - mu)**2 / (2. * sigma**2))

def desired_p(x):
    return 1./3. * gaussian_pdf(x, -2., 1) + 2./3. * gaussian_pdf(x, 2., 1)


N = 120
x0_init = torch.normal(mean=-2*torch.ones((N, 1)), std=1.*torch.ones((N, 1)))




# optimize with adagrad with momentum
alpha = 0.9
stepsize = 5e-3
fudge_factor = 1e-6

x = x0_init.clone().detach()
historical_grad = torch.zeros_like(x)
for i in range(2000):
    x.requires_grad_(True)
    lgrad_x = torch.autograd.grad(torch.log(desired_p(x)).sum(), x)[0]
    x.requires_grad_(False)
    stein_sgd_grad_x = stein_sgd(lgrad_x, x)

    # adagrad
    # adj_grad = stein_sgd_grad_x
    if i == 0:
        historical_grad = historical_grad + stein_sgd_grad_x ** 2
    else:
        historical_grad = alpha * historical_grad + (1 - alpha) * (stein_sgd_grad_x ** 2)
    adj_grad = torch.divide(stein_sgd_grad_x, fudge_factor+torch.sqrt(historical_grad))

    x = x + stepsize * adj_grad


    # plot
    if (i % 10) == 0:
        plt.clf()
        plt.hist(x.detach().numpy(), bins=100, density=True)
        plt.plot(torch.tensor(np.linspace(-10, 10, 100)), desired_p(torch.tensor(np.linspace(-10, 10, 100))), 'r')
        sns.kdeplot(x.squeeze())

        plt.xlim(-10, 10)
        plt.ylim(0, 1.)
        plt.title('iter: {}'.format(i))
        plt.pause(0.1)