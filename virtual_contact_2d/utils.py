import torch
import numpy as np
import matplotlib.pyplot as plt
from sdf import d2, tu, cartesian_product
import matplotlib.patches as patches

def get_skew(r: torch.Tensor): # same as hat
    # Computes skew matrix of shape (N, 3 ,3) from a vector r of shape (N, 3)
    my_skew = r.new_zeros((r.shape[0], 3, 3))
    my_skew[:, 0, 1] = -r[:, 2]
    my_skew[:, 1, 0] =  r[:, 2]
    my_skew[:, 0, 2] =  r[:, 1]
    my_skew[:, 2, 0] = -r[:, 1]
    my_skew[:, 1, 2] = -r[:, 0]
    my_skew[:, 2, 1] =  r[:, 0]
    return my_skew

def rot(angle: torch.Tensor):
    return torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                        [torch.sin(angle),  torch.cos(angle)]])

def to_body_frame(x_w: torch.Tensor, q: torch.Tensor):
    return (x_w - q[:2]).matmul(rot(q[2]))  # R^T(x - t)

def compute_jacobian(y, x):
    """
    Compute the Jacobian of y with respect to x.

    Args:
        y (torch.Tensor): Tensor representing the output of the function y(x).
            Shape: (batch_size, output_size) or (output_size,) for non-batched.
        x (torch.Tensor): Tensor representing the input variables x.
            Shape: (batch_size, input_size) or (input_size,) for non-batched.

    Returns:
        jacobian (torch.Tensor): Computed Jacobian matrix.
            Shape: (batch_size, output_size, input_size) or (output_size, input_size) for non-batched.
    """
    if y.dim() > 1:  # Batched inputs
        batch_size, output_size = y.size()
        input_size = x.size(-1)
        jacobian = torch.zeros(batch_size, output_size, input_size)

        for i in range(output_size):
            gradients = torch.autograd.grad(y[:, i], x, torch.ones_like(y[:, i]), retain_graph=True, allow_unused=True)[0]
            jacobian[:, i, :] = gradients
    else:  # Non-batched inputs
        output_size = y.size(0)
        input_size = x.size(-1)
        jacobian = torch.zeros(output_size, input_size)

        for i in range(output_size):
            gradients = torch.autograd.grad(y[i], x, torch.ones_like(y[i]), retain_graph=True, allow_unused=True)[0]
            jacobian[i, :] = gradients

    return jacobian

def create_gif(record_folder, output_file):
    import os
    from PIL import Image
    images = []
    for filename in os.listdir(record_folder):
        images.append(Image.open(os.path.join(record_folder, filename)))
    images[0].save(output_file,save_all=True,append_images=images[1:],duration=11,loop=0)

    [os.unlink(f) for f in os.scandir(record_folder)]

def estimate_bounds2_0(sdf, x_min=-10., x_max=10, verbose=True):
    """
    ------ Estimate bounds of the sdf (dimension agnostic) ------
    starts with a small cube and expands it until sdf is contained
    it does that by expanding the bounds using sdf values from previous sampled bounds
    """
    n = sdf.dim
    s = 16
    c0 = np.zeros(n) - x_max
    c1 = np.zeros(n) + x_max

    Cs = [np.linspace(x0, x1, s) for x0, x1 in zip(c0, c1)] # linspace for each dimension - (s,s,s) cubes [dim, s]
    d = np.abs(np.array([X[1] - X[0] for X in Cs])) # the diagonal of one of the (s,s,s) cubes

    P = tu.to_torch(cartesian_product(*Cs)).requires_grad_() # shape: (s**n, n) where n can be 2 or 3 -- kind of meshgrid
    volume: torch.Tensor = sdf(P).reshape(tuple([len(X) for X in Cs])) # (s, s, s) or (s, s)
    volume.sum().backward()
    assert P.grad is not None
    vec_2_surface = np.abs(P.grad.numpy().reshape(tuple([len(X) for X in Cs]+[-1])) * volume.detach().numpy()[...,None]) # (s**n, n) * (s, s, s) = (s, s, s, n)

    where = np.argwhere(np.logical_and(vec_2_surface[:,:,0] <= d[0], vec_2_surface[:,:,1] <= d[1]))

    c1 = c0 + where.max(axis=0) * d + d / 2
    c0 = c0 + where.min(axis=0) * d - d / 2

    print(f"c0: {c0} - c1: {c1}, 'd': {d} {where.min(axis=0)}, {where.max(axis=0)}")
    return np.clip(c0-d-5, x_min, x_max), np.clip(c1+d+5, x_min, x_max)

def vis(sdf1: d2.SDF2, sdf2: d2.SDF2, xs: torch.Tensor, x_min, x_max, u_arrow=None, penalty_arrow=None, granurality=200, path=None, title=""):
    """Visualize the 2D SDFs and the contact points.

    Args:
        sdf1/2 (d2.SDF2): _description_
        xs (torch.Tensor): Virtual contact locations
        x_min/max (np.ndarray): Considered area
        granurality (int, optional): Granularity of the considered area for visualization.
    """
    sdf = sdf1 | sdf2
    cmap = plt.get_cmap('seismic')
    VMAX = 20

    extent = x_min[0], x_max[0], x_min[1], x_max[1]
    Xs = np.linspace(x_min[0], x_max[0], granurality)
    Ys = np.linspace(x_min[1], x_max[1], granurality)
    shape = (granurality, granurality)
    P = tu.to_torch(cartesian_product(Xs,  Ys))

    sdf_vals = sdf(P).detach().cpu().numpy().reshape(shape)
    sdf_vals1 = sdf1(P).detach().cpu().numpy().reshape(shape)
    sdf_vals2 = sdf2(P).detach().cpu().numpy().reshape(shape)

    plt.figure()
    plt.imshow(sdf_vals.T, cmap, origin='lower', extent=extent, vmin=-VMAX, vmax=VMAX)
    plt.contour(P[:,0].reshape(shape), P[:,1].reshape(shape), sdf_vals1, [0])
    plt.contour(P[:,0].reshape(shape), P[:,1].reshape(shape), sdf_vals2, [0])
    cpoints = plt.scatter(xs[:, 0].detach().cpu().numpy(), xs[:, 1].detach().cpu().numpy(), c='k', marker='x', s=42, label=f'the {xs.shape[0]} contact points')

    plt.autoscale(False)
    handles = [cpoints]
    if u_arrow is not None:
        plt.arrow(u_arrow[0][0], u_arrow[0][1], u_arrow[1][0], u_arrow[1][1], head_width=0.3, head_length=0.4, fc='blue', ec='blue')
        legend_arrow = patches.Patch(color='blue', label='Input Force')
        handles.append(legend_arrow)
    if penalty_arrow is not None and u_arrow is not None:
        plt.arrow(u_arrow[0][0], u_arrow[0][1], penalty_arrow[0], penalty_arrow[1], head_width=0.3, head_length=0.4, fc='red', ec='red')
        legend_arrow = patches.Patch(color='red', label='Penallty Force')
        handles.append(legend_arrow)


    plt.suptitle(title+"2D-SDF contact detection")
    plt.legend(handles=handles)

# Add the custom legend entry to the legend
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.savefig('image.png')
    plt.close()

def stein_sgd(lgrad_x, x):
    """
    :param lgrad_x: local gradient of x (shape (N, 3))
    :param x: input (shape (N, 3))
    """
    N = x.shape[0]
    y = x  # fixed
    # compute pairwise difference between x and y (x is input, y is fixed)
    pairwise_difference = x[:, None, :] - y[None, :, :] # shape (Nx, Ny, 3)

    # compute pairwise distance between x and y
    pairwise_distance = torch.norm(pairwise_difference, dim=-1, keepdim=True)**2  # Nx, Ny, 1

    # compute RBF kernel for each distance
    # K(x, y) = exp(-0.5||x-y||^2 / h)
    # bandwidth = torch.median(pairwise_distance)
    # bandwidth = torch.sqrt(0.5 * bandwidth / np.log(N+1))**2 # median heuristic
    bandwidth = torch.median(pairwise_distance)**2 / np.log(N) # median heuristic
    # h = 2 * h**2we

    k = torch.exp(-0.5 * pairwise_distance / bandwidth) # shape (Nx, Ny, 1)
    # k = torch.exp(-d / h) # shape (Nx, Ny, 1)

    # first term
    smoothed_grad = torch.sum(k * lgrad_x[None, :, :], dim=1) # shape (Nx, 3)

    # second term
    # kernel gradients(analytical derivative of kernel w.r.t. x)
    # k_grad = 2./h * diff * k # shape (Nx, Ny, 3)
    k_grad = 1./bandwidth * pairwise_difference * k # shape (Nx, Ny, 3)
    repulsive_grad = torch.sum(k_grad, dim=1) # shape (Nx, 3)

    # update x
    return (smoothed_grad + repulsive_grad) / N

def find_contact(sdf1: d2.SDF2, sdf2: d2.SDF2,  xs_int: torch.Tensor, max_iter,
                q1: torch.Tensor=None, q2: torch.Tensor=None, lr=1e-1):
    # intialize contact points
    points_w = xs_int.detach().clone().requires_grad_(True)
    opt = torch.optim.SGD([points_w], lr=lr)

    i = 0
    while True:
        opt.zero_grad()
        pts_b1 = to_body_frame(points_w, q1) if q1 is not None else points_w
        pts_b2 = to_body_frame(points_w, q2) if q2 is not None else points_w
        pts_b1.retain_grad()
        pts_b2.retain_grad()
        sdfs1 = sdf1(pts_b1)
        sdfs2 = sdf2(pts_b2)
        sdf_vec = (sdfs1 - sdfs2)**2 + sdfs1 + sdfs2

        loss = torch.sum(sdf_vec)
        loss.backward()

        # points_w.grad = stein_sgd(points_w.grad, points_w.detach())

        opt.step()

        if i>= max_iter:
            break
        i += 1

    x_c_w = points_w.detach().clone()
    x_c_b1 = pts_b1.detach().clone()
    x_c_b2 = pts_b2.detach().clone()

    n_b1 = pts_b1.grad.numpy()
    n_b2 = pts_b2.grad.numpy()
    # d1 = -sdfs1.detach().clone()
    # d2 = -sdfs2.detach().clone()

    pts_b1 = to_body_frame(points_w.detach().clone(), q1) if q1 is not None else points_w.detach().clone()
    pts_b2 = to_body_frame(points_w.detach().clone(), q2) if q2 is not None else points_w.detach().clone()
    # pts_b1.retain_grad()
    # pts_b2.retain_grad()
    sdfs1 = sdf1(pts_b1)
    sdfs2 = sdf2(pts_b2)


    d1 = -sdfs1
    d2 = -sdfs2

    # n_b1 = torch.autograd.grad(sdfs1, pts_b1, torch.ones_like(sdfs1), is_grads_batched=False)[0]
    # n_b2 = torch.autograd.grad(sdfs2, pts_b2, torch.ones_like(sdfs2), is_grads_batched=False)[0]

    return x_c_w, x_c_b1, x_c_b2, d1, d2, n_b1, n_b2

