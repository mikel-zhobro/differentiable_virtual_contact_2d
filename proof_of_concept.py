# This is a single file proof of concept for smoothing penallty based dynamics with virtual contacts

# fc(q,qdot) = max(0, d(q))*( -kn d(q) + kd d(q)*ddot(qdot) ) s*n_vec
# ft = -min(kt||tdot||), mu *||fc||) s tdot_vec/||tdot_vec||

import time
import numpy as np
import matplotlib.pyplot as plt
import torch

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
            gradients = torch.autograd.grad(y[:, i], x, torch.ones_like(y[:, i]), create_graph=True)[0]
            jacobian[:, i, :] = gradients
    else:  # Non-batched inputs
        output_size = y.size(0)
        input_size = x.size(-1)
        jacobian = torch.zeros(output_size, input_size)

        for i in range(output_size):
            gradients = torch.autograd.grad(y[i], x, torch.ones_like(y[i]), create_graph=True, retain_graph=True)[0]
            jacobian[i, :] = gradients

    return jacobian

def newton(x, func, func_with_derivatives, verbose = True):
    # steepest gradient descent with line search
    # --------------------
    # idea, not related: can use levenbarg-marquardt algorithm for least squares which changes between
    # steepest gradient descent(when error is increasing) and gauss newton's method(when making progress)
    # gauss newton method: is a special case of newton's method where the hessian is approximated by the jacobian
    #                      (linearize function before taking the norm and than make the gradient of the norm 0)
    t_newton_start = time.time()

    _ndof_r = len(x)
    if _ndof_r == 0:
        return

    tol = Sim._solver_options._tol
    MaxIter_Newton = max(20 * _ndof_r, Sim._solver_options._MaxIter_Newton)
    MaxIter_LS = Sim._solver_options._MaxIter_LS

    MaxIter_LS_Fail_Strike = 10

    success_newton = False
    g_last = 1e5
    fail_strike = 0
    g_his = []

    for iter_newton in range(MaxIter_Newton):
        g, H = func_with_derivatives(x)

        dx = np.linalg.solve(H, -g)

        g_new = func(x + dx)

        gnorm = np.linalg.norm(g)
        alpha = 1.0
        success_ls = False

        for trial in range(MaxIter_LS):
            g_new_trial = func(x + alpha * dx)

            if np.linalg.norm(g_new_trial) < gnorm:
                success_ls = True
                break

            alpha *= 0.5

        if success_ls:
            fail_strike = 0
        else:
            fail_strike += 1
            if fail_strike >= MaxIter_LS_Fail_Strike:
                break

        x += alpha * dx

        if np.linalg.norm(g_new) < tol:
            success_newton = True
            break

        g_last = np.linalg.norm(g_new)
        g_his.append(g_last)

    if not success_newton and (g_last > 1e-5 or np.isnan(g_last)):
        if verbose:
            print("Newton method did not converge. g =", g_last)

    # print(f"{iter_newton} newton steps")
    Sim._time_report._time_solver += time.time() - t_newton_start
    return x


# Have to be performed in batch
# Body state, position: q, qdot, orientation: r rdot
# x: contact positions in body frame -> d: penetration depths, ddots: penetration velocities, n: normal vectors, t: tangent vectors
# Torque += Radius x F, Force += F

class Sim:
    class _solver_options:
        _tol = 1e-9
        _MaxIter_Newton = 100
        _MaxIter_LS = 20
    class _time_report:
        _time_solver = 0.
        _time_contact = 0.
        _time_collision = 0.
        _time_dynamics = 0.
        _time_integrator = 0.
        _time_total = 0.
    class _params:
        dt = 1./100  # s
        gravity = 9.81 # m/s^2
        mu = 0.1
        kn = 1e5
        kt = 1e3
        kd = 3e1
        damping = False
        smoothed_penalty = False
        smoothed_alpha = 0.01
    class _object:
        radius = 0.1 # m
        m = 0.1      # kg
        ndof_r = 2
        gravity = torch.zeros(ndof_r)
        q0 = torch.zeros_like(gravity)
        qdot0 = torch.zeros_like(gravity)

        n = torch.zeros_like(qdot0)
        n[-1] = 1.
        t = torch.zeros_like(qdot0)
        t[0] = 1.

        @classmethod
        def comp_penetration(cls, q: torch.Tensor, qdot: torch.Tensor): # circle
            d = cls.radius - q[-1]
            ddot = qdot[-1]
            return d, ddot

        @classmethod
        def contact_penallty(cls, d: torch.Tensor, ddot: torch.Tensor):
            fc = Sim._params.kn
            if Sim._params.damping:
                fc -= Sim._params.kd * ddot

            if Sim._params.smoothed_penalty:
                a = Sim._params.smoothed_alpha
                fc *= torch.where(d >= 0, d+a, a*torch.exp(d/a))
            else:
                fc *= torch.max(torch.zeros(1), d)
            return fc


    def __init__(self,):
        Sim._object.gravity[-1] = Sim._params.gravity

        self.u = torch.zeros(Sim._object.ndof_r)
        self.q = Sim._object.q0
        self.qdot = Sim._object.qdot0

    def reset(self):
        self.q = Sim._object.q0
        self.qdot = Sim._object.qdot0

    def get_qdot(self, q, qprev):
        return (q - qprev) / Sim._params.dt

    def get_penalty_force(self, q: torch.Tensor, qdot: torch.Tensor):
        d, ddot = Sim._object.comp_penetration(q, qdot)
        fc = Sim._object.contact_penallty(d, ddot)
        return fc

    def evaluate_g(self, q1: torch.Tensor):
        qdot1 = self.get_qdot(q1, self.q)
        p_force = self.get_penalty_force(q1, qdot1)
        forces = p_force - Sim._object.gravity * Sim._object.m + self.u
        g = Sim._object.m * (q1 - self.q - Sim._params.dt * self.qdot) - Sim._params.dt**2 * forces
        return g

    def evaluate_g_with_derv(self, q: torch.Tensor):
        _q = q.clone().requires_grad_(True)
        g = self.evaluate_g(_q)
        return g.detach().numpy(), compute_jacobian(g, _q).detach().numpy()

    def integration_BDF1(self, dt, q:torch.Tensor, qdot:torch.Tensor, u: torch.Tensor):
        _q0 = q
        _qdot0 = qdot

        q1 = _q0 + dt * _qdot0

        self.u = u
        # solve for new q1
        q1 = torch.tensor(newton(q1, self.evaluate_g, self.evaluate_g_with_derv, verbose=True))
        q1dot =  self.get_qdot(q1, _q0)
        return q1, q1dot

    def get_derivatives(self, q1:torch.Tensor, qprev:torch.Tensor, qdotprev:torch.Tensor, u:torch.Tensor):
        _, qgrad = self.evaluate_g_with_derv(q1)


        self.qdot = qdotprev.clone().requires_grad_(True)
        self.q = qprev.clone().requires_grad_(True)
        self.u = u.clone().requires_grad_(True)
        g = self.evaluate_g(q1)
        # g.backward()
        # assert self.u.grad is not None
        # uprev_partial = self.u.grad.numpy()
        uprev_partial = compute_jacobian(g, self.u).numpy()

        # self.q = qprev.clone().requires_grad_(True)
        # g = self.evaluate_g(q1)
        # g.backward()
        # assert self.q.grad is not None
        # qprev_partial = self.q.grad.numpy()
        qprev_partial = compute_jacobian(g, self.q).numpy()

        # self.qdot = qdotprev.clone().requires_grad_(True)
        # g = self.evaluate_g(q1)
        # g.backward()
        # assert self.qdot.grad is not None
        # qdotprev_partial = self.qdot.grad.numpy()
        qdotprev_partial = compute_jacobian(g, self.qdot).numpy()

        dqdu = -uprev_partial / qgrad
        dqdqprev = -qprev_partial / qgrad
        dqdqdotprev = -qdotprev_partial / qgrad

        return dqdu, dqdqprev, dqdqdotprev


    def step(self, u: torch.Tensor):
        self.q, self.qdot = self.integration_BDF1(Sim._params.dt, self.q, self.qdot, u)
        return self.q, self.qdot

    def backward_gradients(self,
                           q1s: torch.Tensor,
                           qprevs: torch.Tensor,
                           qdotprevs: torch.Tensor,
                           us: torch.Tensor):

        dqdus, dqdqprevs, dqdqdotprevs = [], [], []
        for q1, qprev, qdotprev, u in zip(q1s, qprevs, qdotprevs, us):
            dqdu_, dqdqprev_, dqdqdotprev_ = self.get_derivatives(q1, qprev, qdotprev, u)
            dqdus.append(dqdu_)
            dqdqprevs.append(dqdqprev_)
            dqdqdotprevs.append(dqdqdotprev_)
        return np.stack(dqdus), np.stack(dqdqprevs), np.stack(dqdqdotprevs)

    def forward(self, inputs: torch.Tensor):
        self.reset()
        qs, qdots = [self.q], [self.qdot]
        for u in inputs:
            q, qdot = self.step(u)
            qs.append(q)
            qdots.append(qdot)

        return torch.stack(qs), torch.stack(qdots)



def main():
    Sim._params.dt = 1./100  # s
    Sim._params.gravity = 9.81 # m/s^2
    Sim._params.kn = 1e3
    Sim._params.kd = 5e1
    Sim._params.smoothed_alpha = 0.001
    Sim._params.damping = False

    Sim._object.radius = 0.1 # m
    Sim._object.m = 0.1      # kg
    Sim._object.q0[-1] = Sim._object.radius + 0.
    Sim._object.qdot0 = 0. * Sim._object.qdot0

    sim = Sim()

    qs, qdots = [], []

    N = 130
    inputs = torch.zeros(N, Sim._object.ndof_r)
    # inputs[:,0] = 0.4
    inputs[:,-1] = torch.linspace(0, 15, N)* Sim._object.m

    Sim._params.smoothed_penalty = False
    qs, qdots = sim.forward(inputs)
    dqdus, dqdqprevs, dqdqdotprevs = sim.backward_gradients(qs[1:], qs[:-1], qdots[:-1], inputs)

   # create a 3 x 3 subplot
    fig, axs = plt.subplots(4, 2, figsize=(15, 12), dpi=80, sharex=True)
    ax_u = axs[0]
    ax_q = axs[1]
    ax_qdot = axs[2]
    ax_du = axs[3]

    label_qs = ['x', 'z']
    for i in range(Sim._object.ndof_r):
        ax_u[i].plot(inputs[:,i] / (Sim._object.m if i==1 else 1.), label='u_'+label_qs[i])
        ax_q[i].plot(qs[:, i], label=label_qs[i] )
        ax_qdot[i].plot(qdots[:, i], label= label_qs[i]+'_dot')
        ax_du[i].plot(np.stack([t[i,i] for t in dqdus]), label= 'd'+label_qs[i]+'_du')

    runs = {}
    Sim._params.smoothed_penalty = True
    for alpha in [0.0005, 0.0008, 0.001]:
        Sim._params.smoothed_alpha = alpha
        qs_smoothed, qdots_smoothed = sim.forward(inputs)
        dqdus_smoothed, dqdqprevs_smoothed, dqdqdotprevs_smoothed = sim.backward_gradients(
            qs_smoothed[1:], qs_smoothed[:-1], qdots_smoothed[:-1], inputs)

        for i in range(Sim._object.ndof_r):
            ax_q[i].plot(qs_smoothed[:, i], label = label_qs[i] + fr', $\alpha={alpha}$')
            ax_qdot[i].plot(qdots_smoothed[:, i], label = label_qs[i] + fr'_dot, $\alpha={alpha}$')
            ax_du[i].plot(np.stack([t[i,i] for t in dqdus_smoothed]), label= 'd'+label_qs[i] + fr'_du, $\alpha={alpha}$')

    [ax.legend() for ax in axs.flatten()]
    plt.show()

##############################################################################

if __name__ == "__main__":
    main()



