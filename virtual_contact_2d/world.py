import torch
import numpy as np
from tqdm import tqdm

from . import utils
from .virtual_contact import BodyPlaneVirtualContacts
from .solver import NewtonSolver


class World:
    def __init__(self, virtual_contact: BodyPlaneVirtualContacts,
                 dt=0.01, gravity=9.81,
                 recompute_contacts_durint_optim=False, update_contacts_iters=1) -> None:
        # Sim params
        self.dt = dt
        self.gravity = torch.tensor([0., -gravity, 0.]) # m/s^2

        # Virutal Contact params
        self.recompute_contacts_durint_optim = recompute_contacts_durint_optim
        self.update_contacts_iters = update_contacts_iters

        # Solver
        self.newton = NewtonSolver(verbose=True)
        # Virtual Contacts
        self.virtual_contact = virtual_contact

        # This input is used only internally, not meant to be used from outside
        self._u = torch.zeros(3) # fx, fy, tau

    # Properties
    # --------------------------------
    @property
    def body(self):
        return self.virtual_contact.body
    @property
    def q(self):
        return self.virtual_contact.body.q
    @q.setter
    def q(self, q):
        self.virtual_contact.body.q = q
    @property
    def qdot(self):
        return self.virtual_contact.body.qdot
    @qdot.setter
    def qdot(self, qdot):
        self.virtual_contact.body.qdot = qdot
    # --------------------------------

    def reset(self):
        self.q = self.virtual_contact.body.q0
        self.qdot = self.virtual_contact.body.qdot0
        self.virtual_contact.reset_contacts()

    def evaluate_g_tensor(self, q1: torch.Tensor):
        qdot1 = (q1 - self.q) / self.dt
        if self.recompute_contacts_durint_optim:
            self.virtual_contact.update_contacts(q1, max_iter=self.update_contacts_iters) # self.ds doesnt get the necessary grad from q1
        p_force = self.virtual_contact.get_penalty_force(q1, qdot1)
        forces = p_force + self.gravity * self.body.m + self._u

        g = self.body.m * (q1 - self.q - self.dt * self.qdot) - self.dt**2 * forces
        return g
    # --------------------------------
    # numpy functions for newton solver
    # --------------------------------
    def evaluate_g(self, q: np.ndarray):
        return self.evaluate_g_tensor(torch.tensor(q)).detach().numpy()

    def evaluate_g_with_derv(self, q: np.ndarray):
        _q = torch.tensor(q).requires_grad_(True)
        g = self.evaluate_g_tensor(_q)
        return g.detach().numpy(), utils.compute_jacobian(g, _q).detach().numpy()

    def integration_BDF1(self, dt: float, q:torch.Tensor, qdot:torch.Tensor, u: torch.Tensor):
        _q0 = q
        _qdot0 = qdot

        q1 = _q0 + dt * _qdot0 # initialize q1

        # solve for new q1
        self._u = u
        if not self.recompute_contacts_durint_optim:
            self.virtual_contact.update_contacts(q1, max_iter=self.update_contacts_iters)
        q1 = torch.tensor(self.newton.solve(q1.detach().numpy(), self.evaluate_g, self.evaluate_g_with_derv))
        q1dot =  (q1 - _q0) / self.dt
        return q1, q1dot

    # --------------------------------
    # simulation methods (forward, backward)
    # --------------------------------
    def step(self, u: torch.Tensor):
        self.q, self.qdot = self.integration_BDF1(self.dt, self.q, self.qdot, u)
        return self.q, self.qdot

    def forward(self, inputs: torch.Tensor):
        self.reset()
        # self.virtual_contact.vis('image.png',u_arrow =(self.q[:2].detach().numpy(), inputs[0].detach().numpy()))
        qs, qdots = [self.q], [self.qdot]
        for i, u in enumerate(tqdm(inputs)):
            q, qdot = self.step(u)
            qs.append(q)
            qdots.append(qdot)

            # visualize
            self.virtual_contact.vis(f'output/image_{i}.png', title=f"step: {i} | ", u_arrow=(q[:2].detach().numpy(), 2*u.detach().numpy()))

        print('Num newton iterations avg:', self.newton.solver_report._num_iterations/len(inputs))
        self.newton.solver_report._num_iterations = 0.
        return torch.stack(qs), torch.stack(qdots)

    # --------------------------------
    # gradient methods (forward, backward)
    # --------------------------------
    def get_derivatives(self, q1:torch.Tensor, qprev:torch.Tensor, qdotprev:torch.Tensor, u:torch.Tensor):
        _, qgrad = self.evaluate_g_with_derv(q1)

        self.qdot = qdotprev.clone().requires_grad_(True)
        self.q = qprev.clone().requires_grad_(True)
        self._u = u.clone().requires_grad_(True)
        g = self.evaluate_g_tensor(q1)

        uprev_partial = utils.compute_jacobian(g, self._u).numpy()
        qprev_partial = utils.compute_jacobian(g, self.q).numpy()
        qdotprev_partial = utils.compute_jacobian(g, self.qdot).numpy()

        dqdu = -uprev_partial / qgrad
        dqdqprev = -qprev_partial / qgrad
        dqdqdotprev = -qdotprev_partial / qgrad

        return dqdu, dqdqprev, dqdqdotprev

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
