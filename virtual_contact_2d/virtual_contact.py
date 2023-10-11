import numpy as np
import torch
from sdf import d2
from . import utils
from .body2d import Body2D


class BodyPlaneVirtualContacts:
    def __init__(self, body: Body2D,
                 N_contacts=10,
                 contact_init_iters = 100, kn=1e5,
                 damping=False, kd=3e1,
                 smoothed_penalty=False, smoothed_alpha=1e-2,
                 friction=False, kt=1e3, mu=0.1) -> None:

        # parameters
        self.smoothed_alpha = smoothed_alpha
        self.contact_init_iters = contact_init_iters
        self.kn = kn
        self.damping = damping
        self.kd = kd
        self.smoothed_penalty = smoothed_penalty
        self.friction = friction
        # friction params
        self.kt = kt
        self.mu = mu

        # N: number of virtual contacts we want to keep track of
        self.N_contacts = N_contacts
        self.body = body
        self.plane_sdf = d2.line(normal=torch.tensor([0., 1.]), point=torch.tensor([0., 0.]))
        self.plane_sdf = d2.rectangle(size=torch.tensor([15., 2.]), center=torch.tensor([0., -2.]))

        self.x_min, self.x_max = utils.estimate_bounds2_0(self.plane_sdf | self.body.sdf(self.body.q0))
        # Uniformly samples and updates contacts
        self.xs_w = torch.tensor(self.x_min + np.random.uniform(size=(self.N_contacts, 2)) * (self.x_max - self.x_min))
        self.reset_contacts()

        # Contact states
        self.penalty_force: torch.Tensor
        self.xs_w   : torch.Tensor # positions of contacts in body world
        self.ds     : torch.Tensor # penetration depths
        self.x_c_b1 : torch.Tensor
        self.x_c_b2 : torch.Tensor
        self.d1     : torch.Tensor
        self.d2     : torch.Tensor

        # For body-ground contacts we can hard code the normal and tangent vectors
        self.ns = torch.zeros((self.N_contacts, 2))      # normal vectors in world frame
        self.ns[:,1] = 1.
        self.ts = torch.zeros((self.N_contacts, 2))      # tangent vectors in world frame
        self.ts[:,0] = 1.

    def reset_contacts(self):
        # Uniformly samples and updates contacts
        self.xs_w = torch.tensor(self.x_min + np.random.uniform(size=(self.N_contacts, 2)) * (self.x_max - self.x_min))
        self.update_contacts(self.body.q, max_iter=self.contact_init_iters)

    def update_contacts(self, q: torch.Tensor, max_iter):
        # Updates contacts given a new state [q] and an initial guess [xs_w] for [max_iter] iterations
        ret = utils.find_contact(self.body._sdf, self.plane_sdf, self.xs_w, max_iter=max_iter, q1=q)
        self.xs_w, self.x_c_b1, self.x_c_b2, self.ds, self.d2, n_b1, n_b2 = ret

    def get_penalty_force(self, q1: torch.Tensor, qdot1: torch.Tensor):
        xbc_w = self.x_c_b1.matmul(utils.rot(q1[2]).T) # R * x_c_b1
        xdots_w = qdot1[:2] + torch.stack([xbc_w[:, 1], -xbc_w[:, 0]], dim=-1) * qdot1[2] # [N, 2] #  v_body_w + w x xbc_w (cross product [0, 0 ,w] x [vx,vy,0])
        ddots = torch.sum(xdots_w * self.ns[:, :2], dim=1, keepdim=True)  # [N, 1] # penetration velocities

        # computing penalty forces
        # needed: self.ds, xdots_w, ddots, -- constant for now --: self.ns, self.ts
        # f(ds, ddots, xdots_w, ns, ts)
        fts = torch.zeros((self.N_contacts, 1)) # [N, 2]
        fcs = self.kn
        if self.damping:
            fcs -= self.kd * ddots # [N, 1]

        if self.smoothed_penalty:
            a = self.smoothed_alpha
            fcs *= torch.where(self.ds >= 0, self.ds+a, a*torch.exp(self.ds/a)) # [N, 1]
        else:
            fcs *= torch.max(torch.zeros(1), self.ds)

        if self.friction:
            # ft = − min(kt‖tdot‖, μ‖fc‖) tdot/‖tdot‖
            xdots_w_tangential_norm = torch.norm(xdots_w * self.ts[:, :2], dim=1, keepdim=True) # [N, 1]
            fts = - torch.min(self.kt * xdots_w_tangential_norm,
                              self.mu * torch.norm(fcs, dim=1, keepdim=True)) # [N, 1]

        # both are of the form [N, 2] where each of the N elements is [fx, fy]
        fxy_points = fcs * self.ns + fts * self.ts # [N, 2]


        # cross product [x,y,0] x [fx, fy, 0] -> [0, 0, x*fy - y*fx]
        taus = xbc_w[:, 0] * fxy_points[:, 1] - xbc_w[:, 1] * fxy_points[:, 0]
        self.penalty_force = torch.cat([fxy_points, taus[:, None]], dim=1).sum(dim=0)#  .mean(dim=0) # [N, 3]
        return self.penalty_force # [3]



    def vis(self, path=None, u_arrow=None, title=""):
        utils.vis(self.plane_sdf, self.body.sdf(self.body.q), self.xs_w, self.x_min, self.x_max,
                  u_arrow = u_arrow,
                  penalty_arrow = 2*self.penalty_force.detach().numpy(),
                  path=path, title=title)
