import torch
from sdf import d2
from . import utils


class Body2D:
    def __init__(self, sdf: d2.SDF2, m:float = 0.1, q0: torch=None, qdot0: torch=None) -> None:
        self._sdf   = sdf
        self.m      = m

        # State Init (used to reset the simulation)
        self.q0     = q0 if q0 is not None else torch.zeros(3)       # initialization state
        self.qdot0  = qdot0 if qdot0 is not None else torch.zeros(3) # initialization state

        # State
        self.q      = self.q0.detach().clone()  # current state (changes)
        self.qdot   = self.qdot0.detach().clone()  # current state deriv (changes) # is in global frame

    def sdf(self, q: torch.Tensor=None):
        # q: theta describing the rotation of the body
        def sdf_modified(xw):
            # xw: points in world frame
            xb = utils.to_body_frame(xw, q) if q is not None else xw
            return self._sdf(xb)
        return sdf_modified


