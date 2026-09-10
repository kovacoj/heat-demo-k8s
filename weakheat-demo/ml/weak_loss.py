"""Weak-form residual in PyTorch.

The Firedrake weak formulation, tested against the complete CG1 basis,
reduces the heat equation to

    M dc/dt + alpha K c = 0

with M and K assembled once by Firedrake.  PyTorch supplies the neural
coefficient vector c_theta(t) and its first time derivative (autograd in
time).  The residual

    r_theta = M c_dot_theta + alpha K c_theta

never requires second spatial derivatives of the network.  This is the
central methodological statement of the demo: integration by parts moves
the Laplacian onto the FE test functions.
"""
import torch

try:
    from .dataset import load_operators
except ImportError:  # script-style
    from dataset import load_operators


class WeakResidual:
    def __init__(self, operators_dir=None, scale_eps=1e-8):
        self.M, self.K, self.lumped = load_operators(operators_dir)
        self.N = self.M.shape[0]
        self.scale_eps = scale_eps

    def residual(self, u, du_dt, alpha):
        """u [B,N], du_dt [B,N], alpha [B] -> r [B,N]."""
        Mdu = torch.sparse.mm(self.M, du_dt.T).T
        Ku = torch.sparse.mm(self.K, u.T).T
        return Mdu + alpha.view(-1, 1) * Ku

    def scaled_residual(self, u, du_dt, alpha):
        """Lumped-mass scaled residual (removes cell-size dependence)."""
        return self.residual(u, du_dt, alpha) / (self.lumped + self.scale_eps)

    def weak_loss(self, u, du_dt, alpha):
        r = self.scaled_residual(u, du_dt, alpha)
        return torch.mean(r ** 2)

    @torch.no_grad()
    def relative_l2(self, u_pred, u_true):
        """FE relative L2 error per sample: sqrt(e^T M e / u^T M u).

        u_pred, u_true: [B,N] -> [B]
        """
        e = u_pred - u_true
        num = (e * torch.sparse.mm(self.M, e.T).T).sum(dim=1)
        den = (u_true * torch.sparse.mm(self.M, u_true.T).T).sum(dim=1)
        return torch.sqrt(num / den.clamp_min(1e-12))

    @torch.no_grad()
    def total_mass(self, u):
        """m(t) = 1^T M u(t) per sample."""
        return (torch.sparse.mm(self.M, u.T).T).sum(dim=1)
