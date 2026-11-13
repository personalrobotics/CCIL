"""Spectral normalization that clips a layer to a Lipschitz bound.

Derived from PyTorch's BSD-3-Clause implementation; see the adjacent license
and notice. Training updates one power-iteration estimate per forward pass,
while evaluation reuses its stored vectors.
"""
from __future__ import annotations

import torch
from torch.nn.functional import normalize


class LipschitzSpectralNorm:
    def __init__(self, lipschitz_constraint: float, n_power_iterations: int = 1, eps: float = 1e-12):
        self.lipschitz_constraint, self.n_power_iterations, self.eps = lipschitz_constraint, n_power_iterations, eps

    def compute_weight(self, module, do_power_iteration: bool) -> torch.Tensor:
        weight, u, v = module.weight_orig, module.weight_u, module.weight_v
        weight_mat = weight.reshape(weight.size(0), -1)
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                # Clone so autograd does not see the in-place buffer updates of later passes.
                u = u.clone(memory_format=torch.contiguous_format)
                v = v.clone(memory_format=torch.contiguous_format)
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        return weight / torch.max(sigma / self.lipschitz_constraint, torch.tensor(1.0, device=sigma.device))

    def __call__(self, module, inputs) -> None:
        module.weight = self.compute_weight(module, do_power_iteration=module.training)


def apply_spectral_norm(module, lipschitz_constraint: float) -> None:
    """Reparameterize module.weight as weight_orig with power-iteration buffers weight_u, weight_v."""
    fn = LipschitzSpectralNorm(lipschitz_constraint)
    weight = module._parameters["weight"]
    with torch.no_grad():
        h, w = weight.size(0), weight[0].numel()
        u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
        v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
    del module._parameters["weight"]
    module.register_parameter("weight_orig", weight)
    module.weight = weight.data
    module.register_buffer("weight_u", u)
    module.register_buffer("weight_v", v)
    module.register_forward_pre_hook(fn)
