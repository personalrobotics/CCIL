from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import d3rlpy

from d3rlpy.argument_utility import (
    ActionScalerArg,
    EncoderArg,
    ScalerArg,
    UseGPUArg,
)
from d3rlpy.models.optimizers import AdamFactory, OptimizerFactory
from d3rlpy.constants import ActionSpace
from d3rlpy.algos.torch.bc_impl import BCImpl, BCBaseImpl


class CustomBC(d3rlpy.algos.BC):
    r"""Customized Behavior Cloning algorithm.
    
    Add functions include
        (1) customized loss function
        (2) inject noise when computing loss

    Behavior Cloning (BC) is to imitate actions in the dataset via a supervised
    learning approach.
    Since BC is only imitating action distributions, the performance will be
    close to the mean of the dataset even though BC mostly works better than
    online RL algorithms.

    .. math::

        L(\theta) = \mathbb{E}_{a_t, s_t \sim D}
            [(a_t - \pi_\theta(s_t))^2]

    Args:
        learning_rate (float): learing rate.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        policy_type (str): the policy type. The available options are
            ``['deterministic', 'stochastic']``.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action scaler. The available options are ``['min_max']``.
        impl (d3rlpy.algos.torch.bc_impl.BCImpl):
            implemenation of the algorithm.

    """

    _policy_type: str
    _impl: Optional[BCImpl]

    def __init__(
        self,
        *,
        learning_rate: float = 1e-3,
        optim_factory: OptimizerFactory = AdamFactory(),
        encoder_factory: EncoderArg = "default",
        batch_size: int = 100,
        n_frames: int = 1,
        policy_type: str = "deterministic",
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        impl: Optional[BCBaseImpl] = None,
        loss_weights: None, # <------ Weight the loss at each dim differently
        noise_cov: None, # <--------- Noise Injection at compute_loss
        **kwargs: Any
    ):
        super().__init__(
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            batch_size=batch_size,
            n_frames=n_frames,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            impl=impl,
            **kwargs,
        )
        self._policy_type = policy_type
        self.loss_weights = loss_weights
        self.noise_cov = noise_cov

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = CustomBCImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=self._learning_rate,
            optim_factory=self._optim_factory,
            encoder_factory=self._encoder_factory,
            policy_type=self._policy_type,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            loss_weights = self.loss_weights,
            noise_cov = self.noise_cov,
        )
        self._impl.build()


###################################################################################
# Swap out BC impl to change the loss function
###################################################################################

import torch
from d3rlpy.models.torch import (
    DeterministicRegressor,
    ProbablisticRegressor,)
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.gpu import Device
from d3rlpy.preprocessing import ActionScaler, Scaler
import torch.nn.functional as F

class CustomBCImpl(BCImpl):
    _policy_type: str
    _imitator: Optional[Union[DeterministicRegressor, ProbablisticRegressor]]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        policy_type: str,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        loss_weights: None, # <------ Weight the loss at each dim differently
        noise_cov: None, # <--------- Noise Injection at compute_loss
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            policy_type=policy_type,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
        )
        self.compute_loss = self.compute_loss_naive if loss_weights is None and noise_cov is None else None
        self.tensor_device = 'cpu' if not use_gpu else 'cuda'
        if loss_weights:
            self.loss_weights = torch.tensor(loss_weights, dtype=torch.float32,
                device=self.tensor_device).reshape([len(loss_weights), 1])
            self.compute_loss = self.compute_loss_weight
        else:
            self.loss_weights = None
        if noise_cov:
            self.compute_loss = self.compute_loss_weight
        self.noise_cov = noise_cov

    def compute_loss_naive(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._imitator is not None
        # The original DeterministicRegressor
        # def compute_error(
        #     self, x: torch.Tensor, action: torch.Tensor
        # ) -> torch.Tensor:
        #     return F.mse_loss(self.forward(x), action)
        return F.mse_loss(self._imitator.forward(obs_t), act_t)
    
    def weight_losses(self, losses: torch.Tensor):
        return torch.matmul(losses, self.loss_weights).sum() / losses.numel() if self.loss_weights is not None else losses.mean()
    
    def compute_loss_weight(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._imitator is not None
        #import pdb;pdb.set_trace()

        # x = obs_t + self.noise_cov * torch.rand(obs_t.shape, device=self.tensor_device) if self.noise_cov else obs_t
        # losses = F.mse_loss(self._imitator.forward(x), act_t, reduction='none')
        # return torch.matmul(losses, self.loss_weights).sum() / losses.numel() if self.loss_weights else losses.mean()

        x = obs_t
        if self.noise_cov:
            mask = torch.rand(x.shape[:1], device=self.tensor_device) > 0.5
            noise = self.noise_cov * torch.rand(obs_t.shape, device=self.tensor_device)
            x[mask] = x[mask] + noise[mask]
        losses = F.mse_loss(self._imitator.forward(x), act_t, reduction='none')
        loss = self.weight_losses(losses)
        return loss
    