# Inspired by https://github.com/aravindr93/mjrl/blob/15bf3c0ed0c97fef761a8924d1b22413beb79900/mjrl/algos/mbrl/nn_dynamics.py
# Modified by Abhay Deshpande, Kay Ke, and Yunchu Zhang

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from typing import Callable, Sequence, Tuple
from functools import partial
from collections import defaultdict

from correct_il.models.spectral_norm import apply_spectral_norm

class WorldModel:
    def __init__(self, state_dim, act_dim, d_config,
                 hidden_size=(64,64),
                 fit_lr=1e-3,
                 fit_wd=0.0,
                 device='cpu',
                 activation='relu'):
        self.state_dim, self.act_dim = state_dim, act_dim
        self.device = device if device != 'gpu' else 'cuda'

        # construct the dynamics model
        self.dynamics_net = DynamicsNet(
            state_dim, act_dim, hidden_size).to(self.device)
        self.dynamics_net.set_transformations()  # in case device is different from default, it will set transforms correctly
        if "spectral_normalization" in d_config.lipschitz_type:
            self.dynamics_net.enable_spectral_normalization(d_config.lipschitz_constraint)
        if activation == 'tanh':
            self.dynamics_net.nonlinearity = torch.tanh
        self.dynamics_opt = torch.optim.Adam(self.dynamics_net.parameters(), lr=fit_lr, weight_decay=fit_wd)

        if "slack" in d_config.lipschitz_type:
            self.learnable_lambda = DynamicsNet(state_dim, act_dim, hidden_size,out_dim=1).to(self.device)
            self.learnable_lambda.set_transformations()
            self.dynamics_opt = torch.optim.Adam(list(self.dynamics_net.parameters())+list(self.learnable_lambda.parameters()), lr=fit_lr, weight_decay=fit_wd)

        """loss_fn is of the signature (X, Y_pred, Y) -> {loss}"""
        self.loss_fn = construct_loss_fn(d_config, self) # default loss is MSE on Y

    def to(self, device):
        self.dynamics_net.to(device)
        self.device = device

    def is_cuda(self):
        return next(self.dynamics_net.parameters()).is_cuda

    def predict(self, s, a):
        s = torch.as_tensor(s).float().to(self.device)
        a = torch.as_tensor(a).float().to(self.device)
        s_next = s + self.dynamics_net.predict(s, a)
        return s_next

    def cal_lambda(self, s, a):
        s = torch.as_tensor(s).float().to(self.device)
        a = torch.as_tensor(a).float().to(self.device)
        return self.learnable_lambda.predict(s, a)

    def f(self, s, a):
        return self.dynamics_net.predict(s, a)

    def init_normalization_to_data(self, s, a, sp, set_transformations=True):
        # set network transformations
        if set_transformations:
            s_shift, a_shift = torch.mean(s, dim=0), torch.mean(a, dim=0)
            s_scale, a_scale = torch.mean(torch.abs(s - s_shift), dim=0), torch.mean(torch.abs(a - a_shift), dim=0)
            out_shift = torch.mean(sp-s, dim=0)
            out_scale = torch.mean(torch.abs(sp-s-out_shift), dim=0)
            self.dynamics_net.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)
        else:
            s_shift     = torch.zeros(self.state_dim).cuda()
            s_scale    = torch.ones(self.state_dim).cuda()
            a_shift     = torch.zeros(self.act_dim).cuda()
            a_scale    = torch.ones(self.act_dim).cuda()
            out_shift   = torch.zeros(self.state_dim).cuda()
            out_scale  = torch.ones(self.state_dim).cuda()
        return (s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)

    def fit_dynamics(self, s, a, sp, batch_size, train_epochs, max_steps=1e10,
                     set_transformations=True, *args, **kwargs):
        # move data to correct devices
        assert type(s) == type(a) == type(sp)
        assert s.shape[0] == a.shape[0] == sp.shape[0]
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
            sp = torch.from_numpy(sp).float()

        s = s.to(self.device); a = a.to(self.device); sp = sp.to(self.device)
        s_shift, s_scale, a_shift, a_scale, out_shift, out_scale = \
            self.init_normalization_to_data(s, a, sp, set_transformations)

        # prepare data for learning
        # note how Y is normalized & residual
        X = (s, a) ; Y = (sp - s - out_shift) / (out_scale + 1e-8)

        return fit_model(
            self.dynamics_net, X, Y, self.dynamics_opt, self.loss_fn,
            batch_size, train_epochs, max_steps=max_steps)

    def local_lipschitz_coeff(self, s, a):
        s_a = torch.cat([s, a], dim=-1)
        def predict_concat(state_action):
            obs = state_action[...,:s.shape[-1]]
            act = state_action[...,s.shape[-1]:]
            return self.predict(obs, act)
        import functorch
        jac_fn = functorch.vmap(functorch.jacrev(predict_concat))
        jacs = jac_fn(s_a)
        assert jacs.shape == (s_a.shape[0], s.shape[-1], s_a.shape[-1])
        local_L = torch.linalg.norm(jacs, ord=2, dim=(-2,-1))
        return local_L

    @torch.no_grad()
    def eval_lipschitz_coeff(self, s, a, batch_size=None):
        batch_size = batch_size if batch_size else len(s)

        lipschits_coeffs = []
        for i in tqdm(range(0, len(s), batch_size)):
            batch_s = torch.as_tensor(s[i:i+batch_size], dtype=torch.float32, device=self.device)
            batch_a = torch.as_tensor(a[i:i+batch_size], dtype=torch.float32, device=self.device)
            local_L = self.local_lipschitz_coeff(batch_s, batch_a).cpu().numpy()
            lipschits_coeffs.append(local_L)
        return np.concatenate(lipschits_coeffs, axis=0)

    @torch.no_grad()
    def eval_prediction_error(self, s, a, s_next, batch_size, reduce_err=True):
        s_next = torch.as_tensor(s_next).float().to(self.device)

        transforms = self.dynamics_net.get_params()['transforms']
        s_shift, s_scale, a_shift, a_scale, out_shift, out_scale = transforms

        i = 0
        err_norms = []
        num_steps = int(s.shape[0] // batch_size)
        for _ in range(num_steps):
            sp = self.predict(s[i:i+batch_size], a[i:i+batch_size])
            error = (sp - s_next[i:i+batch_size]).norm(dim=-1)
            i += batch_size
            err_norms += error.cpu().tolist()
        return np.mean(err_norms) if reduce_err else err_norms

def construct_loss_fn(d_config, dynamics):
    if d_config.lipschitz_type == "soft_sampling":
      return partial(soft_sampling_lipschitz_loss,
                                lipschitz_constraint=d_config.lipschitz_constraint,
                                sampling_eps=d_config.soft_lipschitz_sampling_eps,
                                soft_lipschitz_penalty_weight=d_config.soft_lipschitz_penalty_weight,
                                n_samples=d_config.soft_lipschitz_n_samples,
                                predict_fn=dynamics.predict)
    elif d_config.lipschitz_type == "soft_sampling_slack":
      return partial(soft_sampling_lipschitz_slack_loss,
                                lipschitz_constraint=d_config.lipschitz_constraint,
                                sampling_eps=d_config.soft_lipschitz_sampling_eps,
                                soft_lipschitz_penalty_weight=d_config.soft_lipschitz_penalty_weight,
                                n_samples=d_config.soft_lipschitz_n_samples,
                                predict_fn=dynamics.predict,
                                lambda_fn=dynamics.cal_lambda)
    elif d_config.lipschitz_type == "slack" or d_config.lipschitz_type == "spectral_normalization_slack":
      slack_weight = float(d_config.slack_weight) if 'slack_weight' in d_config else 1.0
      return partial(slack_loss, lambda_fn=dynamics.cal_lambda, slack_weight=slack_weight)
    elif d_config.lipschitz_type == "soft_jac":
      return partial(soft_lipschitz_loss,
                                lipschitz_constraint=d_config.lipschitz_constraint,
                                soft_lipschitz_penalty_weight=d_config.soft_lipschitz_penalty_weight,
                                predict_fn=dynamics.predict)
    else:
      return wrapper_mse_loss

def wrapper_mse_loss(_, Y_pred, Y):
    mse_loss = F.mse_loss(Y_pred, Y)
    return {
        'loss': mse_loss,
        'mean_error': (Y_pred - Y).norm(dim=-1).mean().detach().cpu().numpy()
    }

def sample_estimate_lipschitz(s, a, Y_pred, predict_fn, n_samples, sampling_eps):
  # repeat the rows so we can perturb each element multiple times
  s = torch.as_tensor(s).to(Y_pred.device).repeat_interleave(n_samples, dim=0)
  a = torch.as_tensor(a).to(Y_pred.device).repeat_interleave(n_samples, dim=0)
  Y_pred_repeat = Y_pred.repeat_interleave(n_samples, dim=0)
  s_noise = torch.randn_like(s, device=s.device)
  a_noise = torch.randn_like(a, device=a.device)
  noisy_s = s + s_noise * sampling_eps
  noisy_a = a + a_noise * sampling_eps
  noisy_output = predict_fn(noisy_s, noisy_a)
  noise_mag = torch.linalg.norm(torch.cat([s_noise, a_noise], dim=-1), dim=-1) # batch_size * n_samples
  output_diff = torch.linalg.norm(Y_pred_repeat - noisy_output, dim=-1)
  return noise_mag, output_diff

def soft_sampling_lipschitz_loss(
            X: Tuple[torch.Tensor, torch.Tensor], Y_pred: torch.Tensor, Y: torch.Tensor,
            lipschitz_constraint: float, sampling_eps: float, soft_lipschitz_penalty_weight: float, n_samples: int,
            predict_fn: Callable[[torch.Tensor,torch.Tensor], torch.Tensor]):
  s, a = X
  noise_mag, output_diff = sample_estimate_lipschitz(s, a, Y_pred, predict_fn, n_samples, sampling_eps)
  lipschitz_penalty = torch.relu(output_diff - lipschitz_constraint * noise_mag)
  lipschitz_penalty = soft_lipschitz_penalty_weight * lipschitz_penalty.mean()
  mse_loss = F.mse_loss(Y_pred, Y)
  return {
      'loss': mse_loss + lipschitz_penalty,
      'mse_loss_tensor' : mse_loss,
      'mse_loss' : mse_loss.detach().to('cpu').numpy(),
      'lipschitz_penalty': lipschitz_penalty.detach().to('cpu').numpy()
  }

def soft_sampling_lipschitz_slack_loss(
            X: Tuple[torch.Tensor, torch.Tensor], Y_pred: torch.Tensor, Y: torch.Tensor,
            lipschitz_constraint: float, sampling_eps: float, soft_lipschitz_penalty_weight: float, n_samples: int,
            predict_fn: Callable[[torch.Tensor,torch.Tensor], torch.Tensor], lambda_fn):
  s, a = X
  noise_mag, output_diff = sample_estimate_lipschitz(s, a, Y_pred, predict_fn, n_samples, sampling_eps)
  lipschitz_penalty = torch.relu(output_diff - lipschitz_constraint * noise_mag)
  slack = lambda_fn(s, a)
  lipschitz_penalty = slack * lipschitz_penalty + 1 - torch.exp(- 0.1 * torch.abs(slack).mean())
  lipschitz_penalty = soft_lipschitz_penalty_weight * lipschitz_penalty.mean()
  mse_loss = F.mse_loss(Y_pred, Y)
  return {
      'loss': mse_loss + lipschitz_penalty,
      'mse_loss_tensor' : mse_loss,
      'mse_loss' : mse_loss.detach().to('cpu').numpy(),
      'lipschitz_penalty': lipschitz_penalty.detach().to('cpu').numpy()
  }

def slack_loss(X: Tuple[torch.Tensor, torch.Tensor], Y_pred: torch.Tensor, Y: torch.Tensor,
               lambda_fn=None, slack_weight=0.1):
  s,a = X
  slack = lambda_fn(s, a)
  slack = torch.nn.Sigmoid()(slack) # squeeze to 0~1
  mse_loss = F.mse_loss(Y_pred, Y, reduction='none')
  weighted_mse_loss = torch.linalg.norm(slack * mse_loss, dim=-1).mean()
  avg_slack_size = slack.mean()

  return {
      'loss': weighted_mse_loss.mean() - slack_weight * avg_slack_size,
      'mse_loss_tensor' : mse_loss,
      'mse_loss' : mse_loss.norm(dim=-1).mean().detach().cpu().numpy(),
      'weighted_mse_loss' : weighted_mse_loss.mean().detach().to('cpu').numpy(),
      'slack_penalty': avg_slack_size.detach().to('cpu').numpy(),
      'slack_std': torch.std(slack).detach().cpu().numpy()
  }

def soft_lipschitz_loss(X: Tuple[torch.Tensor, torch.Tensor], Y_pred: torch.Tensor, Y: torch.Tensor,
                        lipschitz_constraint: float, soft_lipschitz_penalty_weight: float,
                        predict_fn: Callable[[torch.Tensor,torch.Tensor], torch.Tensor]):
  s, a = X
  s = torch.as_tensor(s).to(Y_pred.device)
  a = torch.as_tensor(a).to(Y_pred.device)
  def predict_concat(state_action):
    obs = state_action[...,:s.shape[-1]]
    act = state_action[...,s.shape[-1]:]
    return predict_fn(obs, act)
  # compute batched jacobian with vectorization
  import functorch
  jacs = functorch.vmap(functorch.jacrev(predict_concat))(torch.cat([s, a], dim=-1))
  assert jacs.shape == (s.shape[0], s.shape[-1], s.shape[-1]+a.shape[-1])
  local_L = torch.linalg.norm(jacs, ord=2, dim=(-2,-1)) # local lipschitz coeff is spectral norm of jacobian
  lipschitz_penalty = torch.relu(local_L - lipschitz_constraint)
  lipschitz_penalty = soft_lipschitz_penalty_weight * lipschitz_penalty.mean()
  mse_loss = F.mse_loss(Y_pred, Y)
  return {
      'loss': mse_loss + lipschitz_penalty,
      'mse_loss' : mse_loss.detach().to('cpu').numpy(),
      'lipschitz_penalty': lipschitz_penalty.detach().to('cpu').numpy()
  }


def _apply_spectral_normalization_recursively(
    model: nn.Module,
    lipschitz_constraint: float) -> None:
    for _, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            for m in module:
                _apply_spectral_normalization_recursively(m, lipschitz_constraint)
        else:
            if "weight" in module._parameters:
                apply_spectral_norm(module, lipschitz_constraint=lipschitz_constraint)


class DynamicsNet(nn.Module):
    def __init__(self, state_dim, act_dim,
                 hidden_size=(64,64),
                 s_shift = None,
                 s_scale = None,
                 a_shift = None,
                 a_scale = None,
                 out_shift = None,
                 out_scale = None,
                 out_dim = None,
                 use_mask = True,
                 ):
        super(DynamicsNet, self).__init__()

        self.state_dim, self.act_dim, self.hidden_size = state_dim, act_dim, hidden_size
        self.out_dim = state_dim if out_dim is None else out_dim
        self.layer_sizes = (state_dim + act_dim, ) + tuple(hidden_size) + (self.out_dim, )
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                    for i in range(len(self.layer_sizes)-1)])
        self.nonlinearity = torch.relu
        self.use_mask = use_mask
        self.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)

    def enable_spectral_normalization(self, lipschitz_constraint):
        _apply_spectral_normalization_recursively(self.fc_layers, lipschitz_constraint)

    def set_transformations(self, s_shift=None, s_scale=None,
                            a_shift=None, a_scale=None,
                            out_shift=None, out_scale=None):
        if s_shift is None:
            self.s_shift     = torch.zeros(self.state_dim)
            self.s_scale    = torch.ones(self.state_dim)
            self.a_shift     = torch.zeros(self.act_dim)
            self.a_scale    = torch.ones(self.act_dim)
            self.out_shift   = torch.zeros(self.out_dim)
            self.out_scale  = torch.ones(self.out_dim)
        elif type(s_shift) == torch.Tensor:
            self.s_shift, self.s_scale = s_shift, s_scale
            self.a_shift, self.a_scale = a_shift, a_scale
            self.out_shift, self.out_scale = out_shift, out_scale
        elif type(s_shift) == np.ndarray:
            self.s_shift     = torch.from_numpy(np.float32(s_shift))
            self.s_scale    = torch.from_numpy(np.float32(s_scale))
            self.a_shift     = torch.from_numpy(np.float32(a_shift))
            self.a_scale    = torch.from_numpy(np.float32(a_scale))
            self.out_shift   = torch.from_numpy(np.float32(out_shift))
            self.out_scale  = torch.from_numpy(np.float32(out_scale))
        else:
            print("Unknown type for transformations")
            quit()

        device = next(self.parameters()).data.device
        self.s_shift, self.s_scale = self.s_shift.to(device), self.s_scale.to(device)
        self.a_shift, self.a_scale = self.a_shift.to(device), self.a_scale.to(device)
        self.out_shift, self.out_scale = self.out_shift.to(device), self.out_scale.to(device)
        # if some state dimensions have very small variations, we will force it to zero
        self.mask = self.out_scale >= 1e-8

        self.transformations = dict(s_shift=self.s_shift, s_scale=self.s_scale,
                                    a_shift=self.a_shift, a_scale=self.a_scale,
                                    out_shift=self.out_shift, out_scale=self.out_scale)

    def forward(self, s, a):
        # Given raw input return the normalized residual
        if s.dim() != a.dim():
            print("State and action inputs should be of the same size")
        # normalize inputs
        s_in = (s - self.s_shift)/(self.s_scale + 1e-8)
        a_in = (a - self.a_shift)/(self.a_scale + 1e-8)
        out = torch.cat([s_in, a_in], -1)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        return out

    def predict(self, s, a):
        # Given raw input return the (unnormalized) residual
        out = self.forward(s, a)
        out = out * (self.out_scale + 1e-8) + self.out_shift
        out = out * self.mask if self.use_mask else out
        return out

    def get_params(self):
        network_weights = [p.data for p in self.parameters()]
        transforms = (self.s_shift, self.s_scale,
                      self.a_shift, self.a_scale,
                      self.out_shift, self.out_scale)
        return dict(weights=network_weights, transforms=transforms)

    def set_params(self, new_params):
        new_weights = new_params['weights']
        s_shift, s_scale, a_shift, a_scale, out_shift, out_scale = new_params['transforms']
        for idx, p in enumerate(self.parameters()):
            p.data = new_weights[idx]
        self.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)


def fit_model(nn_model, X, Y, optimizer, loss_fn,
              batch_size, epochs, max_steps=1e10):
    """
    :param nn_model:        pytorch model of form Y = f(*X) (class)
    :param X:               tuple of necessary inputs to the function
    :param Y:               desired output from the function (tensor)
    :param optimizer:       optimizer to use
    :param loss_func:       loss criterion
    :param batch_size:      mini-batch size
    :param epochs:          number of epochs
    :return:
    """

    assert type(X) == tuple
    for d in X: assert type(d) == torch.Tensor
    assert type(Y) == torch.Tensor
    device = Y.device
    for d in X: assert d.device == device

    num_samples = Y.shape[0]
    num_steps = int(num_samples // batch_size)
    epoch_losses = []
    steps_so_far = 0
    for ep in tqdm(range(epochs)):
        rand_idx = torch.LongTensor(np.random.permutation(num_samples)).to(device)
        ep_loss = defaultdict(int)

        for mb in range(num_steps):
            data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
            batch_X  = [d[data_idx] for d in X]
            batch_Y  = Y[data_idx]
            optimizer.zero_grad()
            Y_hat    = nn_model.forward(*batch_X)
            loss_dict = loss_fn(batch_X, Y_hat, batch_Y)
            loss_dict['loss'].backward()
            optimizer.step()
            loss_dict['loss'] = loss_dict['loss'].detach().to('cpu').numpy()
            for key, value in loss_dict.items():
                if key == "mse_loss_tensor":
                    continue
                ep_loss[key] += value
        epoch_losses.append(ep_loss)
        steps_so_far += num_steps
        if steps_so_far >= max_steps:
            print("Number of grad steps exceeded threshold. Terminating early.")
            break
    epoch_losses = list_of_dict_to_dict_of_list(epoch_losses, float(num_steps))
    return epoch_losses

def list_of_dict_to_dict_of_list(a_list, divisor=1.0):
    if len(a_list) == 0:
        return []
    keys = a_list[0].keys()
    new_list = {k:[] for k in keys}
    for a_dict in a_list:
        for k in keys:
            new_list[k].append(a_dict[k] / divisor)
    return new_list
