"""Discrete double integrator with closed-form corrective labels.

State s = (x, v), scalar action a, step s' = s + f(s, a) with
f(s, a) = A s + B a, A = [[0, h], [0, 0]], B = [0, h]^T.
Derivations are in docs/fixtures/analytic-dynamics.md.
"""
import numpy as np
import torch


class DoubleIntegrator:
    def __init__(self, h):
        self.h = float(h)
        self.A = np.array([[0.0, self.h], [0.0, 0.0]])
        self.B = np.array([[0.0], [self.h]])

    # --- dynamics ---------------------------------------------------------
    def f_np(self, s, a):
        return s @ self.A.T + a @ self.B.T

    def step_np(self, s, a):
        return s + self.f_np(s, a)

    def model(self, dtype=torch.float32):
        """Exact model exposing the legacy WorldModel.f(s, a) interface."""
        A = torch.as_tensor(self.A, dtype=dtype)
        B = torch.as_tensor(self.B, dtype=dtype)

        class ExactModel:
            def f(self_, s, a):
                return s @ A.T + a @ B.T

        return ExactModel()

    # --- closed-form candidates (float64) ---------------------------------
    def backward_euler(self, s_star, a_star):
        """s_G = s* - f(s*, a*): label (s_G, a*) -> s*."""
        return s_star - self.f_np(s_star, a_star)

    def exact_preimage(self, s_star, a_star):
        """Solve s_G + f(s_G, a*) = s*: s_G = (I + A)^-1 (s* - B a*)."""
        return np.linalg.solve(np.eye(2) + self.A, (s_star - a_star @ self.B.T).T).T

    def noisy_action(self, s_next, a_noisy):
        """s_G = s*_{t+1} - f(s*_{t+1}, a* + delta): label (s_G, a* + delta) -> s*_{t+1}."""
        return s_next - self.f_np(s_next, a_noisy)

    def residual(self, s_g, a_g, target):
        """Norm of the true one-step error of label (s_g, a_g) -> target."""
        return np.linalg.norm(self.step_np(s_g, a_g) - target, axis=-1)
