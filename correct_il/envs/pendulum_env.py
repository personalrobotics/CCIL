from typing import Union
import numpy as np
import scipy.integrate
import gym
import torch

class PendulumEnv_NP(gym.Env):
    def __init__(self, l=1.0, g=9.81, dt=0.02, mode="rk4", discontinuity=False):
        super().__init__()
        self.l = l
        self.g = g
        self.dt = dt
        self.mode = mode
        # state is [sin(theta), cos(theta), theta_dot]
        self.observation_space = gym.spaces.Box(np.array([-1., -1., -4*np.pi]),
                                           np.array([1., 1., 4*np.pi]))
        self.action_space = gym.spaces.Box(np.array([-3.0]), np.array([3.0]))
        self.curr_state = None
        self.has_discontinuity = discontinuity

    def reset(self):
        theta, theta_dot = np.random.uniform([-1,-0.5],[1,0.5])
        self.curr_state = np.array([np.sin(theta), np.cos(theta), theta_dot])
        return self.curr_state

    def _dynamics(self, s, a):
        out = np.array([s[2] * s[1],
                        s[2] * -s[0],
                        -(self.g/self.l)*s[0]+a])
        return out

    def set_state(self, state: np.ndarray):
        assert state.shape == (3,)
        self.curr_state = state

    def forward_solver(self, curr_state, a):
        res = scipy.integrate.solve_ivp(lambda _,s: self._dynamics(s, a),
                                                        (0, self.dt),
                                                        curr_state,
                                                        method="RK45",
                                                        dense_output=True)
        return res.sol(self.dt)

    def _reward(self, s, action, next_state):
        theta = np.arctan2(s[0], s[1])
        if theta < 0:
            theta += 2*np.pi
        state = np.array([theta, s[2]])
        goal = np.array([np.pi, 0.0])
        return -0.5*np.linalg.norm(state-goal)**2 - 0.5*action**2

    def step(self, action: np.ndarray):
        a = action.item()
        if self.mode == "forward":
            next_state = self.curr_state + self.dt * self._dynamics(self.curr_state, a)
        else:
            if self.mode == "backward":
                mode = "BDF"
            elif self.mode == "rk4":
                mode = "RK45"
            else:
                raise ValueError(f"Unrecognized mode {self.mode}")
            res = scipy.integrate.solve_ivp(lambda _,s: self._dynamics(s, a),
                                                        (0, self.dt),
                                                        self.curr_state,
                                                        method=mode,
                                                        dense_output=True)
            next_state = res.sol(self.dt)
        rew = self._reward(self.curr_state, action, next_state)
        self.curr_state = next_state

        if self.has_discontinuity:
            theta = np.arctan2(self.curr_state[0], self.curr_state[1])
            if np.abs(theta - np.pi/6) < 1e-2 or np.abs(theta - np.pi/3) < 1e-2 :
                self.curr_state[2] = -self.curr_state[2]

        return self.curr_state, np.array(rew), False, {}

class PendulumEnv_Torch(gym.Env):
    def __init__(self, l=1.0, g=9.81, dt=0.02, device="cpu"):
        super().__init__()
        self.l = l
        self.g = g
        self.dt = dt
        self.device = device
        # state is [sin(theta), cos(theta), theta_dot]
        self.observation_space = gym.spaces.Box(np.array([-1., -1., -4*np.pi]),
                                           np.array([1., 1., 4*np.pi]))
        self.action_space = gym.spaces.Box(np.array([-3.0]), np.array([3.0]))
        self.curr_state = None

    def _torchify(self, *xs):
        ret = tuple(torch.as_tensor(x).float().to(self.device) for x in xs)
        return ret[0] if len(ret) == 1 else ret

    def reset(self):
        dist = torch.distributions.Uniform(*self._torchify([-1,-0.5], [1,0.5]))
        theta, theta_dot = dist.sample().to(self.device)
        self.curr_state = torch.stack([torch.sin(theta), torch.cos(theta), theta_dot])
        return self.curr_state

    def _dynamics(self, s, a):
        out = torch.stack([s[2] * s[1],
                        s[2] * -s[0],
                        -(self.g/self.l)*s[0]+a])
        return out

    def set_state(self, state):
        state = self._torchify(state)
        assert state.shape == (3,)
        self.curr_state = state

    def to(self, device):
        self.device = device
        if self.curr_state:
            self.curr_state = self.curr_state.to(device)

    def _reward(self, s, action, next_state):
        theta = torch.arctan2(s[0], s[1])
        if theta < 0:
            theta = theta + 2*np.pi
        state = torch.stack([theta, s[2]])
        goal = torch.tensor([np.pi, 0.0], device=self.device)
        return -0.5*torch.norm(state-goal)**2 - 0.5*action**2

    def step(self, action):
        a = self._torchify(action)
        next_state = self.curr_state + self.dt * self._dynamics(self.curr_state, a.reshape(-1)[0])
        rew = self._reward(self.curr_state, a, next_state)
        self.curr_state = next_state
        return self.curr_state, rew, False, {}

class PendulumEnvCont(gym.Wrapper):
    env: Union[PendulumEnv_NP, PendulumEnv_Torch]
    def __init__(self, l=1.0, g=9.81, dt=0.02, backend="numpy", **kwargs):
        if backend == "numpy":
            env = PendulumEnv_NP(l, g, dt, discontinuity=False, **kwargs)
        elif backend == "torch":
            env = PendulumEnv_Torch(l, g, dt, **kwargs)
        else:
            raise ValueError(f"Unknown backend {backend}")
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def set_state(self, state):
        return self.env.set_state(state)

    def step(self, action):
        return self.env.step(action)

class PendulumEnvDiscont(gym.Wrapper):
    env: Union[PendulumEnv_NP, PendulumEnv_Torch]
    def __init__(self, l=1.0, g=9.81, dt=0.02, backend="numpy", **kwargs):
        if backend == "numpy":
            env = PendulumEnv_NP(l, g, dt, discontinuity=True, **kwargs)
        elif backend == "torch":
            env = PendulumEnv_Torch(l, g, dt, **kwargs)
        else:
            raise ValueError(f"Unknown backend {backend}")
        super().__init__(env)

    def reset(self):
        return self.env.reset()

    def set_state(self, state):
        return self.env.set_state(state)

    def step(self, action):
        return self.env.step(action)
