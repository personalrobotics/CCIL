import numpy as np

def create_constant_sampler():
    def sample_state(env):
        point = env.centerline[0]
        next = env.centerline[1]
        dx, dy = next[:2] - point[:2]
        angle = np.arctan2(dy, dx)
        return np.array([point[0], point[1], angle]).reshape((1, -1))

    return sample_state

def create_centerline_sampler(seed=None):
    rng = np.random.default_rng(seed=seed)
    angle_delta = 15 * np.pi / 180  # 15 degrees in radians
    xy_pct_delta = 0.2  # up to 20 percent off center
    
    def sample_state(env):
        size = env.centerline.shape[0]
        idx = rng.choice(size)
        point = np.copy(env.centerline[idx])
        next = env.centerline[(idx + 1) % size]
        angle = np.arctan2(next[1] - point[1], next[0] - point[0])
        delta_x = abs(point[3] * np.sin(angle) * xy_pct_delta)
        delta_y = abs(point[3] * np.cos(angle) * xy_pct_delta)
        angle += rng.random() * 2 * angle_delta - angle_delta
        angle = np.remainder(angle, 2 * np.pi)
        point[0] += rng.random() * 2 * delta_x - delta_x
        point[1] += rng.random() * 2 * delta_y - delta_y
        return np.array([point[0], point[1], angle]).reshape((1, 3))

    return sample_state

if __name__ == "__main__":
    import gym
    from env_wrapper import F1EnvWrapper
    from state_featurizer import transform_state
    import pygame
    env = gym.make("f110_gym:f110-v0", num_agents=100, map="maps/circle")

    sampler = create_centerline_sampler()
    def reset(env):
        states = []
        for _ in range(100):
            states.append(sampler(env))
        ret = np.concatenate(states, axis=0)
        return ret

    env = F1EnvWrapper(env, reset, transform_state, lambda *_: 0.0)
    env.reset()
    env.render("human")
    pygame.image.save(env.screen, "map.png")