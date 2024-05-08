import gym
import numpy as np
import yaml
import os
import matplotlib

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "true"
import pygame

def create_colors(n_colors):
    hsv = np.stack([np.linspace(0, 1, n_colors), np.full(n_colors, 1), np.full(n_colors, 1)], axis=-1)
    assert hsv.shape[-1] == 3
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    rgb_int = (rgb * 255).astype(int)
    return rgb_int

def get_centerline(map_name):
    centerline_path = map_name + '_centerline.csv'
    centerline = np.genfromtxt(centerline_path, delimiter=',', dtype=np.float32)
    return centerline

def get_closest_idx(pose, centerline):
    points = centerline[:, :2]
    pos = pose[:-1].reshape(1, -1)
    dists = np.linalg.norm(points - pos, axis=1)
    closest_idx = np.argmin(dists)
    return closest_idx

def get_centerline_pose(pose, centerline):
    idx = get_closest_idx(pose, centerline)
    next_idx = (idx + 1) % len(centerline)

    to_next = centerline[next_idx, :2] - centerline[idx, :2]
    to_next_norm = to_next / np.linalg.norm(to_next)
    to_car = pose[:2] - centerline[idx, :2]
    t = np.dot(to_car, to_next_norm) / np.linalg.norm(to_next)
    assert t < 1

    dx, dy = centerline[next_idx, :2] - centerline[idx, :2]
    p_theta = np.arctan2(dy, dx)

    if t >= 0:
        x, y = centerline[idx, :2] + t * to_next
        next_next_idx = (next_idx + 1) % len(centerline)
        next_theta = np.arctan2(*(centerline[next_next_idx, :2] - centerline[next_idx, :2])[::-1])
        theta = p_theta + t * np.arctan2(np.sin(next_theta - p_theta), np.cos(next_theta - p_theta))
        return np.array([x, y, theta])
    else:
        p, prev_p = centerline[idx], centerline[idx-1]
        to_prev = prev_p[:2] - p[:2]
        to_prev_norm = to_prev / np.linalg.norm(to_prev)
        prev_theta = np.arctan2(-to_prev[1], -to_prev[0])
        t = np.dot(to_car, to_prev_norm) / np.linalg.norm(to_prev)
        assert t < 1
        theta = p_theta + t * np.arctan2(np.sin(prev_theta - p_theta), np.cos(prev_theta - p_theta))
        x, y = p[:2] + t * to_prev
        return np.array([x, y, theta])


class UnevenSignedActionRescale(gym.ActionWrapper):
    """
    Scales between two ranges containing zero, unevenly between negatives and positives.
    This maps zero to zero.
    """
    def __init__(self, env, low, high):
        super().__init__(env)
        low = np.broadcast_to(low, env.action_space.shape).astype(np.float32)
        high = np.broadcast_to(high, env.action_space.shape).astype(np.float32)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.pos_scale = env.action_space.high / self.action_space.high
        self.neg_scale = env.action_space.low / self.action_space.low

    def action(self, act):
        pos_mask = act >= 0
        act[pos_mask] *= self.pos_scale[pos_mask]
        act[~pos_mask] *= self.neg_scale[~pos_mask]
        return act

class F1EnvWrapper(gym.Wrapper):
    env: gym.Env
    def __init__(self, env, init_state_supplier, state_featurizer, reward_fn, action_repeat=5):
        super().__init__(env)
        steer_min = env.params["s_min"]
        steer_max = env.params["s_max"]
        vel_min = env.params["v_min"]
        vel_max = env.params["v_max"]
        centerline_path = env.map_name + '_centerline.csv'

        self.init_state_supplier = init_state_supplier
        self.action_repeat = action_repeat
        self.reward_fn = reward_fn
        self.state_featurizer = state_featurizer
        self.curr_state = None
        self.screen = None
        self.font = None
        self.map_img = None
        self.map_cfg = None
        self.centerline = np.genfromtxt(centerline_path, delimiter=',', dtype=np.float32)
        if env.num_agents > 1:
            self.car_colors = create_colors(env.num_agents)
        else:
            self.car_colors = [(255, 0, 0)]

        self.action_space = gym.spaces.Box(np.array([steer_min, vel_min]), np.array([steer_max, vel_max]), dtype=np.float32)
        obs_shape = self._transform_state(env.reset(init_state_supplier(self))[0]).shape
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_shape, dtype=np.float32)

    def _transform_state(self, *args, **kwargs):
        return self.state_featurizer(self, *args, **kwargs)

    def _augment_curr_state(self, curr_state):
        """augments state dict in-place and returns it"""
        slip_angles = np.empty(self.env.num_agents)
        steer_angles = np.empty(self.env.num_agents)
        for i in range(self.env.num_agents):
            agent_state = self.env.sim.agents[i].state
            slip_angles[i] = agent_state[6]
            signed_speed = agent_state[3]
            curr_state["linear_vels_x"][i] = signed_speed * np.cos(slip_angles[i])
            curr_state["linear_vels_y"][i] = signed_speed * np.sin(slip_angles[i])
            steer_angles[i] = agent_state[2]
        curr_state["slip_angles"] = slip_angles
        curr_state["steer_angles"] = steer_angles
        return curr_state

    def step(self, action):
        action = np.expand_dims(action, axis=0)
        for _ in range(self.action_repeat):
            next_state, _, done, info = self.env.step(action)
            if done:
                break
        features = self._transform_state(next_state, prev_state=self.curr_state, prev_action=action)
        reward = self.reward_fn(self.curr_state, action, next_state)
        self.curr_state = self._augment_curr_state(next_state)
        return features, reward, done, info
    
    def reset(self):
        init_state = self.init_state_supplier(self)
        next_state, *_ = self.env.reset(init_state)
        self.curr_state = self._augment_curr_state(next_state)
        return self._transform_state(self.curr_state)
    
    def get_raw_state(self):
        return self.curr_state

    def _pos_m_to_px(self, screen, pos):
        origin = np.array(self.map_cfg["origin"][:2])
        m_per_px = self.map_cfg["resolution"]
        pos_px = np.around((pos - origin) / m_per_px).astype(int)
        pos_px[1] = screen.get_height() - pos_px[1]
        return pos_px

    def _draw_car(self, screen, pose, color):
        m_per_px = self.map_cfg["resolution"]
        pos_px = self._pos_m_to_px(screen, pose[:-1])
        car_len_px = self.env.params["length"] / m_per_px
        car_width_px = self.env.params["width"] / m_per_px
        theta = pose[-1]
        rotmat = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        cob = np.array([[1., 0.], [0., -1.]]) # change of basis required to invert y axis
        car_poly = [
            pos_px + cob @ rotmat @ np.array([car_len_px/2, 0]),
            pos_px + cob @ rotmat @ np.array([-car_len_px/2, -car_width_px/2]),
            pos_px + cob @ rotmat @ np.array([-car_len_px/2, car_width_px/2])
        ]
        car_poly = [np.around(x).astype(int) for x in car_poly]
        pygame.draw.polygon(screen, color, car_poly)

    def render(self, mode="human", use_default=False):
        if use_default or (mode != "human" and mode != "human_zoom"):
            return super().render(mode)
        pygame.init()
        if self.font is None:
            self.font = pygame.font.SysFont(None, 24)
        if self.map_img is None:
            self.map_img = pygame.image.load(self.env.map_name + self.env.map_ext)
        if self.map_cfg is None:
            with open(self.env.map_name + ".yaml", "r") as f:
                self.map_cfg = yaml.load(f, Loader=yaml.FullLoader)
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.map_img.get_width(), self.map_img.get_height()))
        map_screen = pygame.Surface(self.map_img.get_size())
        map_screen.blit(self.map_img, (0, 0))
        if mode == "human":
            for i in range(self.env.num_agents):
                pose = np.array([self.curr_state[s][i] for s in ["poses_x", "poses_y", "poses_theta"]])
                self._draw_car(map_screen, pose, self.car_colors[i])
            scale_fac = min(self.screen.get_width() / map_screen.get_width(), self.screen.get_height() / map_screen.get_height())
            map_screen = pygame.transform.rotozoom(map_screen, 0, scale_fac)
            # map_screen = pygame.transform.smoothscale(map_screen, self.screen.get_size())
            self.screen.blit(map_screen, (0, 0))
        else:
            def blitRotate(surf, image: pygame.Surface, pos, originPos, angle):
                image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
                offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
                rotated_offset = offset_center_to_pivot.rotate(-angle)
                rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)
                rotated_image = pygame.transform.rotate(image, angle)
                rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)
                surf.blit(rotated_image, rotated_image_rect)
                def mapper(p: pygame.Vector2):
                    rel_to_center = p - pygame.Vector2(image.get_size())/2
                    rel_to_center_rot = rel_to_center.rotate(-angle)
                    return rel_to_center_rot + rotated_image_center
                return mapper
            pose = np.array([self.curr_state[s][0] for s in ["poses_x", "poses_y", "poses_theta"]])
            centerline_pose = get_centerline_pose(pose, self.centerline)
            centerline_pos_px = pygame.Vector2(*self._pos_m_to_px(map_screen, centerline_pose[:-1]))
            intermediate = pygame.Surface([x//2 for x in self.screen.get_size()])
            SCALE = 4
            inter_center = pygame.Vector2(*self.screen.get_size())/(2*SCALE)
            rot_angle = 90 - centerline_pose[2]*180/np.pi
            mapper_fn = blitRotate(intermediate, map_screen, inter_center, centerline_pos_px, rot_angle)
            intermediate = pygame.transform.rotozoom(intermediate, 0, SCALE)

            car_pos_px = pygame.Vector2(*self._pos_m_to_px(map_screen, pose[:-1]))
            car_pos_px = mapper_fn(car_pos_px) * SCALE
            
            m_per_px = self.map_cfg["resolution"]
            car_len_px = self.env.params["length"] / m_per_px
            car_width_px = self.env.params["width"] / m_per_px
            car_theta = pose[-1] - centerline_pose[-1]
            rot_mat = np.array([[np.cos(car_theta), np.sin(car_theta)], [-np.sin(car_theta), np.cos(car_theta)]])
            car_poly = [
                car_pos_px + SCALE * rot_mat @ np.array([0, -car_len_px/2]),
                car_pos_px + SCALE * rot_mat @ np.array([-car_width_px/2, car_len_px/2]),
                car_pos_px + SCALE * rot_mat @ np.array([car_width_px/2, car_len_px/2])
            ]
            car_poly = [np.around(x).astype(int) for x in car_poly]
            pygame.draw.polygon(intermediate, (255, 0, 0), car_poly)
            self.screen.fill((255, 255, 255))
            self.screen.blit(intermediate, (0,0))
        img = self.font.render(f"{self.curr_state['linear_vels_x'][0]:.2f} m/s", True, (0, 0, 255))
        self.screen.blit(img, (20, 20))
        pygame.display.flip()

