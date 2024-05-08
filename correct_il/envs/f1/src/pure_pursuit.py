import numpy as np

def get_closest_idx(pos, centerline):
    points = centerline[:, :2]
    pos = pos.reshape(1, -1)
    dists = np.linalg.norm(points - pos, axis=1)
    closest_idx = np.argmin(dists)
    return closest_idx


class PurePursuit(object):
    def __init__(self, centerline: np.ndarray, lookahead: float=1.0, turn_kp: float=1.1, vel: float=1.0):
        self.centerline = centerline[:,:2]
        self.lookahead = lookahead
        self.turn_kp = turn_kp
        self.vel = vel
    
    def get_action(self, state: dict):
        ego_idx = state["ego_idx"]
        x = state["poses_x"][ego_idx]
        y = state["poses_y"][ego_idx]
        theta = state["poses_theta"][ego_idx]

        pos = np.array([x,y])

        closest_idx = get_closest_idx(pos, self.centerline)
        lookahead_point = None
        for i in range(1, len(self.centerline)):
            point = self.centerline[(closest_idx + i) % len(self.centerline)]
            if np.linalg.norm(point-pos) >= self.lookahead:
                lookahead_point = point
                break
        if lookahead_point is None:
            print("No lookahead found!")
            return np.array([0.0, 0.0])

        to_lookahead = lookahead_point - pos
        # local to global frame rot mat
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        to_lookahead_local = rot_mat.T @ to_lookahead
        angle_to_lookahead = np.arctan2(to_lookahead_local[1], to_lookahead_local[0])

        return np.array([self.turn_kp * angle_to_lookahead, self.vel])
        
        
