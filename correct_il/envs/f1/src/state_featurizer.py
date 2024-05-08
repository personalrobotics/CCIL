import numpy as np

from env_wrapper import F1EnvWrapper

N_LIDAR = 60
N_CURVATURE_POINTS = 0 #10
CURVATURE_LOOKAHEAD = 0.5
USE_AOA = False


def downsample(arr, target_len):
    assert len(arr) >= target_len
    factor = len(arr) / target_len
    if factor.is_integer():
        return arr[::int(factor)]
    else:
        assert False, "Interpolated downsampling not supported. Implement it?"


def get_closest_idx(pose, centerline):
    points = centerline[:, :2]
    pos = pose[:-1].reshape(1, -1)
    dists = np.linalg.norm(points - pos, axis=1)
    closest_idx = np.argmin(dists)
    return closest_idx


def angle_to_centerline(pose: np.ndarray, centerline: np.ndarray):
    points = centerline[:, :2]
    closest_idx = get_closest_idx(pose, centerline)
    next_idx = (closest_idx + 1) % len(centerline)
    centerline_angle = np.arctan2(
        *(points[next_idx] - points[closest_idx])[::-1])
    raw_angle_diff = pose[2] - centerline_angle
    angle = np.arctan2(np.sin(raw_angle_diff), np.cos(raw_angle_diff))
    return angle


def cross(*args, **kwargs) -> np.ndarray:
    return np.cross(*args, **kwargs)


def menger_curvature_loop(points: np.ndarray):
    points = np.vstack([points[-1], points, points[0]])
    forward_vecs = points[2:] - points[1:-1]
    back_vecs = points[:-2] - points[1:-1]
    forward_vecs /= np.linalg.norm(forward_vecs, axis=1).reshape(-1, 1)
    back_vecs /= np.linalg.norm(back_vecs, axis=1).reshape(-1, 1)
    sins = cross(forward_vecs, back_vecs)
    back_to_fronts = points[2:] - points[:-2]
    curvatures = 2 * sins / np.linalg.norm(back_to_fronts, axis=1)
    return curvatures


def get_future_curvatures(pose: np.ndarray, centerline: np.ndarray, n_points, point_dist):
    idx = get_closest_idx(pose, centerline)

    points = centerline[:, :2]
    # rearrange to start at current point
    points = np.vstack([points[idx:], points[:idx]])
    dists = np.concatenate(
        [np.zeros(1), np.linalg.norm(points[:-1] - points[1:], axis=1)])
    cum_dists = np.cumsum(dists)

    sample_dists = np.linspace(0, point_dist * (n_points - 1), n_points)

    curvature_idxs = np.around(
        np.interp(sample_dists, cum_dists, np.arange(len(dists)))).astype(int)

    curvatures = menger_curvature_loop(points)
    return curvatures[curvature_idxs]


def transform_state_(centerline, state):
    idx = state["ego_idx"]
    pose = np.array([state[s][idx]
                    for s in ["poses_x", "poses_y", "poses_theta"]])
    pose[2] = np.mod(pose[2], 2 * np.pi)
    vel = np.array([state["linear_vels_x"][idx]])
    if USE_AOA:
        angle_of_attack = np.array([angle_to_centerline(pose, centerline)])
    else:
        angle_of_attack = np.array([])
    scans = downsample(state["scans"][idx], N_LIDAR)
    collision = np.array([state["collisions"][idx]])
    if N_CURVATURE_POINTS > 0:
        future_curvatures = get_future_curvatures(
            pose, centerline, N_CURVATURE_POINTS, CURVATURE_LOOKAHEAD)
    else:
        future_curvatures = np.array([])
    return np.concatenate([vel, angle_of_attack, scans, collision, future_curvatures])


def transform_state(env: F1EnvWrapper, state, **kwargs):
    return transform_state_(env.centerline, state)

def main():
    MAP = "maps/random/map0"
    # MAP=  "maps/circle"
    centerline = np.genfromtxt(
        f"{MAP}_centerline.csv", delimiter=",", dtype=np.float32)
    points = centerline[:, :2]
    curvatures = menger_curvature_loop(points)
    import yaml
    with open(f"{MAP}.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    origin = np.array(config["origin"][:2])

    import matplotlib.pyplot as plt
    from PIL import Image
    M_PER_PX = config["resolution"]
    _, ax = plt.subplots()
    track = Image.open(f"{MAP}.png")
    IMG_HEIGHT = track.size[1]
    ax.imshow(track, cmap="gray")

    def interpolate_color(value):
        scale_fac = 2
        if value < 0:
            scale_dir = np.array([0, -1, -1])
        else:
            scale_dir = np.array([-1, -1, 0])
        return np.clip(np.array([1, 1, 1]) + np.abs(value) * scale_fac * scale_dir, 0, 1)
    for point, curvature in zip(points, curvatures):
        px_x, px_y = (point - origin) / M_PER_PX
        color = interpolate_color(curvature)
        circle = plt.Circle((px_x, IMG_HEIGHT - px_y), 5, color=color)
        ax.add_patch(circle)
    plt.savefig("fig.png")


if __name__ == "__main__":
    main()