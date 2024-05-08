import numpy as np


def speed_reward(state, action, next_state):
    COLLISION_WEIGHT = 5
    CENTRIPETAL_WEIGHT = 0.05
    SKID_WEIGHT = 0.5
    STEER_DELTA_WEIGHT = 0.1

    idx = next_state["ego_idx"]
    vel_term = next_state["linear_vels_x"][idx]
    if next_state["collisions"][idx]:
        collision_term = COLLISION_WEIGHT * np.hypot(next_state["linear_vels_x"][idx], next_state["linear_vels_y"][idx])
    else:
        collision_term = 0
    centripetal_term = CENTRIPETAL_WEIGHT * vel_term ** 2 * np.abs(np.tan(action[0, 0]))
    skid_term = SKID_WEIGHT * np.abs(next_state["linear_vels_y"][idx])
    steer_delta_term = STEER_DELTA_WEIGHT * np.abs(action[0, 0] - state["steer_angles"][idx])
    return vel_term - collision_term - centripetal_term - skid_term - steer_delta_term