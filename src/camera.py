import math
import numpy as np
from .utils import normalize


def calculate_camera_basis(yaw, pitch):
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    fx = math.sin(yaw_rad) * math.cos(pitch_rad)
    fy = math.sin(pitch_rad)
    fz = -math.cos(yaw_rad) * math.cos(pitch_rad)
    forward = np.array([fx, fy, fz], dtype=np.float32)
    forward = normalize(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = normalize(right)
    up = np.cross(right, forward)
    up = normalize(up)
    return forward, right, up
