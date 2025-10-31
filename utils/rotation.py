import numpy as np


def rot6d_to_matrix(rot6d):
    x = normalize(rot6d[..., 0:3])
    y = normalize(rot6d[..., 3:6])
    a = normalize(x + y)
    b = normalize(x - y)
    x = normalize(a + b)
    y = normalize(a - b)
    z = normalize(np.cross(x, y))
    matrix = np.stack([x, y, z], axis=-2).swapaxes(-1, -2)
    return matrix


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)
