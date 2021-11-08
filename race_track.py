import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

def create_map(size, padding, thickness=100, randomed_pts=30, color=(0, 255, 0)):
    """
    Creates random map for race track
    """
    np.random.seed(0)
    radius = thickness // 2
    points = np.random.rand(randomed_pts, 2)
    points[:, 0], points[:, 1] = points[:, 0] * size, points[:, 1] * size

    hull = ConvexHull(points)
    xs, ys = points[hull.vertices, 0] + padding, points[hull.vertices, 1] + padding

    padded_map = np.zeros((size + padding * 2, size + padding * 2, 3), dtype=np.uint8)
    for i in range(len(xs)):
        x, y = int(xs[i]), int(ys[i])
        padded_map = cv2.circle(padded_map, (y, x), radius, color, -1)

    vectors = []
    checkpoints = []
    angles = []
    for i in range(len(xs)):
        j = (i + 1) % len(xs)
        pt1 = (int(ys[i]), int(xs[i]))
        pt2 = (int(ys[j]), int(xs[j]))

        v = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
        uv = v / np.linalg.norm(v)
        angle = np.arctan2(uv[1], uv[0])

        checkpoints.append(pt1)
        vectors.append(uv)
        angles.append(angle)
        padded_map = cv2.line(padded_map, pt1, pt2, color, thickness)

    return padded_map, checkpoints[0], angles[0], vectors, checkpoints, angles


if __name__ == "__main__":
    race_track, start_point, angle, vectors, checkpoints, angles = create_map(
        size=800, padding=20, thickness=10
    )

    print(vectors[0], checkpoints[0], angles[0])
    print(vectors[1], checkpoints[1], angles[1])

    plt.imshow(race_track)
    plt.show()