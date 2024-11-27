import math

import numpy as np
import open3d as o3d
from open3d.cuda.pybind.geometry import PointCloud
import cv2

f = 0.7

rendered_img = np.zeros((256, 256))


def clear():
    rendered_img[:, :] = 0


def render_point(p):
    camera_matrix = np.array([
        [f, 0, 0],
        [0, f, 0],
        [0, 0, 1]
    ])

    p_proj = camera_matrix @ p
    p_proj = p_proj / p[2]

    if (abs(p_proj[:2]) > 0.5).sum() > 0:
        return

    sensor_to_pixels = np.array([
        [rendered_img.shape[1], 0, rendered_img.shape[1] / 2],
        [0, -rendered_img.shape[0], rendered_img.shape[0] / 2],
        [0, 0, 1],
    ])

    pixel_point = sensor_to_pixels @ p_proj
    pixel_point = pixel_point / pixel_point[2]

    pixel_point = pixel_point.astype(int)

    rendered_img[pixel_point[1], pixel_point[0]] = 255


pcd: PointCloud = o3d.io.read_point_cloud("data/teapot.ply")

points = np.array(pcd.points)
points -= points.mean()
points = points / points.max()


while True:
    for alpha in np.arange(0, math.pi * 2, 0.05):
        clear()
        xz_rot = np.array([
            [np.cos(alpha) , 0, np.sin(alpha)],
            [       0      , 1,       0     ],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])

        for point in points:
            point = xz_rot @ point
            point[2] += 3
            render_point(point)

        # res = cv2.resize(rendered_img, (640, 640))
        res = rendered_img
        cv2.imshow("img", res)
        cv2.waitKey(10)
