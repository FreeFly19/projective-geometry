import time

import numpy as np
import open3d as o3d
from open3d.cuda.pybind.geometry import PointCloud
import cv2

f = 0.6


def render_points(points, camera_position):
    rendered_img = np.zeros((256, 256))

    camera_matrix = np.array([
        [f, 0, 0, -camera_position[0]],
        [0, f, 0, -camera_position[1]],
        [0, 0, 1, -camera_position[2]],
        [0, 0, 0, 1],
    ])

    sensor_to_pixels = np.array([
        [rendered_img.shape[1], 0, rendered_img.shape[1] / 2, 0],
        [0, -rendered_img.shape[0], rendered_img.shape[0] / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    projection_matrix = sensor_to_pixels @ camera_matrix

    projection_space_points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)

    pixel_point = projection_matrix @ projection_space_points

    pixel_point = pixel_point[:, pixel_point[2] > 0]

    pixel_point = pixel_point / pixel_point[2]

    pixel_point = pixel_point.astype(int).T

    x, y = pixel_point[:, 0], pixel_point[:, 1]

    valid_mask = (x >= 0) & (x < rendered_img.shape[1]) & (y >= 0) & (y < rendered_img.shape[0])

    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    rendered_img[y_valid, x_valid] = 255

    cv2.imshow("img", rendered_img)
    cv2.waitKey(10)


pcd: PointCloud = o3d.io.read_point_cloud("data/teapot.ply")

points = np.array(pcd.points)
# points -= points.mean()
points = points / points.max()
points = points.T

camera_position = np.array([0, 0, -3])

start_time = time.time()

while True:
    ellapsed_time = time.time() - start_time

    alpha = ellapsed_time

    xz_rot = np.array([
        [np.cos(alpha) , 0, np.sin(alpha)],
        [       0      , 1,       0     ],
        [-np.sin(alpha), 0, np.cos(alpha)]
    ])

    points_in_world_coords = xz_rot @ points
    render_points(points_in_world_coords, camera_position)
