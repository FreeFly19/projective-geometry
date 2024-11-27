import time

import numpy as np
import open3d as o3d
import cv2

f = 0.8


def draw_line(p1, p2, img):
    # TODO: rewrite with
    y_offset = abs(p1[1] - p2[1])
    x_offset = abs(p1[0] - p2[0])

    if x_offset > y_offset:
        x_step = 1 if p1[0] < p2[0] else -1
        y_step = (p2[1] - p1[1]) / x_offset if x_offset != 0 else 0
        y = p1[1]

        for x in range(p1[0], p2[0] + 1, x_step):
            y += y_step
            if y > 0 and y < img.shape[0] and x > 0 and x < img.shape[1]:
                img[int(y)][int(x)] = 255
    else:
        y_step = 1 if p1[1] < p2[1] else -1
        x_step = (p2[0] - p1[0]) / y_offset if y_offset != 0 else 0
        x = p1[0]

        for y in range(p1[1], p2[1] + 1, y_step):
            x += x_step
            if y > 0 and y < img.shape[0] and x > 0 and x < img.shape[1]:
                img[int(y)][int(x)] = 255


def render_img(points, camera_position):
    rendered_img = np.zeros((512, 512))

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

    projected_point = projection_matrix @ projection_space_points

    point_ahead_of_camera = projected_point[2] > 0

    pixel_point = (projected_point / projected_point[2]).astype(int).T

    x, y = pixel_point[:, 0], pixel_point[:, 1]

    valid_mask = (x >= 0) & (x < rendered_img.shape[1]) & (y >= 0) & (y < rendered_img.shape[0]) & point_ahead_of_camera

    rendered_img[(y[valid_mask]), (x[valid_mask])] = 255

    for t in triangles:
        p1 = pixel_point[t[0]]
        p2 = pixel_point[t[1]]
        p3 = pixel_point[t[2]]

        if valid_mask[t[0]] or valid_mask[t[1]] or valid_mask[t[2]]:
            draw_line(p1, p2, rendered_img)
            draw_line(p2, p3, rendered_img)
            draw_line(p3, p1, rendered_img)

    return rendered_img.astype(np.uint8)


pcd = o3d.io.read_triangle_mesh("data/teapot.ply")

points = np.array(pcd.vertices)
triangles = np.array(pcd.triangles)
# points -= points.mean()
points = points / points.max()
points = points.T

camera_position = np.array([0, 0, -2])

start_time = time.time()

record = True

try:
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    writer = cv2.VideoWriter('video.mp4', fourcc, 30.0, (512, 512)) if record else None

    while True:
        ellapsed_time = time.time() - start_time

        alpha = ellapsed_time

        xz_rot = np.array([
            [np.cos(alpha) , 0, np.sin(alpha)],
            [       0      , 1,       0     ],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])

        points_in_world_coords = xz_rot @ points
        img = render_img(points_in_world_coords, camera_position)
        cv2.imshow("img", img)
        cv2.waitKey(10)
        if record:
            writer.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


except KeyboardInterrupt:
    if record:
        writer.release()