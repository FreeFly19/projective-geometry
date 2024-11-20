import numpy as np
import open3d as o3d
from open3d.cuda.pybind.geometry import PointCloud
import cv2

f = 0.5

p = np.array([-3, -3, 5])

camera_matrix = np.array([
    [f, 0, 0],
    [0, f, 0],
    [0, 0, 1]
])

p_proj = camera_matrix @ p
p_proj = p_proj / p[2]
# p_proj = p_proj[:2]
print(p_proj)

rendered_img = np.zeros((64, 64))

sensor_to_pixels = np.array([
    [rendered_img.shape[1], 0, rendered_img.shape[1] / 2],
    [0, -rendered_img.shape[0], rendered_img.shape[0] / 2],
    [0, 0, 1],
])

pixel_point = sensor_to_pixels @ p_proj
pixel_point = pixel_point / pixel_point[2]

pixel_point = pixel_point.astype(int)

rendered_img[pixel_point[1], pixel_point[0]] = 255

res = cv2.resize(rendered_img, (640, 640))
cv2.imshow("img", res)
cv2.waitKey(99999)
cv2.waitKey(99999)










#
# pcd: PointCloud = o3d.io.read_point_cloud("data/monkey.ply")
#
# points = np.array(pcd.points)
#
# points = points / points.max()
#
#
#
#
#
#
# print(points)
