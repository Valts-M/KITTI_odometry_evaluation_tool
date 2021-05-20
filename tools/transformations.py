from math import *
import numpy as np

direction_dict = {
    'X': 0,
    'Y': 1,
    'Z': 2
}

roll = 0.0
pitch = 0.0
yaw = 0.0


def rot_x(angle):
    x_rot_matrix = np.array([
        [1, 0, 0, 0],
        [0, cos(angle), sin(angle), 0],
        [0, sin(angle), cos(angle), 0],
        [0, 0, 0, 1]
    ])
    return x_rot_matrix


def rot_y(angle):
    y_rot_matrix = np.array([
        [cos(angle), 0, sin(angle), 0],
        [0, 1, 0, 0],
        [-sin(angle), 0, cos(angle), 0],
        [0, 0, 0, 1]
    ])
    return y_rot_matrix


def rot_z(angle):
    z_rot_matrix = np.array([
        [cos(angle), sin(angle), 0, 0],
        [sin(angle), cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return z_rot_matrix


def rot_xyz(x_angle, y_angle, z_angle):
    xyz_rot_matrix = np.matmul(rot_x(x_angle), rot_y(y_angle), rot_z(z_angle))
    return xyz_rot_matrix


def get_furthest_point(trajectory, direction):
    '''
    :returns: a 4x4 homogenous transformation matrix representing the furthest point
    '''
    furthest_distance = 0
    for i in range(len(trajectory)):
        point = trajectory[i]
        if fabs(point[direction, 3]) > furthest_distance:
            furthest_point = point
            furthest_distance = furthest_point[direction, 3]

    return furthest_point


def get_angles(trajectory):
    '''
    Gets the angle offsets between the furthest points on the map and the origin in the X and Z direction\n
    :returns: [roll, pitch, yaw] -> angles by which to rotate
    '''
    furthest_z_point = get_furthest_point(trajectory, direction_dict['Z'])
    furthest_x_point = get_furthest_point(trajectory, direction_dict['X'])

    global roll
    global pitch
    global yaw
    roll = atan2(furthest_z_point[direction_dict['Y'], 3], furthest_z_point[direction_dict['Z'], 3])
    pitch = atan2(furthest_x_point[direction_dict['Y'], 3], furthest_x_point[direction_dict['X'], 3])
    yaw = 0.0


def rotate_point_cloud(mapping_trajectory, map_points):
    '''
    Since the camera doesn't know it's initial orientation it's possible that the trajectory roll and pitch will be
    offset, so we need to rotate the map and trajectory so that the trajectory is in the XZ plane.\n
    Rotates both the trajectory and the map around the origin (0, 0, 0) so that the trajectory is in the XZ plane.
    Assumes that the mapping was done on a flat plane.\n
    :returns: rotated trajectory and map
    '''
    get_angles(mapping_trajectory)
    rot_matrix = rot_xyz(pitch, yaw, roll)

    # rotates the trajectory
    for i in range(len(mapping_trajectory)):
        pose = mapping_trajectory[i][:, 3]
        mapping_trajectory[i] = np.matmul(pose, rot_matrix, -1 * pose)

    for i in range(len(map_points)):
        point = np.block([map_points[i], 1])
        map_points[i] = np.matmul(point, rot_matrix, -1 * point)

    return mapping_trajectory, map_points


def rotate_trajectory(trajectory):
    rot_matrix = rot_xyz(pitch, yaw, roll)

    for i in range(len(trajectory)):
        pose = trajectory[i][:, 3]
        trajectory[i] = np.matmul(pose, rot_matrix, -1 * pose)

    return trajectory