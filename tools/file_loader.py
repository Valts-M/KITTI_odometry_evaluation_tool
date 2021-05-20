import numpy as np
import msgpack


def toCameraCoord(pose_mat):
    '''
        Convert the pose of lidar coordinate to camera coordinate
    '''
    R_C2L = np.array([[0, 0, 1, 0],
                      [-1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1]])
    inv_R_C2L = np.linalg.inv(R_C2L)
    R = np.dot(inv_R_C2L, pose_mat)
    rot = np.dot(R, R_C2L)
    return rot


def cull_map(map_points):
    culled_map = []
    for i in range(len(map_points)):
        point = map_points[i]
        if 0.1 > point[1] > -0.1:
            culled_map.append(point)

    return culled_map


def loadMap(map_path):
    '''
    Loads the map for use in the report generation
    :param: map_path path to the map.msg file
    :returns: map
    '''
    # Read file as binary and unpack data using MessagePack library
    with open(map_path, "rb") as f:
        data = msgpack.unpackb(f.read(), use_list=False, raw=False)

    # The point data is tagged "landmarks"
    landmarks = data["landmarks"]

    map_points = []
    for _, point in landmarks.items():
        map_points.append(np.asarray(point["pos_w"]))

    map_points = cull_map(map_points)

    return map_points


def loadPoses(file_name, toCameraCoord):
    '''
        Each line in the file should follow one of the following structures
        (1) idx pose(3x4 matrix in terms of 12 numbers)
        (2) pose(3x4 matrix in terms of 12 numbers)
    '''
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = []

    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split()]
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row * 4 + col]
        if toCameraCoord:
            poses.append(toCameraCoord(P))
        else:
            poses.append(P)
    return poses

