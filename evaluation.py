import glob
import argparse
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import msgpack
import matplotlib.backends.backend_pdf
from math import *


class ReportGenerator(object):
    def __init__(self, config):
        super(ReportGenerator).__init__()
        assert os.path.exists(config.traj_dir), "Trajectorie path doesn't exist! path: {}".format(config.traj_dir)
        assert os.path.exists(config.map_path), "Map path doesn't exist! path: {}".format(config.traj_dir)
        traj_files = glob.glob(config.traj_dir + '/*.txt')
        traj_files = [os.path.split(f)[1] for f in traj_files]
        if len(traj_files) == 0:
            print("No trajectories found in trajectory path!")
            exit(1)
        self.trajectories = [os.path.splitext(f)[0] for f in traj_files]

        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)
        self.traj_dir = config.traj_dir
        self.result_dir = config.result_dir
        self.map_path= config.map_path
        self.x = 0
        self.y = 1
        self.z = 2

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)


    def toCameraCoord(self, pose_mat):
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

    def loadMap(self, map_path):
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
        print("Point cloud has {} points.".format(len(landmarks)))
        map = {}

        loop_count = 0
        for id, point in landmarks.items():
            map[loop_count] = np.asarray(point["pos_w"])
            loop_count += 1

        return map

    def loadPoses(self, file_name, toCameraCoord):
        '''
            Each line in the file should follow one of the following structures
            (1) idx pose(3x4 matrix in terms of 12 numbers)
            (2) pose(3x4 matrix in terms of 12 numbers)
        '''
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        poses = {}
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split()]
            withIdx = int(len(line_split) == 13)
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row * 4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            if toCameraCoord:
                poses[frame_idx] = self.toCameraCoord(P)
            else:
                poses[frame_idx] = P
        return poses

    def get_furthest_point(self, trajectory, direction):
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

    def get_angles(self, trayectory):
        '''
        Gets the angle offsets between the furthest points on the map and the origin in the X and Z direction\n
        :returns: [roll, pitch, yaw] -> angles by which to rotate
        '''
        furthest_z_point = self.get_furthest_point(trayectory, self.z)
        furthest_x_point = self.get_furthest_point(trayectory, self.x)

        roll = atan2(furthest_z_point[self.y, 3], furthest_z_point[self.z, 3])
        pitch = atan2(furthest_x_point[self.y, 3], furthest_x_point[self.x, 3])
        yaw = 0.0

        return [roll, pitch, yaw]

    @staticmethod
    def rot_x(angle):
        rot_x = np.array([
            [1,     0,              0,              0],
            [0,     cos(angle),     sin(angle),     0],
            [0,     sin(angle),     cos(angle),     0],
            [0,     0,              0,              1]
        ])
        return rot_x

    @staticmethod
    def rot_y(angle):
        rot_y = np.array([
            [cos(angle),        0,      sin(angle),     0],
            [0,                 1,      0,              0],
            [-sin(angle),       0,      cos(angle),     0],
            [0,                 0,      0,              1]
        ])
        return rot_y

    @staticmethod
    def rot_z(angle):
        rot_z = np.array([
             [cos(angle),   sin(angle),     0,      0],
             [sin(angle),   cos(angle),     0,      0],
             [0,            0,              1,      0],
             [0,            0,              0,      1]
        ])
        return rot_z

    def rot_xyz(self, x_angle, y_angle, z_angle):
        rot_xyz = np.matmul(np.matmul(self.rot_x(x_angle), self.rot_y(y_angle)), self.rot_z(z_angle))
        return rot_xyz

    def rotate_point_cloud(self, mapping_trajectory, map_points):
        '''
        Since the camera doesn't know it's initial orientation it's possible that the trajectory roll and pitch will be
        offset, so we need to rotate the map and trajectory so that the trajectory is in the XZ plane.\n
        Rotates both the trajectory and the map around the origin (0, 0, 0) so that the trajectory is in the XZ plane.
        Assumes that the mapping was done on a flat plane.\n
        :returns: rotated trajectory and map
        '''
        [roll, pitch, yaw] = self.get_angles(mapping_trajectory)
        rot_matrix = self.rot_xyz(pitch, yaw, roll)

        # rotates the trajectory
        for i in range(len(mapping_trajectory)):
            pose = mapping_trajectory[i][:, 3]
            mapping_trajectory[i] = np.matmul(pose, rot_matrix, -1*pose)

        for i in range(len(map_points)):
            point = np.block([map_points[i], 1])
            map_points[i] = np.matmul(point, rot_matrix, -1 * point)

        return mapping_trajectory, map_points

    def get_map_gradiant(self, map_points):
        max_y = max(map_points[:][1])
        min_y = fabs(min(map_points[:][1]))
        delta_y = max_y + min_y

        map_gradiant = []

        for i in range(len(map_points)):
            point = map_points[i]
            point[1] = (point[1] + min_y)/delta_y
            map_gradiant.append(point[1])

        return map_gradiant

    def get_point_color(self, map_points):
        color = []
        for point in map_points:
            if point[1] > 0.1:
                color.append(0)
            elif point[1] < -0.1:
                color.append(1)
            else:
                color.append(0.5)

        return color

    def plot_trajectories_2d(self, trajectories, trajectory_dir, results_dir, map_points, toCameraCoord):
        '''
            Plots the trajectories in the XZ plane and saves them to a png and pdf

            Params:
                trajectories - list of trajectory names
                trajectory_dir - directory that contains all the trajectories
                results_dir - directory that the results will be saved to
                map - point cloud of the enviorments
                toCameraCoord - do the poses need to be converted to the camera coordinate space
        '''
        fontsize_ = 10
        start_point = [0, 0]
        style_O = 'ko'
        style_dict = {
            0: "-b",
            1: "-r",
            2: "-g",
            3: "-p",
            4: "-y"
        }

        fig = plt.figure(figsize=(20, 6), dpi=100)
        ax = plt.gca()
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')

        ### get the values
        for i in range(len(trajectories)):
            poses = self.loadPoses("{}/{}.txt".format(trajectory_dir, trajectories[i]), toCameraCoord)
            poses, map_points = self.rotate_point_cloud(poses, map_points)
            poses_result = [(k, poses[k]) for k in sorted(poses.keys())]
            x_traj = [pose[0] for _, pose in poses_result]
            z_traj = [pose[2] for _, pose in poses_result]

            ### plot the figure
            plt.plot(x_traj, z_traj, style_dict[i], label=trajectories[i])

        culled_map = []
        for i in range(len(map_points)):
            point = map_points[i]
            if 0.1 > point[1] > -0.1:
                culled_map.append(point)

        x_map = [point[0] for point in culled_map]
        z_map = [point[2] for point in culled_map]

        print('point cloud after culling: {} points'.format(len(culled_map)))
        # map_gradiant = self.get_map_gradiant(culled_map)
        point_color = self.get_point_color(culled_map)
        plt.scatter(x_map, z_map, s=0.4, c=point_color, cmap="RdYlGn")

        ### set the range of x and y
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        plot_radius = max([abs(lim - mean_)
                           for lims, mean_ in ((xlim, xmean),
                                               (ylim, ymean))
                           for lim in lims])
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        plt.legend(loc="upper right", prop={'size': fontsize_})
        png_title = "Trajectories"
        plt.savefig(results_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(results_dir + "/" + png_title + ".pdf")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.show()
        plt.close()

    def generate_report(self, toCameraCoord):
        '''
            to_camera_coord: whether the predicted pose needs to be convert to camera coordinate space
        '''

        map = self.loadMap(self.map_path)

        for traj in self.trajectories:
            traj_file_name = self.traj_dir + '/{}.txt'.format(traj)
            assert os.path.exists(traj_file_name), "File path error: {}".format(traj_file_name)

            #poses_result = self.loadPoses(traj_file_name, toCameraCoord=toCameraCoord)
            #self.plotCoverageMap(traj, self.result_dir, map)

        self.plot_trajectories_2d(self.trajectories, self.traj_dir, self.result_dir, map, toCameraCoord=toCameraCoord)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLAM report generator')
    parser.add_argument('--traj_dir', type=str, default='./trajectories',
                        help='Path to directory that contains the trajectories that are to be evaluated')
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Directory path of storing the odometry results')
    parser.add_argument('--map_path', type=str, default='./map/map.msg',
                        help='Path to the map msg file')
    parser.add_argument('--toCameraCoord', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Whether to convert the pose to camera coordinate')

    args = parser.parse_args()
    report_gen = ReportGenerator(args)
    report_gen.generate_report(toCameraCoord=args.toCameraCoord)
