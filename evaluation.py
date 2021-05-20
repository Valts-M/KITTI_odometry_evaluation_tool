import glob
import argparse
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from math import *
from tools.transformations import rotate_point_cloud
import tools.file_loader as fl


class ReportGenerator(object):
    def __init__(self, config):
        super(ReportGenerator).__init__()
        assert os.path.exists(config.traj_dir), "Trajectorie path doesn't exist! path: {}".format(config.traj_dir)
        assert os.path.exists(config.map_dir), "Map path doesn't exist! path: {}".format(config.map_dir)

        map_files = glob.glob(config.map_dir + '/*.msg')
        if len(map_files) > 1:
            print('Too many map files in the map folder! Should only be 1!')
            exit(1)
        elif len(map_files) == 0:
            print('No map files found in the map directory {}!'.format(config.map_dir))
            exit(1)
        self.map_file = map_files[0]

        map_trajectories = glob.glob(config.map_dir + '/*.txt')
        if len(map_trajectories) > 1:
            print('Too many map trajectories in the map directory {}! Should only be 1!'.format(config.map_dir))
            exit(1)
        elif len(map_trajectories) == 0:
            print('No trajectory files found in the map directory {}!'.format(config.map_dir))
            exit(1)
        self.map_trajectory = map_trajectories[0]

        traj_files = glob.glob(config.traj_dir + '/*.txt')
        traj_files = [os.path.split(f)[1] for f in traj_files]
        if len(traj_files) == 0:
            print("No trajectories found in trajectory path!")
            exit(1)
        self.trajectories = [os.path.splitext(f)[0] for f in traj_files]
        self.traj_dir = config.traj_dir
        self.result_dir = config.result_dir

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

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
        font_size = 10
        start_point = [0, 0]
        style_start_point = 'ko'
        style_dict = {
            0: "-b",
            1: "-r",
            2: "-g",
            3: "-p",
            4: "-y"
        }

        fig = plt.figure(figsize=(20, 6), dpi=100)
        ax = plt.gca()
        plt.xlabel('x (m)', fontsize=font_size)
        plt.ylabel('z (m)', fontsize=font_size)
        plt.plot(start_point[0], start_point[1], style_start_point, label='Start Point')

        #  load each trajectory and plot it
        for i in range(len(trajectories)):
            poses = fl.loadPoses("{}/{}.txt".format(trajectory_dir, trajectories[i]), toCameraCoord)
            poses, map_points = rotate_point_cloud(poses, map_points)
            x_traj = [pose[0] for pose in poses]
            z_traj = [pose[2] for pose in poses]

            plt.plot(x_traj, z_traj, style_dict[i], label=trajectories[i])

        #  plot the map as a scatter plot of points
        x_map = [point[0] for point in map_points]
        z_map = [point[2] for point in map_points]
        # map_gradiant = self.get_map_gradiant(culled_map)
        #  gets each points color according to its y value on the map
        point_color = self.get_point_color(map_points)
        plt.scatter(x_map, z_map, s=0.4, c=point_color, cmap="RdYlGn")

        # set the range of x and y
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

        plt.legend(loc="upper right", prop={'size': font_size})
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
        
        map_points = fl.loadMap(self.map_file)
        self.plot_trajectories_2d(self.trajectories, self.traj_dir, self.result_dir, map_points, toCameraCoord=toCameraCoord)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLAM report generator')
    parser.add_argument('--traj_dir', type=str, default='./trajectories',
                        help='Path to directory that contains the trajectories that are to be evaluated')
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Path to directory where the results will be stored')
    parser.add_argument('--map_dir', type=str, default='./map/',
                        help='Path to directory that contains the map.msg file and it\'s corresponding trajectory')
    parser.add_argument('--toCameraCoord', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Whether to convert the pose to camera coordinate')

    args = parser.parse_args()
    report_gen = ReportGenerator(args)
    report_gen.generate_report(toCameraCoord=args.toCameraCoord)
