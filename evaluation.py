#
# Copyright Qing Li (hello.qingli@gmail.com) 2018. All Rights Reserved.
#
# References: 1. KITTI odometry development kit: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
#             2. A Geiger, P Lenz, R Urtasun. Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite. CVPR 2012.
#

import glob
import argparse
import os, os.path
import numpy as np
import matplotlib.pyplot as plt

# choose other backend that not required GUI (Agg, Cairo, PS, PDF or SVG) when use matplotlib
plt.switch_backend('agg')
import matplotlib.backends.backend_pdf
import tools.transformations as tr
from tools.pose_evaluation_utils import quat_pose_to_mat


class trajEval():
    def __init__(self, config):
        assert os.path.exists(config.traj_dir), "Trajectorie path doesn't exist! path: {}".format(config.traj_dir)
        assert os.path.exists(config.map_dir), "Map path doesn't exist! path: {}".format(config.traj_dir)
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
        self.map_dir= config.map_dir

        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        print(self.trajectories)


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

    def loadPoses(self, file_name, toCameraCoord):
        '''
            Each line in the file should follow one of the following structures
            (1) idx pose(3x4 matrix in terms of 12 numbers)
            (2) pose(3x4 matrix in terms of 12 numbers)
        '''
        f = open(file_name, 'r')
        s = f.readlines()
        f.close()
        file_len = len(s)
        poses = {}
        frame_idx = 0
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

    def trajectoryDistances(self, poses):
        '''
            Compute the length of the trajectory
            poses dictionary: [frame_idx: pose]
        '''
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx) - 1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i + 1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        self.distance = dist[-1]
        return dist

    def lastFrameFromSegmentLength(self, dist, first_frame, len_):
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + len_):
                return i
        return -1

    def plot_xyz(self, seq, poses_ref, poses_pred, plot_path_dir):

        def traj_xyz(axarr, positions_xyz, style='-', color='black', title="", label="", alpha=1.0):
            """
                plot a path/trajectory based on xyz coordinates into an axis
                :param axarr: an axis array (for x, y & z) e.g. from 'fig, axarr = plt.subplots(3)'
                :param traj: trajectory
                :param style: matplotlib line style
                :param color: matplotlib color
                :param label: label (for legend)
                :param alpha: alpha value for transparency
            """
            x = range(0, len(positions_xyz))
            xlabel = "index"
            ylabels = ["$x$ (m)", "$y$ (m)", "$z$ (m)"]
            # plt.title('PRY')
            for i in range(0, 3):
                axarr[i].plot(x, positions_xyz[:, i], style, color=color, label=label, alpha=alpha)
                axarr[i].set_ylabel(ylabels[i])
                axarr[i].legend(loc="upper right", frameon=True)
            axarr[2].set_xlabel(xlabel)
            if title:
                axarr[0].set_title('XYZ')

        fig, axarr = plt.subplots(3, sharex="col", figsize=tuple([20, 10]))

        pred_xyz = np.array([p[:3, 3] for _, p in poses_pred.items()])
        traj_xyz(axarr, pred_xyz, '-', 'b', title='XYZ', label='Ours', alpha=1.0)
        if poses_ref:
            ref_xyz = np.array([p[:3, 3] for _, p in poses_ref.items()])
            traj_xyz(axarr, ref_xyz, '-', 'r', label='GT', alpha=1.0)

        name = "{}_xyz".format(seq)
        plt.savefig(plot_path_dir + "/" + name + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir + "/" + name + ".pdf")
        fig.tight_layout()
        pdf.savefig(fig)
        # plt.show()
        pdf.close()

    def plotPath_2D_3(self, seq, poses_gt, poses_result, plot_path_dir):
        '''
            plot path in XY, XZ and YZ plane
        '''
        fontsize_ = 10
        plot_keys = ["Ground Truth", "Ours"]
        start_point = [0, 0]
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'

        ### get the value
        if poses_gt:
            poses_gt = [(k, poses_gt[k]) for k in sorted(poses_gt.keys())]
            x_gt = np.asarray([pose[0, 3] for _, pose in poses_gt])
            y_gt = np.asarray([pose[1, 3] for _, pose in poses_gt])
            z_gt = np.asarray([pose[2, 3] for _, pose in poses_gt])
        poses_result = [(k, poses_result[k]) for k in sorted(poses_result.keys())]
        x_pred = np.asarray([pose[0, 3] for _, pose in poses_result])
        y_pred = np.asarray([pose[1, 3] for _, pose in poses_result])
        z_pred = np.asarray([pose[2, 3] for _, pose in poses_result])

        fig = plt.figure(figsize=(20, 6), dpi=100)
        ### plot the figure
        plt.subplot(1, 3, 1)
        ax = plt.gca()
        if poses_gt: plt.plot(x_gt, z_gt, style_gt, label=plot_keys[0])
        plt.plot(x_pred, z_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
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

        plt.subplot(1, 3, 2)
        ax = plt.gca()
        if poses_gt: plt.plot(x_gt, y_gt, style_gt, label=plot_keys[0])
        plt.plot(x_pred, y_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xlabel('x (m)', fontsize=fontsize_)
        plt.ylabel('y (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        plt.subplot(1, 3, 3)
        ax = plt.gca()
        if poses_gt: plt.plot(y_gt, z_gt, style_gt, label=plot_keys[0])
        plt.plot(y_pred, z_pred, style_pred, label=plot_keys[1])
        plt.plot(start_point[0], start_point[1], style_O, label='Start Point')
        plt.legend(loc="upper right", prop={'size': fontsize_})
        plt.xlabel('y (m)', fontsize=fontsize_)
        plt.ylabel('z (m)', fontsize=fontsize_)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

        png_title = "{}_path".format(seq)
        plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir + "/" + png_title + ".pdf")
        fig.tight_layout()
        pdf.savefig(fig)
        # plt.show()
        plt.close()

    def plotPath_3D(self, seq, poses_gt, poses_result, plot_path_dir):
        """
            plot the path in 3D space
        """
        from mpl_toolkits.mplot3d import Axes3D

        start_point = [[0], [0], [0]]
        fontsize_ = 8
        style_pred = 'b-'
        style_gt = 'r-'
        style_O = 'ko'

        poses_dict = {}
        poses_dict["Ours"] = poses_result
        if poses_gt:
            poses_dict["Ground Truth"] = poses_gt

        fig = plt.figure(figsize=(8, 8), dpi=110)
        ax = fig.gca(projection='3d')

        for key, _ in poses_dict.items():
            plane_point = []
            for frame_idx in sorted(poses_dict[key].keys()):
                pose = poses_dict[key][frame_idx]
                plane_point.append([pose[0, 3], pose[2, 3], pose[1, 3]])
            plane_point = np.asarray(plane_point)
            style = style_pred if key == 'Ours' else style_gt
            plt.plot(plane_point[:, 0], plane_point[:, 1], plane_point[:, 2], style, label=key)
        plt.plot(start_point[0], start_point[1], start_point[2], style_O, label='Start Point')

        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        zmean = np.mean(zlim)
        plot_radius = max([abs(lim - mean_)
                           for lims, mean_ in ((xlim, xmean),
                                               (ylim, ymean),
                                               (zlim, zmean))
                           for lim in lims])
        ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
        ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

        ax.legend()
        # plt.legend(loc="upper right", prop={'size':fontsize_}) 
        ax.set_xlabel('x (m)', fontsize=fontsize_)
        ax.set_ylabel('z (m)', fontsize=fontsize_)
        ax.set_zlabel('y (m)', fontsize=fontsize_)
        ax.view_init(elev=20., azim=-35)

        png_title = "{}_path_3D".format(seq)
        plt.savefig(plot_path_dir + "/" + png_title + ".png", bbox_inches='tight', pad_inches=0.1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(plot_path_dir + "/" + png_title + ".pdf")
        fig.tight_layout()
        pdf.savefig(fig)
        # plt.show()
        plt.close()

    # def eval(self, toCameraCoord):
    #     '''
    #         to_camera_coord: whether the predicted pose needs to be convert to camera coordinate
    #     '''
    #
    #     for traj in self.trajectories:
    #         pred_file_name = self.result_dir + '/{}_pred.txt'.format(traj)
    #         # pred_file_name = self.result_dir + '/{}.txt'.format(traj)
    #         gt_file_name = self.gt_dir + '/{}.txt'.format(traj)
    #         save_file_name = eva_seq_dir + '/{}.pdf'.format(traj)
    #         assert os.path.exists(pred_file_name), "File path error: {}".format(pred_file_name)
    #
    #         # ----------------------------------------------------------------------
    #         # load pose
    #         # if traj in self.seqs_with_gt:
    #         #     self.call_evo_traj(pred_file_name, save_file_name, gt_file=gt_file_name)
    #         # else:
    #         #     self.call_evo_traj(pred_file_name, save_file_name, gt_file=None)
    #         #     continue
    #
    #         poses_result = self.loadPoses(pred_file_name, toCameraCoord=toCameraCoord)
    #
    #         if not os.path.exists(eva_seq_dir): os.makedirs(eva_seq_dir)
    #
    #         if traj not in self.seqs_with_gt:
    #             self.calcSequenceErrors(poses_result, poses_result)
    #             print("\nSequence: " + str(traj))
    #             print('Distance (m): %d' % self.distance)
    #             print('Max speed (km/h): %d' % (self.max_speed * 3.6))
    #             self.plot_rpy(traj, None, poses_result, eva_seq_dir)
    #             self.plot_xyz(traj, None, poses_result, eva_seq_dir)
    #             self.plotPath_3D(traj, None, poses_result, eva_seq_dir)
    #             self.plotPath_2D_3(traj, None, poses_result, eva_seq_dir)
    #             continue
    #
    #         poses_gt = self.loadPoses(gt_file_name, toCameraCoord=False)
    #
    #         # ----------------------------------------------------------------------
    #         # compute sequence errors
    #         seq_err = self.calcSequenceErrors(poses_gt, poses_result)
    #         self.saveSequenceErrors(seq_err, eva_seq_dir + '/{}_error.txt'.format(traj))
    #
    #         total_err += seq_err
    #
    #         # ----------------------------------------------------------------------
    #         # Compute segment errors
    #         avg_segment_errs = self.computeSegmentErr(seq_err)
    #         avg_speed_errs = self.computeSpeedErr(seq_err)
    #
    #         # ----------------------------------------------------------------------
    #         # compute overall error
    #         ave_t_err, ave_r_err = self.computeOverallErr(seq_err)
    #         print("\nSequence: " + str(traj))
    #         print('Distance (m): %d' % self.distance)
    #         print('Max speed (km/h): %d' % (self.max_speed * 3.6))
    #         print("Average sequence translational RMSE (%):   {0:.4f}".format(ave_t_err * 100))
    #         print("Average sequence rotational error (deg/m): {0:.4f}\n".format(ave_r_err / np.pi * 180))
    #         with open(eva_seq_dir + '/%s_stats.txt' % traj, 'w') as f:
    #             f.writelines('Average sequence translation RMSE (%):    {0:.4f}\n'.format(ave_t_err * 100))
    #             f.writelines('Average sequence rotation error (deg/m):  {0:.4f}'.format(ave_r_err / np.pi * 180))
    #         ave_errs[traj] = [ave_t_err, ave_r_err]
    #
    #         # ----------------------------------------------------------------------
    #         # Ploting
    #         self.plot_rpy(traj, poses_gt, poses_result, eva_seq_dir)
    #         self.plot_xyz(traj, poses_gt, poses_result, eva_seq_dir)
    #         self.plotPath_3D(traj, poses_gt, poses_result, eva_seq_dir)
    #         self.plotPath_2D_3(traj, poses_gt, poses_result, eva_seq_dir)
    #         self.plotError_segment(traj, avg_segment_errs, eva_seq_dir)
    #         self.plotError_speed(traj, avg_speed_errs, eva_seq_dir)
    #
    #         plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLAM report generator')
    parser.add_argument('--traj_dir', type=str, default='./trajectories',
                        help='Path to all trajectories, that need to be evaluated')
    parser.add_argument('--result_dir', type=str, default='./results',
                        help='Directory path of storing the odometry results')
    parser.add_argument('--map_dir', type=str, default='./map',
                        help='Path to all trajectories, that need to be evaluated')
    parser.add_argument('--toCameraCoord', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Whether to convert the pose to camera coordinate')

    args = parser.parse_args()
    pose_eval = trajEval(args)
    # pose_eval.eval(toCameraCoord=args.toCameraCoord)  # set the value according to the predicted results
