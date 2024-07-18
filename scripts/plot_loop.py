# coding=UTF-8
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def PlotPath(keyframes, loops):
    fig = plt.figure('keyframes')
    ax = Axes3D(fig)

    # opti2
    p03, = ax.plot(keyframes[:, 26], keyframes[:, 27], keyframes[:, 28], 'r-')

    # 确保各轴比例相同
    ax.set_zlim(-60.0, 60.0)

    # 绘制所有相关坐标值之间的直线
    for index_pair in loops:
        start_idx = index_pair[0]
        end_idx = index_pair[1]
        start_point = [keyframes[start_idx, 26], keyframes[start_idx, 27], keyframes[start_idx, 28]]
        end_point = [keyframes[end_idx, 26], keyframes[end_idx, 27], keyframes[end_idx, 28]]
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2]], [end_point[2]], 'g-')
    plt.legend(['keyframes', 'loop'])
    plt.show()



def LoadMappingTxt(keyframes_file, loops_file):
    keyframes = np.loadtxt(keyframes_file)
    loops = np.loadtxt(loops_file)
    loops = loops[:, :2].astype(int)
    PlotPath(keyframes, loops)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Please input vaild param !!!')
        exit(1)
    else:
        keyframes = sys.argv[1]
        loop = sys.argv[2]
        LoadMappingTxt(keyframes, loop)
        exit(1)
        exit(1)

