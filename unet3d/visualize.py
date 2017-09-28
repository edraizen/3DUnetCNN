import numpy as np
import mayavi.mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_v(volume):
    xx, yy, zz = np.where(volume == True)
    plot_voxels(xx,yy,zz)

def plot_voxels(xx,yy,zz, colors=None, grayscale=True):
    nodes = mayavi.mlab.points3d(xx, yy, zz, mode="cube", color=(0, 1, 0), scale_factor=1)

    if colors:
        nodes.mlab_source.dataset.point_data.scalars = colors

    mayavi.mlab.show()

def plot_full_volume(volume, grayscale=True):
    xx, yy, zz = np.indices((volume[0,...,0].shape))
    nodes = mayavi.mlab.points3d(xx, yy, zz, mode="cube", scale_factor=1, opacity=0.05)
    nodes.glyph.scale_mode = 'scale_by_vector'
    nodes.mlab_source.dataset.point_data.scalars = volume[0,...,0].flatten()

def plot_volume(volume):
    """Plots volume in 3D, interpreting the coordinates as voxels
    From: EnzyNet
    """
    # Initialization
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection = '3d')
    ax.set_aspect('equal')

    # Parameters
    len_vol = volume.shape[0]

    # Set position of the view
    ax.view_init(elev = 20, azim = 135)

    # Hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Plot
    plot_matrix(ax, volume)

    # Tick at every unit
    ax.set_xticks(np.arange(volume.shape[0]))
    ax.set_yticks(np.arange(volume.shape[1]))
    ax.set_zticks(np.arange(volume.shape[2]))

    # Min and max that can be seen
    ax.set_xlim(0, volume.shape[0]-1)
    ax.set_ylim(0, volume.shape[1]-1)
    ax.set_zlim(0, volume.shape[2]-1)

    # Clear grid
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Change thickness of grid
    ax.xaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.yaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.1

    # Change thickness of ticks
    ax.xaxis._axinfo["tick"]['linewidth'] = 0.1
    ax.yaxis._axinfo["tick"]['linewidth'] = 0.1
    ax.zaxis._axinfo["tick"]['linewidth'] = 0.1

    # Change tick placement
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.2
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.2
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.2
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.2

    plt.show()

def plot_cube_at(pos = (0,0,0), ax = None, color=1):
    """Plots a cube element at position pos
    From: EnzyNet
    """
    assert 0 <= color <= 1
    if ax != None:
        X, Y, Z = cuboid_data(pos)
        ax.plot_surface(X, Y, Z, color=(0,color,0), rstride=1, cstride=1, alpha=1)

def plot_matrix(ax, matrix):
    'Plots cubes from a volumic matrix'
    for i in xrange(matrix.shape[0]):
        for j in xrange(matrix.shape[1]):
            for k in xrange(matrix.shape[2]):
                if matrix[i,j,k] == 1:
                    #print "Plotting voxel at", i, j, k
                    plot_cube_at(pos = (i-0.5,j-0.5,k-0.5), ax = ax)

def cuboid_data(pos, size = (1,1,1)):
    """Gets coordinates of cuboid
    From: EnzyNet
    """
    # Gets the (left, outside, bottom) point
    o = [a - b / 2. for a, b in zip(pos, size)]

    # Get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]] for i in range(4)]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]

    return x, y, z
