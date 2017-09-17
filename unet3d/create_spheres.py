import numpy as np
import mayavi.mlab

def create_spheres(num_spheres, shape=(144, 144, 144), border=50, min_r=5, max_r=15):
	"""Create randomly placed and randomy sized spheres inside of a grid
	"""
	volume = np.random.random(shape)
	labels = np.zeros(shape)

	for i in xrange(num_spheres):
		#Define random center of sphere and radius
		center = [np.random.randint(border, edge-border) for edge in shape]
		r = np.random.randint(min_r, max_r)

		y, x, z = np.ogrid[-center[0]:shape[0]-center[0], -center[1]:shape[1]-center[1], -center[2]:shape[2]-center[2]]
		m = x*x + y*y + z*z < r*r
		indices = np.where(m==True)
		print indices
		volume[indices] = 1
		labels[indices] = 1

	return volume, labels
		

def plot_voxels(volume):
	xx, yy, zz = np.where(volume == True)
	mayavi.mlab.points3d(xx, yy, zz,
	                     mode="cube",
	                     color=(0, 1, 0),
	                     scale_factor=1)
	mayavi.mlab.show()