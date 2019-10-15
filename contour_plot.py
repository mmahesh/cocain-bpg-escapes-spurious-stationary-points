import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


import numpy as np

def f(x, y):
	# function of two variable
	# Check README file
	
	return 0.5*np.log(1+100*(np.abs(x-1)**2))+0.5*np.log(1+(100*np.abs(y-1)**2))\
				+ np.log(1+np.abs(x))+np.log(1+np.abs(y))

# creating the grid for plotting
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y) # obtain the function values over grid


# We need to create 6 plots
# i=0 reserved for contour plot
# i=0 reserved for contour plot
for i in range(6):
	plt.contour(X, Y, Z, 100, cmap='RdBu');
	plt.plot(0, 1, 'bh', fillstyle='none')
	plt.plot(1, 0, 'bh', fillstyle='none')
	plt.plot(1, 1, 'bh', fillstyle='none')
	plt.plot(0, 0, 'bh', fillstyle='none')

	if i>1:
		# to plot with various initializations
		# the array in the file name denote the initialization point

		if i==2:
			p1 = np.loadtxt('results/cocain_trajectory_[2 2].txt')
		elif i ==3:
			p1 = np.loadtxt('results/cocain_trajectory_[ 2 -2].txt')
		elif i==4:
			p1 = np.loadtxt('results/cocain_trajectory_[-2  2].txt')
		else:
			p1 = np.loadtxt('results/cocain_trajectory_[-2 -2].txt')

		plt.plot(p1[:,0],p1[:,1])
		plt.savefig('figures/example_contour_'+str(i)+'.pdf')
		plt.savefig('figures/example_contour_'+str(i)+'.png')
	else:
		# i=0 reserved for contour plot
		plt.savefig('figures/example_contour.pdf')
		plt.savefig('figures/example_contour.png')
	plt.clf()


# The following code is to visualize the loss function
# check the Function surface plot in README file
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
    linewidth=10, antialiased=False)
ax.view_init(73,30)
ax.zaxis.set_major_locator(LinearLocator(4))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$z$')

plt.savefig('figures/example_surface.pdf')
plt.savefig('figures/example_surface.png')
