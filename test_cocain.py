"""
Code for Section 6.2 of the paper:
Convex-Concave Backtracking for Inertial Bregman Proximal Gradient Algorithms in Non-Convex Optimization
by Mahesh Chandra Mukkamala, Peter Ochs, Thomas Pock and Shoham Sabach.
Link: https://arxiv.org/abs/1904.03537
Algorithm implemented: CoCaIn BPG
Contact: Mahesh Chandra Mukkamala (mukkamala@math.uni-sb.de)
"""

import numpy as np

from my_functions import *

# some fixed values for backward compatibility
# TODO: Maybe remove them later
breg_num = 1
fun_num = 1
abs_fun_num = 1


dim = 2 # dimension of the problme

# for spurious stationary point experiment
# Initialization of hyper-parameters
# in particular delta, varepsilon as in the paper
del_val = 0.15
eps_val = 0.00001

# upper L initialization
uL_est = 1/(1-del_val) + 1e-3

# lower L guess for each iteration
# can be anything something small
# higher means low inertia
# low means high inertia 
# hence lower L must be tuned according to the problem
lL_est = 1e-4*uL_est
lL_est_main = lL_est # redundant variable for later usage 


# creating arguments to automate the experiments
import argparse
parser = argparse.ArgumentParser(description='Plot Experiments')
parser.add_argument('--init_Ux', '--init_Ux', default=0,type=int,  dest='init_Ux') # x coordinate
parser.add_argument('--init_Uy', '--init_Uy', default=0,type=int,  dest='init_Uy') # y coordinate
args = parser.parse_args()

# initialization of iterate
U = np.array([args.init_Ux,args.init_Uy])
init_U = U

# previous U
prev_U = U

# maximum iterations
max_iter = 10000





def get_proj_log(u,l):
	# Proximal mapping for the function log(1+|x|)
	# check Section 6.2 of https://arxiv.org/abs/1904.03537
	# or https://arxiv.org/pdf/1303.4434.pdf

	if ((np.abs(u) - 1)**2 - 4*(l-np.abs(u))) >=0:
		new_x = [0, 0.5*((np.abs(u)-1) + np.sqrt(((np.abs(u) - 1)**2 - 4*(l-np.abs(u))))), 0.5*((np.abs(u)-1) - np.sqrt(((np.abs(u) - 1)**2 - 4*(l-np.abs(u))))) ]
		new_y= [0,0,0]
		for i in [0,1,2]:
			tw = max(new_x[i],0)
			new_y[i]= 0.5*((tw-np.abs(u))**2) + l*(np.log(1+tw))
		new_x = np.array(new_x)
		new_y = np.array(new_y)
		return new_x[np.argmin(new_y)]
	else:	
		return 0

def make_update(y,grad,uL_est):
	# computing full update with proximal mapping

	temp_p = y - (1/uL_est)*grad
	return np.array([get_proj_log(temp_p[0],(1/uL_est)),get_proj_log(temp_p[1],(1/uL_est))])



def find_gamma(U,prev_U,uL_est, lL_est):
	# gamma initial guess
	gamma = 1

	kappa = (del_val - eps_val)*(uL_est/(uL_est+lL_est)) # Inertial step coefficient
	y_U = U+ gamma*(U-prev_U) #extrapolation
	while (kappa*breg(prev_U,U,breg_num=breg_num)<breg(U, y_U,breg_num=breg_num)):
		# in the above condition we can also threshold rather than absolute condition

		gamma = gamma*0.9 # scaling parameter can be anything <1
		y_U = U+ gamma*(U-prev_U)
	return y_U, gamma


def do_lb_search( U, U1, uL_est,lL_est):
	y_U,gamma = find_gamma(U,U1,uL_est, lL_est) 

	while((abs_func( U, y_U, abs_fun_num = abs_fun_num, fun_num=fun_num)\
		-main_func( U, fun_num=fun_num)\
		-(lL_est*breg(U, y_U,  breg_num=breg_num)))>1e-7):
		# thresholding
		
		lL_est = (2)*lL_est # scaling parameter can be anything >1

		y_U,gamma = find_gamma(U,U1,uL_est, lL_est)

	return lL_est, y_U,gamma



def do_ub_search( y_U, uL_est):
	# make update step
	grad_u = grad( y_U, fun_num=fun_num)
	x_U = make_update(y_U, grad_u, uL_est)
	
	while((abs_func( x_U,y_U, abs_fun_num = abs_fun_num, fun_num=fun_num)\
		-main_func( x_U,  fun_num=fun_num)\
		+(uL_est*breg(x_U,y_U, breg_num=breg_num)))<-1e-7):
		# thresholding

		uL_est = (2)*uL_est # scaling parameter can be anything >1
		
		x_U = make_update(y_U, grad_u, uL_est)
		

	return uL_est, x_U


# tracking gamma
gamma_vals = [0]

# tracking upper L
uL_est_vals = [uL_est]

# tracking lower L
lL_est_vals = [uL_est]

# initial function value stored in temporary variable
temp = main_func( U, fun_num=fun_num)

func_vals = [temp]
lyapunov_vals = [temp] # tracking Lyapunov function value

U_vals = [init_U] # tracking the iterates to plot later

# certain time tracking essentials
import time
time_vals = np.zeros(max_iter+1)
time_vals[0] = 0 # can be initialized to something non-zero


for i in range(max_iter):
	st_time = time.time() # starting time
	
	# cocain algo in 3 steps
	# step 1: Governs inertia with Lower L
	lL_est, y_U,gamma = do_lb_search( U, prev_U,  uL_est,lL_est=lL_est_main)

	# step 2: store the current U for next iteration
	prev_U = U

	# step 3: Make the update with backtracking
	uL_est, U = do_ub_search( y_U, uL_est)

	temp = main_func( U, fun_num=fun_num) # compute function value
	print('function value is '+ str(temp)) 
	
	uL_est_vals = uL_est_vals + [uL_est] # track upper L
	lL_est_vals = lL_est_vals + [lL_est] # track upper L
	gamma_vals = gamma_vals + [gamma]    # track gamma
	U_vals = U_vals + [U]				 # track iterates

	
	# to see if there are any numerical issues
	if np.isnan(temp):
		raise

	func_vals = func_vals + [temp] 		# track function values

	# track Lyapunov function values
	lyapunov_vals = lyapunov_vals + [(1/uL_est)*temp+ del_val*breg( U, prev_U, breg_num=breg_num)]


	time_vals[i+1] = time.time() - st_time # ending time for each iteration

# storing the variables
filename = 'results/cocain_'+str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
np.savetxt(filename,np.c_[func_vals, lyapunov_vals, uL_est_vals, lL_est_vals, gamma_vals, time_vals])

# storing the iterates to plot the trajectory according the initialization
# which is useful for later comparisons.
filename_1 = 'results/cocain_trajectory_'+str(init_U)+'.txt'
np.savetxt(filename_1,U_vals)
