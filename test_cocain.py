import numpy as np

from my_functions import *

breg_num = 1
fun_num = 1
abs_fun_num = 1

dim = 2

# for spurious sp exps
del_val = 0.15
eps_val = 0.00001
uL_est = 1/(1-del_val) + 1e-3
lL_est = 1e-4*uL_est

# creating arguments to automate the experiments
import argparse
parser = argparse.ArgumentParser(description='Plot Experiments')
parser.add_argument('--init_Ux', '--init_Ux', default=0,type=int,  dest='init_Ux') # x coordinate
parser.add_argument('--init_Uy', '--init_Uy', default=0,type=int,  dest='init_Uy') # y coordinate
args = parser.parse_args()

U = np.array([args.init_Ux,args.init_Uy])

init_U = U



alpha_val =  2

prev_U = U
max_iter = 10000

lL_est_main = lL_est




def get_proj_log(u,l):
	#https://arxiv.org/pdf/1303.4434.pdf
	
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
	temp_p = y - (1/uL_est)*grad
	return np.array([get_proj_log(temp_p[0],(1/uL_est)),get_proj_log(temp_p[1],(1/uL_est))])



def find_gamma(U,prev_U,uL_est, lL_est):
	gamma = 1
	kappa = (del_val - eps_val)*(uL_est/(uL_est+lL_est))
	y_U = U+ gamma*(U-prev_U)
	while (kappa*breg(prev_U,U,breg_num=breg_num)<breg(U, y_U,breg_num=breg_num)):
		gamma = gamma*0.9
		y_U = U+ gamma*(U-prev_U)
	return y_U, gamma


def do_lb_search( U, U1, uL_est,lL_est):
	y_U,gamma = find_gamma(U,U1,uL_est, lL_est)

	while((abs_func( U, y_U, abs_fun_num = abs_fun_num, fun_num=fun_num)\
		-main_func( U, fun_num=fun_num)\
		-(lL_est*breg(U, y_U,  breg_num=breg_num)))>1e-7):
		

		lL_est = (2)*lL_est

		y_U,gamma = find_gamma(U,U1,uL_est, lL_est)

	return lL_est, y_U,gamma



def do_ub_search( y_U, uL_est):
	# make update step
	grad_u = grad( y_U, fun_num=fun_num)
	x_U = make_update(y_U, grad_u, uL_est)
	
	while((abs_func( x_U,y_U, abs_fun_num = abs_fun_num, fun_num=fun_num)\
		-main_func( x_U,  fun_num=fun_num)\
		+(uL_est*breg(x_U,y_U, breg_num=breg_num)))<-1e-7):

		uL_est = (2)*uL_est
		
		x_U = make_update(y_U, grad_u, uL_est)
		

	return uL_est, x_U


gamma_vals = [np.sqrt(uL_est/(uL_est+lL_est))]
uL_est_vals = [uL_est]
lL_est_vals = [uL_est]

temp = main_func( U, fun_num=fun_num)

func_vals = [temp]
lyapunov_vals = [temp]

U_vals = [init_U]
import time
time_vals = np.zeros(max_iter+1)
time_vals[0] = 0
for i in range(max_iter):
	st_time = time.time()
	lL_est, y_U,gamma = do_lb_search( U, prev_U,  uL_est,lL_est=lL_est_main)
	prev_U = U
	
	temp_ulest = uL_est
	
	uL_est, U = do_ub_search( y_U, uL_est)
	print(main_func( U, fun_num=fun_num))
	uL_est_vals = uL_est_vals + [uL_est]
	lL_est_vals = lL_est_vals + [lL_est]

	gamma_vals = gamma_vals + [gamma]
	U_vals = U_vals + [U]
	temp = main_func( U, fun_num=fun_num)
	if np.isnan(temp):
		raise
	func_vals = func_vals + [temp]

	lyapunov_vals = lyapunov_vals + [(1/uL_est)*temp+ del_val*breg( U, prev_U, breg_num=breg_num)]
	time_vals[i+1] = time.time() - st_time

filename = 'results/cocain_'+str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
np.savetxt(filename,np.c_[func_vals, lyapunov_vals, uL_est_vals, lL_est_vals, gamma_vals, time_vals])

filename_1 = 'results/cocain_trajectory_'+str(init_U)+'.txt'
np.savetxt(filename_1,U_vals)
