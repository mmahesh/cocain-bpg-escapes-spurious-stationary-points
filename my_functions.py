import numpy as np


def main_func( U, fun_num=1):
	# function value
	# fun_num is not necessary TODO: will fix this later but the code works for now
	if fun_num==1:
		temp_sum = 0
		count = 0
		x = U[0]
		y = U[1]

		return 0.5*np.log(1+100*(np.abs(x-1)**2))+0.5*np.log(1+(100*np.abs(y-1)**2))\
		+ np.log(1+np.abs(x))+np.log(1+np.abs(y))
	


def grad( U,  fun_num=1):
	# Gradient
	if fun_num==1:
		temp_sum = 0
		count = 0
		x = U[0]
		y = U[1]

		grad_x = 100*(x-1)/(1+100*(np.abs(x-1)**2))
		grad_y = 100*(y-1)/(1+(100*np.abs(y-1)**2))

		return np.array([grad_x, grad_y])



def abs_func( U, U1, abs_fun_num=1, fun_num=1):
	# f(x) + g(y)+ <grad, x-y> at point y

	if fun_num==1 and abs_fun_num==1:
		G = grad( U1, fun_num=fun_num)
		x = U[0]
		y = U[1]

		x_1 = U1[0]
		y_1 = U1[1]

		return np.log(1+np.abs(x))+np.log(1+np.abs(y)) + np.sum(np.multiply(G,U-U1))\
		 + 0.5*np.log(1+100*(np.abs(x_1-1)**2))+0.5*np.log(1+(100*np.abs(y_1-1)**2))


def breg( U, U1, breg_num=1):
	# Bregman distance here it is just the Euclidean distance
	# breg_num is not required TODO: will fix this later.
	
	if breg_num==1:
		temp =  0.5*(np.sum(np.multiply(U-U1,U-U1)))
		if temp >=1e-15:
			# numerical fix
			return temp
		else:
			return 0
