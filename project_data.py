#! /usr/bin/env python3
""" 

Reads the csv file containing
the paths observed in the experiment.
Fits a linear and a quadratic function to the data.

#plots the diffusion constnat found with labview

"""

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np


data = pd.read_csv('2mum25C.csv.txt',delimiter='\t')
no_paths = data.shape[1]//2
no_steps = 300

# linear fit function
def lin_f(x,a):
    return a*x
# quadratic fit function
def quad_f(x,a,b):
    return (a*x)**2+b*x

# linar fit with offest at the origin
def lin_f_shifted(x,a,b):
    return a*x+b

def get_path_lengths(data):
	length_of_paths = np.ones(no_paths,dtype=int)*no_steps
	for i in range(no_steps):
		for path_index in range(no_paths):
			if length_of_paths[path_index] == no_steps:
				if np.isnan(data['x_'+str(path_index)][i]):
					length_of_paths[path_index] = i
	return length_of_paths

# cuts chucnks of determined length out of every path
# caluclates the mean square distance covered in a
# given number of dimensions.
# Adjusts for 1px = 0.7299 mum
def sample_D(data,d):

	result = np.zeros(no_steps)
	length_of_paths = get_path_lengths(data)

	for delta_t in range(1,no_steps):
		r_2_buffer = []
		for path_index in range(no_paths):
			for start_index in range(length_of_paths[path_index]-delta_t):
				x_path = np.array(data['x_'+str(path_index)][start_index:start_index+delta_t])
				y_path = np.array(data['y_'+str(path_index)][start_index:start_index+delta_t])
				x_path -= x_path[0]
				y_path -= y_path[0]
				path = np.array([x_path,y_path]).T
				r_2_buffer.append(np.sum((0.7299*path[-1][:d])**2))
		result[delta_t] = np.mean(np.array(r_2_buffer))
		print('finished '+str(float(delta_t/no_steps)*100)+'%')

	return result

# fits function to the mean square distance
# adjusts for 5Hz measurments
# same number of points as in LabView
# plots the data
def plot_D(result,d):
    delta_t = np.arange(no_steps)/5.
    fit_limit = 209

    lin_coeff,_ = curve_fit(lin_f,delta_t[:fit_limit],result[:fit_limit])
    quad_coeff,_ = curve_fit(quad_f,delta_t[:fit_limit],result[:fit_limit])
    lin_graph = lin_f(delta_t[:fit_limit],*lin_coeff)
    quad_graph = quad_f(delta_t[:fit_limit],*quad_coeff)

    print('D(linear fit): '+str(lin_coeff[0]/(2*d)))
    print('D(quadratic fit): '+str(quad_coeff[1]/(2*d)))
    print('drift: '+str(quad_coeff[0]))

    plt.xlabel('$\Delta t $ $[s]$',fontsize=15)
    plt.ylabel('$<r^2>$ $[\mu m^2/s]$',fontsize=15)
    plt.plot(delta_t[:fit_limit],lin_graph,'--',color = 'green')
    plt.plot(delta_t[:fit_limit],quad_graph,'--',color = 'red')
    plt.plot(delta_t,result,color = 'blue')
    plt.plot([fit_limit/5,fit_limit/5],[0,120],color = 'gray',alpha=0.5)
    plt.show()


# plots the diffusion constants from labview
def plot_D():

    T = np.arange(25,46,4)
    mum05 = np.array([0.63,0.39,0.52,0.56,0.61,0.7])
    mum2 = np.array([0.29,0.25,0.22,0.4,0.39,0.31])

    lin_coeff,_ = curve_fit(lin_f_shifted,T,mum05)
    lin_coeff_,_ = curve_fit(lin_f_shifted,T[1:],mum05[1:])
    lin_graph = lin_f_shifted(T,*lin_coeff)
    lin_graph_ = lin_f_shifted(T,*lin_coeff_)

    print(lin_coeff)
    print(lin_coeff_)

    plt.xlabel('Temperatur [$^\circ C$]',fontsize=15)
    plt.ylabel('D [$\mu m^2/s$]',fontsize = 15)
    plt.plot(T,mum05,'.',color='blue')
    plt.plot(T,mum2,'.',color='orange')
    plt.plot(T,lin_graph,color='gray')
    plt.plot(T,lin_graph_,color='gray',alpha=0.5)

    plt.show()

if __name__=='__main__':
	result = sample_D(data,1)
	np.savetxt('D_2mum25C_1d.csv',result)
	result = np.loadtxt('D_2mum25C_1d.csv')
	plot_D(result,1)