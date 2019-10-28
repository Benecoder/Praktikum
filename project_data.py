#! /usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np



data = pd.read_csv('2mum25C.csv.txt',delimiter='\t')
no_paths = data.shape[1]//2
no_steps = 300


def lin_f(x,a):
    return a*x


def quad_f(x,a,b):
    return (a*x)**2+b*x

def get_path_lengths(data):
	length_of_paths = np.ones(no_paths,dtype=int)*no_steps
	for i in range(no_steps):
		for path_index in range(no_paths):
			if length_of_paths[path_index] == no_steps:
				if np.isnan(data['x_'+str(path_index)][i]):
					length_of_paths[path_index] = i
	return length_of_paths

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

def test_data(data):

    length_of_paths = get_path_lengths(data)
    paths = []
    path_index = 0
    x_path = data['x_'+str(path_index)][:length_of_paths[path_index]]



def plot_D(result,d):
    delta_t = np.arange(no_steps)/5.
    fit_limit = 209#no_steps

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

if __name__=='__main__':
	result = sample_D(data,1)
	np.savetxt('D_2mum25C_1d.csv',result)
	result = np.loadtxt('D_2mum25C_1d.csv')
	plot_D(result,1)