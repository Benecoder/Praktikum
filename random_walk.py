#! /usr/bin/env python3
"""

simulates a random walk in 3 dimension
caluclates the mean square distance covered in any number of dimensions
fits a linear and quadratic function
Calulctaes the diffusion constant

plots numbers 

"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
from scipy.optimize import curve_fit


np.random.seed(2)

no_steps = 1000
steps = 2e-6*(np.random.rand(no_steps,3)-0.5)
coords = np.cumsum(steps,axis=0)

# linear fit
def lin_f(x,a):
    return a*x
# quadractic fit function
def quad_f(x,a,b):
    return a*x**2+b*x

# caluclates the means square distance in given dimension
# finds diffiusion constant
def plot_D_sim(steps,d):

    result = np.zeros(no_steps)

    for delta_t in range(1,no_steps):
        r_2_buffer = np.empty(no_steps-delta_t)
        for start_index in range(no_steps-delta_t):
            path = np.cumsum(steps[start_index:start_index+delta_t],axis=0)
            r_2_buffer[start_index] = np.sum(path[-1][:d]**2)
        result[delta_t] = np.mean(r_2_buffer)

    delta_t = np.arange(no_steps)/5.
    fit_limit = int(100*5)

    lin_coeff,_ = curve_fit(lin_f,delta_t[:fit_limit],result[:fit_limit])
    quad_coeff,_ = curve_fit(quad_f,delta_t[:fit_limit],result[:fit_limit])
    lin_graph = lin_f(delta_t[:fit_limit],*lin_coeff)
    quad_graph = quad_f(delta_t[:fit_limit],*quad_coeff)

    print('D(linear fit): '+str(lin_coeff[0]))
    print('D(quadratic fit): '+str(quad_coeff[1]))
    print('drift: '+str(quad_coeff[0]))

    plt.xlabel('$\Delta t $ $[s]$',fontsize=15)
    plt.ylabel('$<r^2>$ $[\mu m^2/s]$',fontsize=15)
    plt.ylim((0,4e-10))
    plt.plot(delta_t[:fit_limit],lin_graph,'--',color = 'green')
    plt.plot(delta_t[:fit_limit],quad_graph,'--',color = 'red')
    plt.plot(delta_t,result,color = 'blue')
    plt.plot([fit_limit/5,fit_limit/5],[0,4e-10],color = 'gray',alpha=0.5)
    plt.show()

def plot_2d():
    plt.plot(coords[:,0],coords[:,1],'black')
    plt.show()


if __name__ == '__main__':

#    plot_D()
#    plot_2d()
    plot_D_sim(steps,3)
