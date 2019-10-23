#! /usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True


np.random.seed(1)

no_steps = 1000
steps = (np.random.rand(no_steps,3)-0.49)
coords = np.cumsum(steps,axis=0)


def plot_D(steps):

    result = np.zeros(no_steps)

    for delta_t in range(1,no_steps):
        r_2_buffer = np.empty(no_steps-delta_t)
        for start_index in range(no_steps-delta_t):
            path = np.cumsum(steps[start_index:start_index+delta_t],axis=0)
            r_2_buffer[start_index] = np.sum(path[-1]**2)
        result[delta_t] = np.mean(r_2_buffer)

    delta_t = np.arange(no_steps)
    fit_limit = 550

    lin_coeff = np.polyfit(delta_t[:fit_limit],result[:fit_limit],1)
    quad_coeff = np.polyfit(delta_t[:fit_limit],result[:fit_limit],2)
    lin_fit = np.poly1d(lin_coeff)
    quad_fit = np.poly1d(quad_coeff)
    lin_graph = lin_fit(delta_t[:fit_limit])
    quad_graph = quad_fit(delta_t[:fit_limit])

    print('linear coefficents: '+str(lin_coeff))
    print('quad coefficents: '+str(quad_coeff))

    plt.xlabel('$\Delta t$',fontsize=15)
    plt.ylabel('$<r^2> over \Delta t$',fontsize=15)
    plt.plot(lin_graph,'--',color = 'green')
    plt.plot(quad_graph,'--',color = 'red')
    plt.plot(result,color = 'blue')
    plt.plot([fit_limit,fit_limit],[0,np.max(result)],color = 'gray',alpha=0.5)
    plt.show()

def plot_2d():
    plt.plot(coords[:,0],coords[:,1])
    plt.show()


if __name__ == '__main__':
    plot_2d()
