
import matplotlib.pyplot as plt
import numpy as np


def function(x,A,B,C):
    return (A*(x-C))/np.log(B*(x-C))


x_ticks = np.arange(0,100,0.1)

series_1 = function(x_ticks,24.4,0.16,-5.7)
series_2 = function(x_ticks,18.7,0.13,-9.5)

plt.plot(x_ticks,series_1)
plt.title('paschen kurve 1 (A=24.4,B=0.16...)')
plt.show()


plt.plot(x_ticks,series_2)
plt.title('paschen kurve 2 (A=18.6,B=0.13...)')
plt.show()

