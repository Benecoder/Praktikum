#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import pandas as pd



class Line():

    def __init__(self,v,w):
        self.v = v
        self.w = w

    def get_point(self,lamda):
        return lamda*self.v+self.w

    def d(self,x):
        lamda = np.dot(x,self.v)-np.dot(self.w,self.v)
        con_vector = self.get_point(lamda)-x
        return np.sqrt(np.dot(con_vector,con_vector))

    def mse(self,points):
        ds = np.apply_along_axis(self.d,0,points)
        return np.mean(ds)

    def get_corner(self,line):

        v1v2 = np.dot(self.v,line.v)
        w1v2 = np.dot(self.w,line.v)
        w2v2 = np.dot(line.w,line.v)
        w2v1 = np.dot(line.w,self.v)
        w1v1 = np.dot(self.w,self.v)

        mu1 = (w1v2*v1v2-w2v2*v1v2+w2v1-w1v1)/(1-v1v2**2)
        mu2 = mu1*v1v2+w1v2-w2v2

        return mu1,mu2

    def get_sample(self,b):
        return np.array([self.get_point(i) for i in np.arange(b-20,b+150,10)])

# deals with the tsv files
def extract_data(sides,plot1=False,print_measurement=False):

    names = [str(n) for n in range(31)]
    names[4] = 'state1'
    names[9] = 'x1'
    names[10] = 'y1'
    names[11] = 'z1'
    names[12] = 'error1'
    names[14] = 'state2'
    names[-3] = 'x2'
    names[-2] = 'y2'
    names[-1] = 'z2'


    file_names = ['Data_OT/'+sides[i]+'.tsv' for i in range(4)]
    tracks = []
    errors = []


    for side in sides:

        file_name = 'Data_OT/'+side+'.tsv'
        data = pd.read_csv(file_name,delimiter='\t',names=names,skiprows=1,index_col=2)

        data = data[data['state1']=='OK']
        tracks.append(np.array([data['x1'],data['y1'],data['z1']]))
        errors.append(data['error1'])

    return tracks,errors


def measure_object(tracks,errors,sides,plot1=False,print_measurement=False,residual=False):


    if plot1:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

    lines = []
    res = []
    for i,side in enumerate(sides):
        if plot1:
            ax.scatter(tracks[i][0],tracks[i][1],tracks[i][2],marker='.',color='red',alpha=0.4)

        cov = np.cov(tracks[i])
        w,v = np.linalg.eigh(cov)
        v_max = v[:,np.argmax(w)]
        mean = np.mean(tracks[i],axis=1)

        lines.append(Line(v_max,mean))
        if residual:
            res.append(lines[-1].mse(tracks[i]))

    if residual:
        print(res)


    # find corner
    corners = np.zeros((4,3))
    line_distance = np.zeros(4)
    for side_i in range(4):

        line1 = lines[side_i-1]
        line2 = lines[side_i]

        mu1,mu2 = line1.get_corner(line2)
        corner_l1 = line1.get_point(mu1)
        corner_l2 = line2.get_point(mu2)
        line_distance[side_i] = np.linalg.norm(corner_l1-corner_l2)
        corners[side_i] = (corner_l1+corner_l2)/2.

    d = [np.linalg.norm(corners[i-1]-corners[i]) for i in range(4)]


    # plotting fun
    if print_measurement:
        print('always in oder of: ')
        print(sides)
        print('line_distance in mm')
        print(line_distance)
        print('distance between corners')
        print(d)


    if plot1:
        plotted_line = np.zeros((5,3))
        plotted_line[:4] = corners
        plotted_line[-1] = corners[0]

        ax.plot(plotted_line[:,0],plotted_line[:,1],plotted_line[:,2],color='black')

        ax.set_xlabel('x in mm')
        ax.set_ylabel('y in mm')
        ax.set_zlabel('z in mm')
        plt.show()

def error_hist(error):

    tot_error = np.concatenate(error)
    plt.xlabel('error in mm')
    plt.ylabel('absolute number')
    plt.hist(tot_error,bins=60)
    plt.show()

def plot_track(track1,track2):

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(track1[0],track1[1],track1[2])
    ax.scatter(track2[0],track2[1],track2[2])
    ax.set_xlabel('x in mm')
    ax.set_ylabel('y in mm')
    ax.set_zlabel('z in mm')

    plt.show()


def get_hist(d,title):

    measured = 137.0
    mean = np.mean(d)
    std = np.std(d)
    no_bins = 60
    hist,edges = np.histogram(d,bins=no_bins)
    max = np.max(hist)

    plt.bar(edges[:-1],hist/max,width=0.01)
    plt.title(title)
    plt.xlabel('distance in mm')
    plt.ylabel('proportion')
    plt.plot([mean,mean],[0,1],color='black')
    plt.plot([mean+std,mean+std],[0,1],color='black',alpha=0.2)
    plt.plot([mean-std,mean-std],[0,1],color='black',alpha=0.2)
    plt.plot([measured,measured],[0,1],color='red')

    plt.show()


def get_accuracy():

    names = [str(n) for n in range(35)]
    names[-3] = 'x1'
    names[-2] = 'y1'
    names[-1] = 'z1'
    names[-7] = 'x2'
    names[-6] = 'y2'
    names[-5] = 'z2'
    names[-9] = 'markers'


    mid_data = pd.read_csv('Data_OT/distance_mitte.tsv',delimiter='\t',names=names,skiprows=1,index_col='0')
    boarder_data = pd.read_csv('Data_OT/distance_rand.tsv',delimiter='\t',names=names,skiprows=1,index_col='0')
    mid_length = mid_data.shape[0]

    mid_track_1 = np.array([mid_data['x1'],mid_data['y1'],mid_data['z1']])
    mid_track_2 = np.array([mid_data['x2'],mid_data['y2'],mid_data['z2']])

    # 3d plot mid
    plot_track(mid_track_1,mid_track_2)
    boarder_track_1 = np.array([boarder_data['x1'],boarder_data['y1'],boarder_data['z1']])
    boarder_track_2 = np.array([boarder_data['x2'],boarder_data['y2'],boarder_data['z2']])

    # 3d plot boarder
#    plot_track(boarder_track_1,boarder_track_2)

    marker_mid_overflow = (mid_data['markers']==4).astype('int64')
    marker_mid_overflow = np.nonzero(marker_mid_overflow)[0]

    d_mid = np.sqrt(np.sum((mid_track_1-mid_track_2)**2,axis=0))
#    plt.ylabel('measured distance in mm')
#    plt.xlabel('points of measurment')
#    plt.plot(d_mid)
#    plt.plot(marker_mid_overflow,np.max(d_mid)*np.ones(marker_mid_overflow.shape[0]),'.',color='red')
#    plt.show()

    marker_boarder_overflow = (boarder_data['markers']!=2).astype('int64')
    marker_boarder_overflow = np.nonzero(marker_boarder_overflow)[0]

    d_board = np.sqrt(np.sum((boarder_track_1-boarder_track_2)**2,axis=0))
#    plt.ylabel('measured distance in mm')
#    plt.xlabel('points of measurment')
#    plt.plot(d_board)
#    plt.plot(marker_boarder_overflow,np.max(d_board)*np.ones(marker_boarder_overflow.shape[0]),'.',color='red')
#    plt.show()


    measured = 137.0

    d_mid_filtered = d_mid[mid_data['markers']==2]
    mean_mid = np.mean(d_mid_filtered)
    std_mid = np.std(d_mid_filtered)
#    plt.plot(d_mid_filtered)
#    plt.plot([0,d_mid_filtered.shape[0]],[measured,measured],color='red')
#    plt.plot([0,d_mid_filtered.shape[0]],[mean_mid,mean_mid],color='black')
#    plt.plot([0,d_mid_filtered.shape[0]],[mean_mid+std_mid,mean_mid+std_mid],color='black',alpha=0.2)
#    plt.plot([0,d_mid_filtered.shape[0]],[mean_mid-std_mid,mean_mid-std_mid],color='black',alpha=0.2)
#    plt.xlabel('points of measurement')
#    plt.ylabel('measured distance in mm')
#    plt.show()

    print('mean_mid: '+str(mean_mid))
    print(' std_mis: '+str(std_mid))
    print('95th perc mid: '+str(std_mid*2))

    d_boarder_filtered = d_board[boarder_data['markers']==2]
    mean_boarder = np.mean(d_boarder_filtered)
    std_boarder = np.std(d_boarder_filtered)
#    plt.plot(d_boarder_filtered)
#    plt.plot([0,d_boarder_filtered.shape[0]],[measured,measured],color='red')
#    plt.plot([0,d_boarder_filtered.shape[0]],[mean_boarder,mean_boarder],color='black')
#    plt.plot([0,d_boarder_filtered.shape[0]],[mean_boarder+std_boarder,mean_boarder+std_boarder],color='black',alpha=0.2)
#    plt.plot([0,d_boarder_filtered.shape[0]],[mean_boarder-std_boarder,mean_boarder-std_boarder],color='black',alpha=0.2)
#    plt.xlabel('points of measurement')
#    plt.ylabel('measured distance in mm')
#    plt.show()


    print('mean_boarder: '+str(mean_boarder))
    print('std_boarder: '+str(std_boarder))
    print('95th perc boarder: '+str(std_boarder*2))


#    get_hist(d_boarder_filtered,'border measurement')
#    get_hist(d_mid_filtered,'mid measurement')





if __name__ == '__main__':

    global_sides = ['front_2', 'right', 'back', 'left']
    local_sides = ['frontlocal','rightlocal','backlocal','leftlocal']

    tracks,errors = extract_data(global_sides)
    measure_object(tracks,errors,global_sides,plot1=True,residual=True)
