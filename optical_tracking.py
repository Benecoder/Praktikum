#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def measure_object():

   names = [str(n) for n in range(18)]
   names[4] = 'state1'
   names[9] = 'x1'
   names[10] = 'y1'
   names[11] = 'z1'
   names[14] = 'state2'
   names[15] = 'x2'
   names[16] = 'y2'
   names[17] = 'z2'

   back = pd.read_csv('Data_OT/back.tsv',delimiter='\t',names=names,skiprows=1,index_col='0')
   front = pd.read_csv('Data_OT/front_2.tsv',delimiter='\t',names=names,skiprows=1,index_col='0')
   right = pd.read_csv('Data_OT/right.tsv',delimiter='\t',names=names,skiprows=1,index_col='0')
   left = pd.read_csv('Data_OT/left.tsv',delimiter='\t',names=names,skiprows=1,index_col='0')

   plt.plot(back['x1'])
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



measure_object()
