#!/usr/bin/python
'''
Usage: 
    Plot the HIRO `CartPos*.dat` XYZ data. 
    Based on `newtanvec.py`, removed redundant code.
    Change the `path` variable to set the HIRO data location.
Last Edit: 2016/03/25 Fri
'''
print "importing...",
import os

import numpy
from numpy import pi, cos, sin, linspace, sign, sqrt

import matplotlib
import matplotlib.lines as mlines

from matplotlib import pyplot as plt
matplotlib.interactive(True)

#from mpl_toolkits.mplot3d import Axes3D

import baxterbezier as bezier
print "ok"

def reader(filename):
    lines = file(filename,'r').readlines();
    data = numpy.array(
        map(
            lambda x:map(
                lambda x:float(x),
                x[:len(x)-2].split('\t'))
            ,lines))
    return data

def getxyz(filename):
    data = reader(filename).T

    x = data[1]
    y = data[2]
    z = data[3]
    return x,y,z
    

def plotdata(filename,ax,color=None):
    x,y,z = getxyz(filename)
    if not color:
        ax.plot(x, y, z)#, label='parametric curve')    
    else:
        ax.plot(x, y, z,c=color)#, label='parametric curve')    
    return x,y,z



if __name__ == "__main__":
    # file reading
    path = '~/finalyear/HIRO_SideApproach/' # path of the HIRO data
    path = os.popen('echo '+path).read().strip() # get the absolute path
    filelist = os.popen('find '+path+' -name "CartPos*"').read() # find all CartPos*.dat files
    filepath = filelist.strip(path+'\n').split('\n'+path) # filename list
    filepath.sort()

    # split the list into 3 sublists
    data0 = filter(lambda x:x.startswith('20121127'),filepath)
    data1 = filter(lambda x:x.startswith('Prelim'),filepath) # useless? only four files
    data2 = filter(lambda x:x.startswith('x'),filepath)
    
    #prepare plotting
    matplotlib.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    for i in data0:
        x,y,z = plotdata(path+i,ax,'b')
    for i in data2:
        x,y,z = plotdata(path+i,ax,'r')
        
    # legend box
    blue_line = mlines.Line2D([], [], c='blue',  label='20121127')
    red_line = mlines.Line2D([], [], c='red',  label='x*')
    ax.legend([red_line,blue_line],['20121127','x*'])

    # background colour
    ax.set_axis_bgcolor('DimGray')
    #ax.set_axis_bgcolor((0.2,0.2,0.2))

    plt.show()
