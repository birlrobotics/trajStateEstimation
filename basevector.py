#!/usr/bin/python

print "importing..."
import numpy
from numpy import pi, cos, sin, linspace, sign, sqrt
import scipy
import matplotlib
from matplotlib import pyplot as plt
matplotlib.interactive(True)
#from mayavi import mlab
#from mayavi.mlab import plot3d,quiver3d

##set parallel projection as window default
#mlab.gcf().scene.parallel_projection = True

from copy import deepcopy

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt    

def normalize(v):
    return v/numpy.linalg.norm(v)

def baseEnhance(base_):
    base = deepcopy(base_)
    bvlist = map(numpy.array,[i for i in base])
    Nbvlist = len(bvlist)
    for i in xrange(Nbvlist):
        for j in xrange(Nbvlist):
            if i==j or tuple(numpy.cross(bvlist[i],bvlist[j]))==(0,0,0):
                continue
            tmpbv = tuple(normalize(bvlist[i]+bvlist[j]))
            if tmpbv in base:
                continue
            base.add(tmpbv)
    return base

def main(fig, it=3):

    plt.hold(True)
    ax = fig.add_subplot(2,2,it, projection='3d')
    #u = numpy.linspace(0, 2*numpy.pi, 100)
    #v = numpy.linspace(0, 2*numpy.pi, 100)

    #x = 0.95*numpy.outer(numpy.cos(u), numpy.sin(v))
    #y = 0.95*numpy.outer(numpy.sin(u), numpy.sin(v))
    #z = 0.95*numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))
    #ax.plot_surface(x, y, z, color='w', rstride=7, cstride=7, linewidth=0)

    v = numpy.array([1,1,0])
    u = numpy.array([-1,1,0])
    v = v/numpy.linalg.norm(v)
    u = u/numpy.linalg.norm(u)
    base = [set()]
    base[0].add((0,0,0))
    #for i in map(tuple,[v, u,numpy.cross(v,u),numpy.cross(u,v),-v,-u]):
    for i in map(tuple,[-v,-u,numpy.cross(v,u)]):
        base[0].add(i)
    
    #it = 3 # time of iteration

    #x_,y_,z_ = a[0],a[1],a[2]
    #ax.plot(x_, y_, z_, label='parametric curve')    

    baseadd = [base[0]]
    for i in xrange(0,it-1):
        base.append(baseEnhance(base[i]))
        baseadd.append(base[i+1].difference(base[i]))

    barr = []
    color = ['red','green','blue','yellow']
    for i in xrange(it):
        barr.append(map(lambda x:numpy.array(((0,0,0),x)).T,baseadd[i]))
        for j in barr[i]:
            ax.plot(j[0],j[1],j[2], color=color[i])

    barradd = []
    color = 'rgbk'
    for i in xrange(it):
        barradd.append(numpy.array(list(baseadd[i])).T)
        ax.scatter(barradd[i][0],barradd[i][1],barradd[i][2],c=color[3],marker='o')
    ax.legend()


if __name__ == "__main__":
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    #ax = fig.gca(projection='3d')

    main(fig,1)
    main(fig,2)
    main(fig,3)
    main(fig,4)

    plt.show()
