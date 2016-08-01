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

import vector_sympy
reload(vector_sympy)
#from vector_sympy import dcc_base
from load_data import normalize

dcc_base = [[numpy.zeros(3)]]

def gen_base_numpy():
    global dcc_base
    x,y,z = numpy.eye(3)
    for i in [x,y,z]:
        dcc_base[0].append(i)
        dcc_base[0].append(-i)

gen_base_numpy()

def main(fig, it=3, oct=False):

    global dcc_base

    plt.hold(True)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_aspect('equal')
    #ax = fig.add_subplot(2,2,it, projection='3d')


    #u = numpy.linspace(0, 2*numpy.pi, 100)
    #v = numpy.linspace(0, 2*numpy.pi, 100)

    #x = 0.95*numpy.outer(numpy.cos(u), numpy.sin(v))
    #y = 0.95*numpy.outer(numpy.sin(u), numpy.sin(v))
    #z = 0.95*numpy.outer(numpy.ones(numpy.size(u)), numpy.cos(v))
    #ax.plot_surface(x, y, z, color='w', rstride=7, cstride=7, linewidth=0)

    if_oct = oct

    #v = -numpy.array([1,1,0])
    #u = -numpy.array([-1,1,0])
    v = numpy.array([1,0,0])
    u = numpy.array([0,1,0])
    v = v/numpy.linalg.norm(v)
    u = u/numpy.linalg.norm(u)
    base = [set()]
    base[0].add((0,0,0))

    for i in map(tuple,[v, u,numpy.cross(v,u),numpy.cross(u,v),-v,-u]):
        base[0].add(i)

    #if not if_oct:
    #    for i in map(tuple,[v, u,numpy.cross(v,u),numpy.cross(u,v),-v,-u]):
    #        base[0].add(i)
    #else:
    #    for i in map(tuple,[v,u,numpy.cross(v,u)]):
    #        base[0].add(i)
    
    #it = 3 # time of iteration

    #x_,y_,z_ = a[0],a[1],a[2]
    #ax.plot(x_, y_, z_, label='parametric curve')    


    b = numpy.cross(v,u)

    v *= 1.2
    u *= 1.1
    b *= 1.1
    ax.text(v[0],v[1],v[2],'T')
    ax.text(u[0],u[1],u[2],'N')
    ax.text(b[0],b[1],b[2],'B')

    ax.plot(*(numpy.array([[1,0,0],[1,1,0],[0,1,0],[0,1,1],[0,0,1],[1,0,1],[1,0,0]])*0.1).T,color='black',linewidth=0.5)
    
    if if_oct:
        t = numpy.linspace(0,numpy.pi/2,361)
        x = numpy.sin(t)
        y = numpy.cos(t)
        z = numpy.zeros(x.shape)
        ax.plot(x,y,z,color='orange',linewidth=0.5)
        ax.plot(y,z,x,color='orange',linewidth=0.5)
        ax.plot(z,x,y,color='orange',linewidth=0.5)
    else:
        t = numpy.linspace(0,numpy.pi*2,361)
        x = numpy.sin(t)
        y = numpy.cos(t)
        z = numpy.zeros(x.shape)
        ax.plot(x,y,z,color='orange',linewidth=0.5)
        ax.plot(y,z,x,color='orange',linewidth=0.5)
        ax.plot(z,x,y,color='orange',linewidth=0.5)




    base_ = dcc_base
    #for i in dcc_base:
    #    normalize(i)
    base = map(lambda bs_:set(map(tuple,bs_)),base_)
    baseadd = [base_[0]]
    for i in xrange(1,it):
        baseadd.append(base[i].difference(base[i-1]))
    for i in xrange(0,it):
        print "==",it, i,len(base[i]),len(baseadd[i])

    if if_oct:
        for i in xrange(0,it):
            base[i] = filter(lambda x: x[0]>=0 and x[1]>=0 and x[2]>=0,base[i])
            baseadd[i] = filter(lambda x: x[0]>=0 and x[1]>=0 and x[2]>=0,baseadd[i])
            print "--",it, i,len(base[i]),len(baseadd[i])

    barr = []
    color = ['red','green','blue','yellow']
    xyz111 = numpy.array([1,1,1])
    xyz111 = xyz111/numpy.linalg.norm(xyz111)
    for i in xrange(it):
        barr.append(map(lambda x:numpy.array(((0,0,0),x)).T,baseadd[i]))
        for j in barr[i]:
            dist111 = 1-(numpy.linalg.norm(xyz111-numpy.array(j.T[1]))/2.0)*0.7
            if(j.T[1]>j.T[0]).all():
                dist111 = 1
            ax.plot(j[0],j[1],j[2], color=color[i],linewidth=2,alpha=dist111)

    barradd = []
    color = 'rgbk'
    for i in xrange(it):
        barradd.append(numpy.array(list(baseadd[i])).T)
        #if(it>=4 and not if_oct):
        #    continue
        ax.scatter(barradd[i][0],barradd[i][1],barradd[i][2],c=color[3],marker='o')
    ax.legend()
    #ax.title = "DCC "+str(len(base))
    

    if if_oct:
        rangelim = (0,1.1)
    else:
        rangelim = (-1.1,1.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim3d(rangelim)
    ax.set_ylim3d(rangelim)
    ax.set_zlim3d(rangelim)
    #ax.set_xticks(())
    #ax.set_yticks(())
    #ax.set_zticks(())
    #ax.set_axis_off()
    ax.set_title("DCC "+str(dcc_base[it-1].shape[0]))

    ax.view_init(azim=22.5)
    fig.savefig("DCC "+str(dcc_base[it-1].shape[0])+"-"+("oct" if if_oct else "full")+".png")
    plt.close(fig)
    return base


if __name__ == "__main__":
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    #ax = fig.gca(projection='3d')

    base = main(fig,1)
    base = main(fig,2)
    base = main(fig,3)
    base = main(fig,4)

    base = main(fig,1,oct=True)
    base = main(fig,2,oct=True)
    base = main(fig,3,oct=True)
    base = main(fig,4,oct=True)

    plt.close(fig)
    #plt.show()
