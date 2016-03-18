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

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def reader(filename):
    lines = file(filename,'r').readlines();
    data = map(lambda x:map(lambda x:float(x),x[:len(x)-2].split('\t')),lines)
    return numpy.array(data)

def rpy2rotmat(r,p,y):
    ret = numpy.zeros((3,3))
    ret[0][0] = cos(y)*cos(p)
    ret[0][1] = cos(y)*sin(p)*sin(r) - sin(y)*cos(r)
    ret[0][2] = cos(y)*sin(p)*cos(r) + sin(y)*sin(r)
    ret[1][0] = sin(y)*cos(p)
    ret[1][1] = sin(y)*sin(p)*sin(r) + cos(y)*cos(r)
    ret[1][2] = sin(y)*sin(p)*cos(r) - cos(y)*sin(r)
    ret[2][0] = -sin(p)
    ret[2][1] = cos(p)*sin(r)
    ret[2][2] = cos(p)*cos(r)
    return ret

#Gram-Schmidt process (orthonomalising)
def gsproc(v):
    assert isinstance(v,numpy.ndarray) \
        and len(v.shape) == 2\
        and all(v.shape)>0
    n = v.shape[0]
    
    p = numpy.zeros(v.shape) # p = v-proj(v), orthogonal to the projection vector
    o = numpy.zeros(v.shape) # orthonormal basis

    for i in xrange(n):
        projv = 0
        for j in xrange(i):
            projv += numpy.inner(v[i],o[j]) * o[j]
        p[i] = v[i] - projv
        nrm = numpy.linalg.norm(p[i])
        if nrm:
            o[i] = p[i] / nrm
        else:
            o[i] = 0

    return o


# do Gram-Schmidt process for X,Y,Z frames
def basis_gsproc(b):
    assert isinstance(b,numpy.ndarray) \
        and len(b.shape) == 3 \
        and all(b.shape)>0
    n = b.shape[0] 
    for i in xrange(n):
        b[i] = gsproc(b[i])
    #b[99]=b[99]*100
    

def preprocess(filename):
    data = reader(filename).T
    count = data.shape[1]

    C = 0.04

    displayMult = 1

    x = data[1]*displayMult
    y = data[2]*displayMult
    z = data[3]*displayMult
    return x,y,z
    dx = numpy.zeros(x.shape)
    dy = numpy.zeros(y.shape)
    dz = numpy.zeros(z.shape)

    dx[:count-1] = x[1:] - x[:count-1]
    dy[:count-1] = y[1:] - y[:count-1]
    dz[:count-1] = z[1:] - z[:count-1]

    dx[count-1] = dx[count-2]
    dy[count-1] = dy[count-2]
    dz[count-1] = dz[count-2]

    d = sqrt(dx*dx+dy*dy+dz*dz)
    vx = dx/d
    vy = dy/d
    vz = dz/d

    v = numpy.array([vx,vy,vz]).T
    w = numpy.zeros(v.shape)
    w[1:] = numpy.cross(v[:count-1],v[1:])
    w[0] = w[1]
    w[count-1] = w[count-2]
    w = w.T
    wx,wy,wz = w[0],w[1],w[2]
    lw = sqrt(wx*wx+wy*wy+wz*wz)
    wx /= lw
    wy /= lw
    wz /= lw

    #print 'plotting...'
    if False:
        plt.plot(l,wy)
        plt.plot(l,vy)
        
        plt.show()
    
    
    x_,y_,z_ = x[:count],y[:count],z[:count]
    vx_,vy_,vz_ = vx[:count],vy[:count],vz[:count]
    wx_,wy_,wz_ = wx[:count],wy[:count],wz[:count]
    
    pls = []
    #pls.append(plot3d(x_,y_,z_,y_,tube_radius=0.01))
    #pls.append(quiver3d(x_,y_,z_,vx_,vy_,vz_,color=(1,0,0),mode='arrow',scale_factor=C))
    #pls.append(quiver3d(x_,y_,z_,wx_,wy_,wz_,color=(0,1,0),mode='arrow',scale_factor=C))

    



    #mlab.show()
    
    return x,y,z
    

# Main process
#def main():

def plotdata(filename,ax):
    x_,y_,z_ = preprocess(filename)
    ax.plot(x_, y_, z_, label='parametric curve')    
    return x_,y_,z_

def comp(a,b):
    a = preprocess('CartPos-'+str(a)+'.dat')
    b = preprocess('CartPos-'+str(b)+'.dat')
    #for i in xrange(3):
    #    print any(a[i]!=b[i],)
    ret = any([a[i].tolist()!=b[i].tolist() for i in xrange(3)])
    return ret

if __name__ == "__main__":
    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for i in xrange(10,22):
        x,y,z = plotdata('CartPos-'+str(i)+'.dat',ax)
    #a = preprocess('CartPos-21.dat')
    #b = preprocess('CartPos-20.dat')
    
    #x_,y_,z_ = a[0],a[1],a[2]
    #ax.plot(x_, y_, z_, label='parametric curve')    

    ax.legend()
    plt.show()
