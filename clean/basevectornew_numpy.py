#!/usr/bin/python

print "importing..."
import numpy
from numpy import pi, cos, sin, linspace, sign, sqrt
import scipy
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
matplotlib.interactive(True)
#from mayavi import mlab
#from mayavi.mlab import plot3d,quiver3d

##set parallel projection as window default
#mlab.gcf().scene.parallel_projection = True

from copy import deepcopy


#import datadumper
#reload(datadumper)
from datadumper import dumper

def base_sort_cmp(x,y):
    for i in xrange(3):
        if x[i]>y[i]:
            return -1
        elif x[i]<y[i]:
            return 1
    return 0

def normalize(v,axis=1): #vectors, shape=(n,3)
    v_norm = numpy.linalg.norm( v, axis=axis )
    nonzero = v_norm != 0 # zero may exists
    v[nonzero] /= numpy.expand_dims(v_norm,2)[nonzero] 


def extend_base(ob):
    nb = [i.copy() for i in ob]
    l = len(ob)
    new1s = set()
    o1s = set(map(tuple,nb))
    for i in xrange(l):
        for j in xrange(i,l):
            new1 = ob[i] + ob[j]
            normnew1 = numpy.linalg.norm(new1)
            if not normnew1:
                continue
            new1 /= normnew1
            tnew1 = tuple(new1)
            if tnew1 in new1s or tnew1 in o1s:
                continue
            new1s.add(tnew1)
    return nb+map(numpy.array,new1s)


def neighbour_filter(base,threshold):
    base = map(tuple,base)
    base.sort(base_sort_cmp)
    lbase = len(base)
    similar1s = [0]*lbase #boolean flags
    basenparr = numpy.array(base)
    dif = numpy.linalg.norm(basenparr[1:]-basenparr[:lbase-1],axis=1)
    simtype = 1

    for i in xrange(1,lbase):
        if dif[i-1] < threshold:
            similar1s[i-1] = simtype
            similar1s[i] = simtype
        elif similar1s[i-1]:
            simtype += 1
        else:
            pass

    if similar1s[lbase-1]:
        simtype += 1

    #print l,similar1s.count(0),len(set(similar1s))-1,similar1s.count(0)+len(set(similar1s))-1+l
    #print similar1s
    #print basenparr

    #print simtype,"---\n"#,similar1s
    simsets = dict.fromkeys(xrange(simtype))
    for i in simsets:
        simsets[i] = set()
    for i in xrange(lbase):
        #print similar1s[i],
        simsets[similar1s[i]].add(base[i])
    #print ""
    rbase = list(simsets[0])
    for i in xrange(1,simtype):
        rbase.append(numpy.average(list(simsets[i]),axis=0))
        
    return map(numpy.array,rbase)


def sim_assert(it,base,threshold):
    lbase = len(base)
    tbase = map(tuple,base)
    npabase = numpy.array(base)
    tilebase = numpy.tile(base,(lbase,1,1))
    difmat = tilebase - numpy.transpose(tilebase,axes=(1,0,2))
    simat = numpy.linalg.norm(difmat,axis=2)
    simat = simat<threshold

    n_neighbour_close = numpy.count_nonzero(numpy.linalg.norm(npabase[1:]-npabase[:lbase-1],axis=1))
    assert n_neighbour_close==lbase-1, "%d close base vector pairs detected in DCC%d, iter %d"%(n_neighbour_close,lbase,it)
    n_global_close = numpy.count_nonzero(simat)
    assert n_global_close==lbase, "%d close base vector pairs detected in DCC%d, iter %d"%(n_global_close,lbase,it)

def global_filter(base, threshold):
    lbase = len(base)
    tbase = map(tuple,base)
    npabase = numpy.array(base)
    tilebase = numpy.tile(base,(lbase,1,1))
    difmat = tilebase - numpy.transpose(tilebase,axes=(1,0,2))
    simat = numpy.linalg.norm(difmat,axis=2)
    simat = simat<threshold

    if False:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(simat,cmap='gray',interpolation='none')

    simcount = 0
    simsets = {}
    simarr = numpy.zeros(lbase) # reverse index of `simsets`
    rangarr = numpy.array(xrange(lbase))
    for i in xrange(lbase):
        sim1s = rangarr[i:][simat[i:,i]]
        if not simarr[i]:
            simcount += 1
            simsets[simcount] = set([i])
            typenum = simcount
        else:
            typenum = simarr[i]
        simarr[sim1s] = typenum
        for j in sim1s:
            simsets[typenum].add(j)

    rbase = []
    for i in simsets:
        rbase.append(numpy.average(npabase[list(simsets[i])],axis=0))
    
    return rbase


def remove_overlap(base,threshold):
    base = neighbour_filter(base,threshold)
    base = global_filter(base,threshold)
    return base


def gen_base_numpy(it,threshold):
    dcc_base = [[numpy.zeros(3)]]

    x,y,z = numpy.eye(3)
    for i in [x,y,z]:
        dcc_base[0].append(i)
        dcc_base[0].append(-i)

    for i in xrange(it):
        dcc_base.append(remove_overlap(extend_base(dcc_base[i]),threshold))

    for i in xrange(len(dcc_base)):
        dcc_base[i].sort(base_sort_cmp)
        dcc_base[i] = numpy.array(dcc_base[i])

    print map(len,dcc_base)
    return dcc_base



def draw_base(it=3, oct=False):

    global dcc_base

    fig = plt.figure()
    plt.hold(True)
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
    plt.close()
    return base





dcc_base_numpy_dump_file_name = "dcc_base_numpy.pydump"

def get_dcc_basis_numpy(it, threshold):
    global dcc_base_numpy_dump_file_name
    return dumper.save_load( 
        dcc_base_numpy_dump_file_name, 
        data=None,
        mode=None,
        datagen=gen_base_numpy,
        param={'it':it, 'threshold':threshold},
        dataname="DCC basis using numpy",
    )
    
itc = 3
dcc_base = get_dcc_basis_numpy(itc,1e-6)



def draw_all_basis(itc=3):
    for i in xrange(itc+2):
        base = draw_base(i)
        base = draw_base(i,oct=True)
    

if __name__ == "__main__":
    #print dcc_base
    matplotlib.rcParams['legend.fontsize'] = 10
    draw_all_basis(itc)    
    
    errth = 1e-1  # error threshold
    for i in xrange(itc):
        sim_assert(i,dcc_base[i],numpy.pi/numpy.power(2,i+2)*errth)
        
    pass

