import os

import numpy
from numpy import pi, cos, sin, linspace, sign, sqrt

import scipy
import matplotlib
import matplotlib.lines as mlines

from matplotlib import pyplot as plt
matplotlib.interactive(True)

from mpl_toolkits.mplot3d import Axes3D

from copy import deepcopy

import load_data
reload(load_data)
from load_data import data, dcc_base

import basevectornew_numpy
reload(basevectornew_numpy)
from basevectornew_numpy import draw_all_basis

#   hiro_cartpos_process_all_traj.py (show trajs)
# - hiro_cartpos_code_visualize.py (colormaps)
# - hiro_cartpos_git_updater.py (frames on path)

#hiro_cartpos_code_visualize.py
def codeshow(data,ka,kb,type_,it):
    s = data[ka][kb]
    ss = []
    minlen = numpy.inf

    DCC_BASE_NUM = dcc_base[it].shape[0]    
    typename_ = type_+'_'+str(DCC_BASE_NUM)
    for i in s:
        codestr = s[i][typename_]
        ss.append(deepcopy(codestr))
        if len(codestr)<minlen:
            minlen = len(codestr)
        #plt.plot(s[i],'o-')

    print minlen,ka,kb
    for i in xrange(len(ss)):
        ss[i] = ss[i][:minlen]

    nss = numpy.array(ss)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.imshow(nss, interpolation='none', extent=[0,minlen,0,len(ss)], aspect='auto', cmap='afmhot')

    typename_ = typename_.replace(type_,type_.split('_')[0])
    
    ax.set_title(ka+' ['+kb+']_'+typename_)
    os.system('mkdir -p codemap')
    fig.savefig('./codemap/'+ka+'['+kb+']_'+typename_+'.png')
    plt.close(fig)



#hiro_cartpos_git_updater.py
def drawframes(r,frames,ax):
    assert r.shape[0] == frames.shape[0]
    #totallen = numpy.linalg.norm(numpy.diff(r,axis=0),axis=1).min()
    seglen = numpy.linalg.norm(numpy.diff(r,axis=0),axis=1)
    #seglenm = seglen.argmax()
    #seglen[seglenm] = seglen[seglenm-1]
    #minlen = seglen.min()
    totallen = seglen.sum()
    c = r.shape[0]
    numbers = 50
    assert c > numbers
    step = c/numbers
    #ax2 = fig.add_subplot(1,2,2)
    #ax2.set_ylim((0,0.0003))
    #ax2.scatter(numpy.arange(c-1),seglen)
    r_ = numpy.transpose(numpy.tile(r,(3,1,1)),axes=(1,0,2))
    frames_ = r_ + frames*(totallen/(c-1)*(step/2))#/3)#1000)
    
    tnb = numpy.transpose(
        numpy.concatenate([[r_],[frames_]],axis=0),
        axes = (2,1,3,0))

    t = tnb[0]
    n = tnb[1]
    b = tnb[2]

    color = 'rgb'

    for i in xrange(0,c,step):
        for j in xrange(3):
            ax.plot(
                tnb[j][i][0],
                tnb[j][i][1],
                tnb[j][i][2],
                color=color[j])


def showit():
    draw_all_basis()
    for i in data:
        for j in data[i]:
            for k in xrange(3):
                codeshow(data,i,j,'FF_code',k)
                codeshow(data,i,j,'AFF_code',k)
    #plt.show()

showit()
