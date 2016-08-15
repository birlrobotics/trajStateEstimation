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

#import load_data
#reload(load_data)
from load_data import data, dcc_base

#import basevectornew_numpy
#reload(basevectornew_numpy)
from basevectornew_numpy import draw_all_basis

#   hiro_cartpos_process_all_traj.py (show trajs)
# - hiro_cartpos_code_visualize.py (colormaps)
# - hiro_cartpos_git_updater.py (frames on path)

#hiro_cartpos_code_visualize.py
def codeshow(data,ka,kb,type_,it,cut=True,show=False):
    s = data[ka][kb]
    ss = []
    minlen = numpy.inf

    DCC_BASE_NUM = dcc_base[it].shape[0]    
    
    

    typename_ = type_+'_code_'+str(DCC_BASE_NUM)
    for i in s:
        codestr = s[i][typename_]
        ss.append(deepcopy(codestr))

    if cut:
        minlen = len(reduce(lambda x,y: x if len(x)<len(y) else y,ss))
        #for codestr in ss:
        #    if len(codestr)<minlen:
        #        minlen = len(codestr)
            #plt.plot(s[i],'o-')

        print ka,kb,minlen,type_,DCC_BASE_NUM
        
        for i in xrange(len(ss)):
            ss[i] = ss[i][:minlen]
    
    else:
        maxlen = len(reduce(lambda x,y:x if len(x)>len(y) else y,ss))

        print ka,kb,maxlen,type_,DCC_BASE_NUM

        for i in xrange(len(ss)):
            lssi = len(ss[i])
            #print ss[i],type(ss[i])
            ss[i] = ss[i].tolist()+[0]*(maxlen-lssi)

    nss = numpy.array(ss)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.imshow(nss, interpolation='none', extent=[0,minlen if cut else maxlen,0,len(ss)], aspect='auto', cmap='afmhot')

    typename_ = typename_.replace('_code','')
    
    figure_title = ka+' ['+kb+']_'+typename_+('_cut' if cut else '_full')
    ax.set_title(figure_title)
    image_path = '/'.join(['./codemap',str(DCC_BASE_NUM),('cut' if cut else 'full'),type_,''])
    os.system('mkdir -p '+image_path)
    fig.savefig(image_path+figure_title+'.png')
    if not show:
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
    if False:
        draw_all_basis()
    if True:
        for i in data:
            for j in data[i]:
                for k in xrange(3):
                    #codeshow(data,i,j,'FF',k,show=True)
                    #return 
                    codeshow(data,i,j,'FF',k)
                    codeshow(data,i,j,'AFF',k)
                    codeshow(data,i,j,'FF',k,cut=False)
                    codeshow(data,i,j,'AFF',k,cut=False)

    #plt.show()

showit()
