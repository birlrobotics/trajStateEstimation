#!/usr/bin/python
print "importing...",
import os
import cPickle
from collections import Iterable
import numpy
print "ok"

import ori_data_read
reload(ori_data_read)
from ori_data_read import *

# Encode corrected data with DCC using FF and AFF
# IF the `reprocess` in main process is True, 
# do the encoding once again and save results.
# ELSE load pre-processed results from files.

def normalize(v,axis=1): #vectors, shape=(n,3)
    v_norm = numpy.linalg.norm( v, axis=axis )
    nonzero = v_norm != 0 # zero may exists
    v[nonzero] /= numpy.expand_dims(v_norm,2)[nonzero] 
    return v


def basegenOne(t, n, b):
    nb = -b
    nt = -t
    nn = -n
    # DCC19 base directions
    base = numpy.transpose(
        numpy.dstack([
            numpy.zeros(t.shape),
            t,n,b,nb,nn,nt,
            n +t, n +nt, nn+t, nn+nt,
            b +t, b +nt, b +n, b +nn,
            nb+t, nb+nt, nb+n, nb+nn
        ]),
        axes=(0,2,1))
    #print base
    #b_s = base
    #return base
    normalize(base,axis=2)
    return base[0]

# T == 0 : keep base
# T != 0 :
#     N == 0 : keep N
#     N != 0 : nice
# B = T x N
def basegen(t, n, b):
    naxis = len(t.shape)
    nb = -b
    nt = -t
    nn = -n
    # DCC19 base directions
    base = numpy.transpose(
        numpy.dstack([
            numpy.zeros(t.shape),
            t,n,b,nb,nn,nt,
            n +t, n +nt, nn+t, nn+nt,
            b +t, b +nt, b +n, b +nn,
            nb+t, nb+nt, nb+n, nb+nn
        ]),
        axes=(0,2,1))
    #print base
    #b_s = base
    #return base
    normalize(base,axis=naxis)

    tz = numpy.linalg.norm(t,axis=naxis-1) == 0
    #nz = numpy.linalg.norm(n,axis=naxis-1) == 0
    #bz = numpy.linalg.norm(b,axis=naxis-1) == 0
    iszero = tz #+nz+bz #(tz | nz | bz)
    
    c = t.shape[0]
    i = 0
    while i < c:
        if not iszero[i]:
            for j in xrange(i-1,-1,-1):
                base[j] = base[i]
            break
        i += 1

    while i < c:
        if iszero[i]:
            base[i] = base[i-1]
        i += 1
        
    return base


# Discrete fenet frame T (tangent vector)
def gett(r): # r is the curve, r.shape = (n,3)
    assert len(r.shape) is 2
    assert r.shape[1] is 3

    c = r.shape[0] # count
    t = numpy.zeros(r.shape)
    t[:c-1] = r[1:] - r[:c-1]
    t[c-1] = t[c-2]

    # normalize t
    normalize(t)
    return t


# Discrete Frenet frame B (binormal vector)
def getb(t): # t is the tangent vector
    assert len(t.shape) is 2
    assert t.shape[1] is 3
    
    c = t.shape[0]
    b = numpy.zeros(t.shape)
    b[1:] = numpy.cross(t[:c-1],t[1:]) #
    b[0] = b[1]
    
    normalize(b)

    norm = lambda x:numpy.linalg.norm(x,axis=1)
    keepn = (norm(b)==0)#*(norm(t)!=0) # straight line

    c = t.shape[0]
    i = 0
    while i < c:
        if not keepn[i]:
            for j in xrange(i-1,-1,-1):
                b[j] = b[j+1]
            break
        i += 1

    while i < c:
        if keepn[i]:
            b[i] = b[i-1]
        i += 1
    
    b[norm(t)==0] = numpy.zeros((3,))
    return b




def encode(r,t,base):
    assert t.shape[0] == base.shape[0]
    t_ = numpy.transpose(numpy.tile(t,(19,1,1)),axes=(1,0,2))
    c = base.shape[0]
    nbase = base.shape[1]
    dbase = numpy.zeros((c-1,nbase))
    dbase = numpy.sum(t_[1:]*base[:c-1],axis=2)
    # dbase0: absolute
    # base[0][1] is T vector
    #dbase[0] = numpy.zeros((19,))
    return dbase.argmax(axis=1)


def FFencode(r):
    t = gett(r)
    b = getb(t)
    n = numpy.cross(b,t)
    ## DCC19 base directions
    base = basegen(t,n,b)
    ## DFF uses difference as tangent vector directly.
    code = encode(r,t,base)
    #return dict(zip("t,n,b,base,code".split(','),(t,n,b,base,code)))
    return code

def AFFencode(r):
    code = numpy.zeros((r.shape[0]-1,))
    t = gett(r)

    b0 = numpy.cross(r[0],r[1])
    normalize(b0.reshape(1,3))
    n0 = numpy.cross(b0,t[0])
    normalize(n0.reshape(1,3))
    base0 = basegenOne(t[0],n0,b0)

    pi_d_2 = numpy.pi / 2.0
    norm = numpy.linalg.norm
    
    npoint = r.shape[0]
    for i in xrange(1,npoint):
        n_i_1 = norm(base0[1])
        n_i = norm(t[i])
        if(not (n_i_1 and n_i)):
            angle = 0
        else:
            cos_ = numpy.dot(base0[1],t[i])/(n_i_1*n_i)
            if(numpy.abs(cos_)>=1):
                cos_ = 1 if cos_>0 else -1
            angle = numpy.arccos(cos_)
        if(angle > pi_d_2):
            #b[i] = numpy.cross(r[i-1],r[i])
            #normalize(b[i].reshape(1,3))
            #n[i] = numpy.cross(b[i],t[i])
            #normalize(n[i].reshape(1,3))
            #base[i] = basegenOne(t[i],n[i],b[i])
            b1 = numpy.cross(t[i-1],t[i])
            normalize(b1.reshape(1,3))
            n1 = numpy.cross(b1,t[i])
            normalize(n1.reshape(1,3))
            base1 = basegenOne(t[i],n1,b1)
            code[i-1] = numpy.argmax(numpy.dot(base0,t[i]))
            print i,
        else:
            base1 = base0
            code[i-1] = 1

        base1,base0 = base0,base1
    print ""
    return code



def save_code_FF(data,i,j,k,dataijk,fullpath,alldata,timestamp):
    r = dataijk['r']
    proccode = FFencode(r)
    dataijk['FF_code'] = proccode
    splitc = []
    timestamp_i = dataijk['state_stamp']
    for tsi in xrange(1,len(timestamp_i)):
        splitc.append(proccode[timestamp_i[tsi-1]:timestamp_i[tsi]])
    dataijk['FF_code_split'] = splitc

def save_code_AFF(data,i,j,k,dataijk,fullpath,alldata,timestamp):
    r = dataijk['r']
    proccode = AFFencode(r)
    dataijk['AFF_code'] = proccode
    splitc = []
    timestamp_i = dataijk['state_stamp']
    for tsi in xrange(1,len(timestamp_i)):
        splitc.append(proccode[timestamp_i[tsi-1]:timestamp_i[tsi]])
    dataijk['AFF_code_split'] = splitc

# return: data,filearray
def data_read_full():
    return data_read(redump=False, process=[
        save_correct_r,
        save_code_AFF,
        save_code_FF,
    ])



if __name__ == "__main__":
    reprocess = False
    fdatafile = 'datafile.cpk'
    ffilearray = 'filearray.cpk'
    if reprocess:
        data,filearray = data_read_full()
        dumpdata(data,fdatafile)
        dumpdata(filearray,ffilearray)
        print "Filenames saved"
    else:
        data = loaddata(fdatafile)
        filearray = loaddata(ffilearray)
        print "Filenames loaded"


