#!/usr/bin/python
'''
Usage: 
    Plot the HIRO `CartPos*.dat` XYZ data. 
    Based on `newtanvec.py`, removed redundant code.
    Change the `path` variable to set the HIRO data location.
Last Edit: 2016/05/05 Thu
'''
print "importing...",
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

import cPickle

from sklearn import svm

import time

import random

print "ok"






tasktype_tree = {
    'data_003_SIM_HIRO_SA_Success': ['2012'],
    'data_004_SIM_HIRO_SA_ErrorCharac_Prob': ['FC','exp'],
    'data_008_HIRO_SideApproach_SUCCESS':['2012','x']
}

'''
    'find -maxdepth 3 -name "CartPos*" |grep "\./FC0[^/]*/CartPos*"'
'''

root = '~/temp/apriltest/data'
root = os.popen('echo '+root).read().strip() # get the absolute path

def findfilenames(root):
    filename = tasktype_tree.fromkeys(tasktype_tree.keys())
    for i in filename:
        filename[i] = {}
        for j in tasktype_tree[i]:
            path = root+'/'+i+'/'
            CartFileList = os.popen('cd '+path+' && find . -maxdepth 3 -name "CartPos*" |grep "\./'+j+'.*/CartPos*"').read().strip('./').strip().split('\n./') # find all CartPos*.dat files
        
            #        CartFileList = os.popen('find '+path+' -maxdepth 3 -name "CartPos*" |grep "'+i+'/'+j+'.*/CartPos*"').read() # find all CartPos*.dat files

            print path
            CartFileList.sort(reverse=True)
            PathList = dict(map(lambda x:(x[:x.find("CartPos")],x[x.find("CartPos"):]),CartFileList))
            CartFileList = PathList.values()
            PathList = PathList.keys()
            len_filelist = len(PathList)
            StateFileList = [None]*len_filelist
            for li in xrange(len_filelist):
                StateFileList[li] = os.popen('cd "'+path+PathList[li]+'" && find . -name "State*.dat"|grep "\./State"').read().strip('./').strip().split()[0]
                filename[i][j] = dict(zip(PathList,map(lambda x:{"CartPos":x[0],"State":x[1]},zip(CartFileList,StateFileList))))
    return filename





filenames_save = "filenames.txt"
if False:
    data= findfilenames(root)
    ftemp = open(filenames_save,"w")
    cPickle.dump(data,ftemp)
    ftemp.close()
    print "Filenames saved"
else:
    ftemp = open(filenames_save,"r")
    data = cPickle.load(ftemp)
    ftemp.close()
    print "Filenames loaded"

codedir = 'trajcode'
alldatafile = "alldata.txt"


def reader(filename):
    lines = file(filename,'r').readlines();
    data = numpy.array(
        map(
            lambda x:map(
                lambda x:float(x),
                x[:len(x)-2].split('\t'))
            ,lines))
    return data

def normalize(v,axis=1): #vectors, shape=(n,3)
    v_norm = numpy.linalg.norm( v, axis=axis )
    nonzero = v_norm != 0 # zero may exists
    v[nonzero] /= numpy.expand_dims(v_norm,2)[nonzero] 


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


# Discrete fenet frame B (binormal vector)
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


def rebuild(pos, ori, code):
    c = code.shape[0]
    r = numpy.zeros((c,3))
    r[0] = pos
    for i in xrange(1,c):
        #bs = basegen(*numpy.array([[[1,0,0]], [[0,1,0]]]))
        pass
    
# [ WARNING ] THIS IS Frenet Frame encoding, not Accumulated Frenet Frame
# code.shape == r.shape[0] - 1
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


def process(r):
    t = gett(r)
    b = getb(t)
    n = numpy.cross(b,t)
    #DCC19 base directions
    base = basegen(t,n,b)
    #DFF uses difference as tangent vector directly.
    code = encode(r,t,base)
    return dict(zip("t,n,b,base,code".split(','),(t,n,b,base,code)))


def correctR(r):
    l = r.shape[0]
    dr = numpy.zeros(r.shape)
    dr[1:] = r[1:] - r[:l-1] 
    normdr = numpy.linalg.norm(dr,axis=1)
    avglen = normdr.sum()/(l-1)

    n_shift = filter(lambda x:(x[0]-avglen)/avglen>5,zip(normdr,[i for i in xrange(normdr.shape[0])])) # (number, norm)
    n_shift.sort(reverse=True)

    if len(n_shift) is 0:
        return r

    n_shift=n_shift[0]

    print n_shift

    displmt = r[n_shift[1]]-r[n_shift[1]-1] #displacement

    assert numpy.linalg.norm(displmt) == n_shift[0]
    
    for i in xrange(n_shift[1],l):
        r[i] = r[i] - displmt
       
    return r

def resample(r,npoint):
    assert len(r.shape)==2 and r.shape[1]==3
    old_n = r.shape[0]
    new_n = npoint

    dr = numpy.zeros(r.shape)
    dr[1:] = r[1:]-r[:old_n-1]
    len_dr = numpy.linalg.norm( dr, axis=1 )
    len_total = len_dr.sum()
    step = len_total / (new_n-1)

    rr = numpy.zeros((npoint,3))
    rr[[0,new_n-1]] = r[[0,old_n-1]]
    
    o_ptr = 0
    o_acc_len = 0
    for i in xrange(1,new_n-1):
        while(o_acc_len<step):
            o_ptr += 1
            o_acc_len += len_dr[o_ptr]
        pos_percentage = (o_ptr-(o_acc_len-step))/o_ptr
        rr[i] = r[o_ptr-1] + dr[o_ptr]*pos_percentage
        o_acc_len -= step

    return rr
    

def savecodefile(data):
    falldata = open(alldatafile,"w")
    os.system('mkdir '+codedir)
    for i in data:
        os.system('mkdir '+codedir+'/'+i)
        for j in data[i]:
            os.system('mkdir '+codedir+'/'+i+'/'+j)
            for k in data[i][j]:
                f = open(codedir+'/'+i+'/'+j+'/'+k.replace('/','__'),'w')
                timestamp = map(float,open('/'.join([root,i,k,data[i][j][k]['State']]),"r").read().strip().split())
                alldata =reader('/'.join([root,i,k,data[i][j][k]['CartPos']]))
                r = alldata[:,1:4]
                correctR(r)
                timestamp_i = map(lambda x:alldata[:,0].searchsorted(x),timestamp)
                data[i][j][k]['state_stamp'] = timestamp_i
                data[i][j][k]['r'] = r
                print i,j,k
                proccode = process(r)['code']
                data[i][j][k]['code'] = proccode
                data[i][j][k]['length'] = r.shape[0]

                splitr = []
                splitc = []
                timestamp_i.append(r.shape[0])
                for tsi in xrange(1,len(timestamp_i)):
                    splitr.append(r[timestamp_i[tsi-1]:timestamp_i[tsi]])
                    splitc.append(proccode[timestamp_i[tsi-1]:timestamp_i[tsi]])

                data[i][j][k]['r_split'] = splitr
                data[i][j][k]['code_split'] = splitc

                tcode = ','.join([str(k) for k in proccode])
                f.write(tcode)
                f.close()
    cPickle.dump(data,falldata)
    falldata.close()
    print "code saved to ./"+codedir+". This is for intuitive analysis."
    print "all data saved to ./"+alldatafile+". data will be loaded from this file."
    return data

def loadcodefile():
    print "loading data from ./"+alldatafile
    return cPickle.load(open(alldatafile,"r"))


#if __name__ == "__main__":
def main(data_name):
    assert(type(data_name) is str)
    if False:
        data = savecodefile(data)
    else:
        data = loadcodefile()
    
    data_name_map = {
       'A' : 'data_003_SIM_HIRO_SA_Success',
       'B' : 'data_004_SIM_HIRO_SA_ErrorCharac_Prob',
       'C' : 'data_008_HIRO_SideApproach_SUCCESS',
    }
    
    data_list = []
    for i in data_name:
        data_list.append(data_name_map[i])

    lengths = []
    typeorder = []
    testX = []
    testY = []
    caterange_min = 0
#    for i in [
#            'data_008_HIRO_SideApproach_SUCCESS',
#            'data_003_SIM_HIRO_SA_Success',
#            'data_004_SIM_HIRO_SA_ErrorCharac_Prob',
#    ]:#data:
    #for i in data:
    print "================================="
    print "data selected :",data_name

    cls_behavior = False

    for i in data_list:
        for j in data[i]:
            typeorder.append(i+'-'+j)
            if cls_behavior:
                maxcatek = 0
            else:
                typeorder.append(i+'-'+j)
            for k in data[i][j]:
                if cls_behavior:
                    ncatek = len(data[i][j][k]['r_split'])
                    if(maxcatek < ncatek):
                        maxcatek = ncatek
                if cls_behavior:
                    for l in xrange(ncatek):
                        newdata = data[i][j][k]['r_split'][l]
                        if(newdata.shape[0]<100):
                            continue
                        testX.append(newdata)
                        testY.append(caterange_min+l)
                        lengths.append(newdata.shape[0])
                else:
                    newdata = data[i][j][k]['r']
                    #if(newdata.shape[0]<1000):
                    #    continue
                    testX.append(newdata)
                    testY.append(len(typeorder)-1)
                    lengths.append(newdata.shape[0])
            if cls_behavior:
                caterange_min += maxcatek

    avglen = sum(lengths)/len(lengths)
    minlen_threshold = 500
    minlen = min(lengths)
    if minlen < minlen_threshold:
        minlen = minlen_threshold
    
    testX = map(lambda x:process(resample(x,avglen))['code'],testX)

    print 'avg:',avglen
    print 'min:',minlen

    #testX = map(lambda x:process(x[:minlen] if len(x)>=minlen else resample(x,minlen))['code'],testX)
    X_ = numpy.array(testX)
    Y_ = numpy.array(testY)
    totalnum = Y_.shape[0]

    clfname = "svc, lin_svc, rbf_svc, poly_svc".split(',')

    for i in clfname:
        print i,
        #print nvalpass[i],
    print ""
    nvalpass = dict.fromkeys(clfname,0)
    time_st = time.clock()

    # N-fold cross validation
    if False:
        for ival in xrange(totalnum):

            X = numpy.delete(X_,ival,0)
            Y = numpy.delete(Y_,ival)
            print ival,"\t",X.shape,Y.shape,

            C = 1.0  # SVM regularization parameter
            svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
            rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
            poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
            lin_svc = svm.LinearSVC(C=C).fit(X, Y)

            for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
                #Z = clf.predict(X)
                #a = Z==Y
                Z = clf.predict([X_[ival]])
                if Z==Y_[ival]:
                    nvalpass[clfname[i]] += 1
                print clfname[i],#avgstrlen,
                print "(","PA  " if Z==Y_[ival] else "  FA", int(Z[0]), int(Y_[ival]),") ",
                #print (a.shape[0]-a.sum()),
                #print float(a.shape[0]),
                #print (a.shape[0]-a.sum())/float(a.shape[0])
                #for i,j in enumerate(a):
                #    if(not j):
                #        print i,Y[i]
            time_insec = int(time.clock()-time_st)#int((time.clock()-time_st)*100)
            time_inmin = time_insec/60
            time_insec = time_insec%60
            print time_inmin,"min",time_insec,"sec"
        
        for i in nvalpass:
            print i,nvalpass[i],totalnum,nvalpass[i]/float(totalnum)

    # k-fold validation
    elif False:
        testtime = 200
        kfold = 5
        ntest = totalnum/kfold
        ntrain = totalnum - ntest
        for ival in xrange(testtime):
            tests = random.sample(xrange(totalnum),ntest)
            trains = list(set(xrange(totalnum)).difference(tests))

            X = X_[trains]
            Y = Y_[trains]
            print ival,"\t",X.shape,Y.shape,

            C = 1.0  # SVM regularization parameter
            svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
            rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
            poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
            lin_svc = svm.LinearSVC(C=C).fit(X, Y)

            for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
                nvalpassin = 0
                #Z = clf.predict(X)
                #a = Z==Y
                Z = clf.predict(X_[tests])
                testY_ = Y_[tests] 
                for j in xrange(ntest):
                    if Z[j]==testY_[j]:
                        nvalpassin += 1
                nvalpass[clfname[i]] += nvalpassin
                print clfname[i],#avgstrlen,
                print "(","P",nvalpassin,"/", ntest,"-",int(nvalpassin/float(ntest)*100),"% ) ",
                #print (a.shape[0]-a.sum()),
                #print float(a.shape[0]),
                #print (a.shape[0]-a.sum())/float(a.shape[0])
                #for i,j in enumerate(a):
                #    if(not j):
                #        print i,Y[i]
            time_insec = int(time.clock()-time_st)#int((time.clock()-time_st)*100)
            time_inmin = time_insec/60
            time_insec = time_insec%60
            print time_inmin,"min",time_insec,"sec"
        
        for i in nvalpass:
            print i,nvalpass[i],ntest*testtime,nvalpass[i]/float(ntest*testtime)

    else:
        testtime = 10
        kfoldmax = 20
        for kfold in xrange(2,kfoldmax+1):
            ntest = totalnum/kfold
            ntrain = totalnum - ntest
            nvalpass = dict.fromkeys(clfname,0)
            print kfold,",",
            for ival in xrange(testtime):
                tests = random.sample(xrange(totalnum),ntest)
                trains = list(set(xrange(totalnum)).difference(tests))

                X = X_[trains]
                Y = Y_[trains]
                #print "\r\t  "+" "*50+"\r\t  ",
                #print ival,
                #print ",",
                #print "\t",X.shape,Y.shape,

                C = 1.0  # SVM regularization parameter
                svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
                rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
                poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
                lin_svc = svm.LinearSVC(C=C).fit(X, Y)

                for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
                    nvalpassin = 0
                    #Z = clf.predict(X)
                    #a = Z==Y
                    Z = clf.predict(X_[tests])
                    testY_ = Y_[tests] 
                    for j in xrange(ntest):
                        if Z[j]==testY_[j]:
                            nvalpassin += 1
                    nvalpass[clfname[i]] += nvalpassin
                    #print clfname[i],#avgstrlen,
                    #print "(","P",nvalpassin,"/", ntest,"-",int(nvalpassin/float(ntest)*100),"% ) ",
                    #print (a.shape[0]-a.sum()),
                    #print float(a.shape[0]),
                    #print (a.shape[0]-a.sum())/float(a.shape[0])
                    #for i,j in enumerate(a):
                    #    if(not j):
                    #        print i,Y[i]
                time_insec = int(time.clock()-time_st)#int((time.clock()-time_st)*100)
                time_inmin = time_insec/60
                time_insec = time_insec%60
                #print time_inmin,"min",time_insec,"sec",
                #print ""
            #print ""
            for iname in clfname:#nvalpass:
                i = iname
                #print i,
                #print nvalpass[i],
                #print ntest*testtime,
                #print '[',
                print str(int(nvalpass[i]/float(ntest*testtime)*100)),
                #print "%]",
                print ",",
            print ""


if __name__ == "__main__":
    #main("A")
    main("B")
    main("C")
    main("AC")
    main("BC")
    main("ABC")
