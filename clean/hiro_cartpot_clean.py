#!/usr/bin/python
print 'importing...'
import numpy
from numpy import pi, cos, sin, linspace, sign, sqrt
from sklearn import svm
import time
import random

import load_data
reload(load_data)
from load_data import *
print 'OK.\n'

data_name_map = {
    'A' : 'data_003_SIM_HIRO_SA_Success',
    'B' : 'data_004_SIM_HIRO_SA_ErrorCharac_Prob',
    'C' : 'data_008_HIRO_SideApproach_SUCCESS',
}


def resample(r,npoint):
    assert len(r.shape)==2
    old_n = r.shape[0]
    new_n = npoint

    # length of the path
    dr = numpy.zeros(r.shape)
    dr[1:] = r[1:]-r[:old_n-1]
    len_dr = numpy.linalg.norm( dr, axis=1 )
    len_total = len_dr.sum()
    step = len_total / (new_n-1)

    # new path
    rr = numpy.zeros((npoint,r.shape[1]))
    rr[[0,new_n-1]] = r[[0,old_n-1]]
    
    o_ptr = 0 # point counter on old path
    o_acc_len = 0 # accumulated length on old path
    for i in xrange(1,new_n-1):
        while(o_acc_len<step):
            o_ptr += 1
            o_acc_len += len_dr[o_ptr]
        pos_percentage = (len_dr[o_ptr]-(o_acc_len-step))/len_dr[o_ptr]
        rr[i] = r[o_ptr-1] + dr[o_ptr]*pos_percentage
        o_acc_len -= step

    return rr





def main(data_name):
    assert(type(data_name) is str)
    global data        
    global data_name_map  

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

    cls_behavior = True

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
    main("A")
    main("B")
    main("C")
    main("AC")
    main("BC")
    main("ABC")


