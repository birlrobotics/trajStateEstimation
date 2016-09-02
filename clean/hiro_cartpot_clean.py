#!/usr/bin/python
print 'importing...'
import numpy
from numpy import pi, cos, sin, linspace, sign, sqrt
from sklearn import svm
import time
import random

#import load_data
#reload(load_data)
from load_data import *
print 'OK.\n'


# problem[1]: didn't consider the time (can't handle `halt` points)
# solution[1]: Use 4-dimensional arrays (with unit step in time axis)
#
# problem[2]: 
# solution[2]: 
#
# TODO: Create a new resample() method to replace this
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
    o_seg_len_nonzero = 0
    for i in xrange(1,new_n-1):
        while(o_acc_len<step):
            o_ptr += 1
            o_acc_len += len_dr[o_ptr]
            if len_dr[o_ptr]:
                o_seg_len_nonzero = len_dr[o_ptr]
        assert o_seg_len_nonzero,"%f %f\n%s"%(o_acc_len,step,repr(r))
        pos_percentage = (o_seg_len_nonzero-(o_acc_len-step))/o_seg_len_nonzero
        rr[i] = r[o_ptr-1] + dr[o_ptr]*pos_percentage
        o_acc_len -= step

    return rr


# 4D interpolation of 3D trajectories
# not ready yet.
def resample_time(_r,npoint,times):
    assert isinstance(_r, numpy.ndarray)
    assert isinstance(times, numpy.ndarray)
    assert len(_r.shape) == 2
    assert len(times.shape) == 1
    assert _r.shape[0] == times.shape[0]
    l = _r.shape[0]
    dim = _r.shape[1]
    r = numpy.zeros((l,dim+1))
    r[:,:dim] = _r
    r[:, dim] = times
    return resample(r,npoint)[:,:dim]

clfname = "svc, lin_svc, rbf_svc, poly_svc".split(', ')

def main(data_name, align="cut",encode="FF",level="task",subset=False,base=1):
#def main(data_name, align="cut",subset=True):
    '''
@param data_name (str): ["A","B","C"] and any combination of them
@param align (str): ["cut","interp","extend","dtw"(not implemented yet)]
@param encode (str): ["FF","AFF"], use FF or AFF
@param level (str): ["task", "behavior"]
@param subset (bool): (Not implemented)
      Whether to separate datasets into smaller subsets or not. 
      True for separation.
##@param svmimpl (str): ["SVC_linear", "SVC_RBF", "SVC_Polynomial", "LinearSVC"]
##      implementation of svm, 
##      more implementations should be added in the future.
@param base (int): which base to choose
'''

    global clfname

    test_result = {
        "align" : align,
        "encode" : encode,
        "level" : level,
        "subset" : subset,
        "dataset" : data_name,
        "avg_len" : None,
        "min_len" : None,
        "max_len" : None,
        "accuracy" : dict.fromkeys(clfname), # result table
    }

    #===================================
    #
    # SOME BEHAVIOR SEGMENTS HAVE VERY SMALL LENGTHS
    # WHICH SHOULD BE IGNORED DIRECTLY
    # USE THIS `r_len_threshold` PARAMETER TO 
    # CONSTRAIN THE LENGTH OF THE STRINGS TO BE TRAINED & CLASSIFIED
    # 
    #===================================
    r_len_threshold = 100

    testtime = 10
    kfoldmax = 20    
    for i in test_result['accuracy']:
        test_result['accuracy'][i] = dict.fromkeys(xrange(2,kfoldmax+1))

    assert align in ("cut","interp","extend")
    assert encode in ("FF","AFF")
    assert level in ("task","behavior")
    
    assert(type(data_name) is str)
    global data        
    global data_name_map  

    data_list = []
    for i in data_name:
        data_list.append(data_name_map[i])

    lengths = []
    typeorder = []
    classCount = 0
    testX = []
    Xtimes = []
    testX_precomputed_code = []
    testY = []
    caterange_min = 0


    print "================================="
    print "data selected :",data_name

    if level == 'behavior':
        cls_behavior = True
    elif level == 'task':
        cls_behavior = False      

    code_type = '_'.join([encode,'code',str(dcc_base[base].shape[0])])+('_split' if cls_behavior else '')

    for i in data_list:
        for j in data[i]:
            typeorder.append(i+'-'+j)
            if cls_behavior:
                maxcatek = 0
            else:
                typeorder.append(i+'-'+j)
            for k in data[i][j]:
                dataijk = data[i][j][k]
                if cls_behavior:
                    ncatek = len(dataijk['r_split'])
                    if(maxcatek < ncatek):
                        maxcatek = ncatek
                if cls_behavior:
                    for l in xrange(ncatek):
                        newdata = dataijk['r_split'][l]
                        if(newdata.shape[0]<r_len_threshold):
                            continue
                        testX.append(newdata)
                        Xtimes.append(dataijk['time_split'][l])
                        testX_precomputed_code.append(dataijk[code_type][l])
                        testY.append(caterange_min+l)
                        lengths.append(newdata.shape[0])
                else:
                    newdata = dataijk['r']                               
                    if(newdata.shape[0]<r_len_threshold):
                        continue
                    Xtimes.append(dataijk['time'])
                    testX.append(newdata)
                    testX_precomputed_code.append(dataijk[code_type])
                    testY.append(len(typeorder)-1)
                    lengths.append(newdata.shape[0])
            if cls_behavior:
                caterange_min += maxcatek

    avglen = sum(lengths)/len(lengths)
    minlen = min(lengths)
    maxlen = max(lengths)

#===================================
#
#   IS THIS THRESHOLD NECCESSARY?
#   HOW TO SET `minlen_threshold`?
# 
#===================================
    minlen_threshold = r_len_threshold*5 #500
    if minlen < minlen_threshold:
        minlen = minlen_threshold

    
    base_size = str(dcc_base[base].shape[0])
    
    #code_name = encode+'_code'+base_size
    
    if encode=="FF":
        process__ = FFencode
    elif encode=='AFF':
        process__ = AFFencode

    if align == "interp":
        #testX = map(lambda x:process__(resample(x,avglen))[code_name],testX)
        testX = map(lambda x:process__(resample_time(x[0],avglen,x[1])),zip(testX,Xtimes))
    elif align == "cut":
        #testX = map(lambda x:process__(x[:minlen] if len(x)>=minlen else resample(x,minlen))[code_name],testX)
        #testX = map(lambda x:testX_precomputed_code[x[0]][:minlen] if len(x[1])>=minlen else process__(resample(x[1],minlen)),enumerate(testX))
        testX = map(lambda x:process__(x[0][:minlen] if len(x[0])>=minlen else resample_time(x[0],minlen,x[1])),zip(testX,Xtimes))
    elif align == "extend":
        testX = map(lambda x:process__(numpy.array(x.tolist()+[[0,0,0]]*(maxlen-x.shape[0]))),testX)


    #print testX_precomputed_code[0]

    print 'avg:',avglen
    print 'min:',minlen
    print 'max:',maxlen
    test_result['avg_len'] = avglen
    test_result['min_len'] = minlen
    test_result['max_len'] = maxlen


    X_ = numpy.array(testX)
    Y_ = numpy.array(testY)
    totalnum = Y_.shape[0]

    

    for i in clfname:
        print i,
        #print nvalpass[i],
    print ""
    nvalpass = dict.fromkeys(clfname,0)
    time_st = time.clock()


    if True:
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

                test_result['accuracy'][iname][kfold] = nvalpass[i]/float(ntest*testtime)*100.0

                #print "%]",
                print ",",
            print ""

    return test_result


def proc_all_dataset(align,encode,level,base):
    #for subset in [True, False]:
    results = []
    main_ = lambda x: main(
        x,
        align=align,
        encode=encode,
        level=level,
        subset=False,
        base=base,
    )
    #if (level=='behavior'):
    data_name__ = '_'.join(map(str,["result",align,encode,level,base,"part",""]))
    main_dump = lambda x: dumper.save_load(
        data_name__+x+'.pydump.result',
        data=None,
        mode=None,
        datagen=main_,
        param={'x':x},
        dataname=data_name__+x,
    )

    if (level=='behavior'):
        results.append(main_dump("A"))
    results.append(main_dump("B"))
    results.append(main_dump("C"))
    results.append(main_dump("AB"))
    results.append(main_dump("AC"))
    results.append(main_dump("BC"))
    results.append(main_dump("ABC"))

    return results
    

'''
Design: 
   - Save all of the results into a large variable called `result`,
     and save it to a file.
   - Use a specific function to output the result (online & offline).
   - Use text file currently, but database in the future.
'''

def all_config_process(base):
    '''
@param data_name (str): ["A","B","C"] and any combination of them
@param align (str): ["cut","interp","dtw"(not implemented yet)]
@param encode (str): ["FF","AFF"], use FF or AFF
@param level (str): ["task", "behavior"]
@param subset (bool): 
      Whether to separate datasets into smaller subsets or not. 
      True for separation.
##@param svmimpl (str): ["SVC_linear", "SVC_RBF", "SVC_Polynomial", "LinearSVC"]
##      implementation of svm, 
##      more implementations should be added in the future.
'''
    all_config_result = []
    for align in ['extend','cut','interp']:
        for encode in ['FF','AFF']:
            for level in ['task','behavior']:
                data_name__ = '_'.join(map(str,["result",align,encode,level,base]))
                one_conf_res = dumper.save_load(
                    data_name__+'.pydump.result',
                    data=None,
                    mode=None,
                    datagen=proc_all_dataset,
                    param={
                        'align':align,
                        'encode':encode,
                        'level':level,
                        'base':base,
                    },
                    dataname=data_name__,
                )
                all_config_result.append(one_conf_res)
    return all_config_result


if __name__ == "__main__":    
    base = 1
    all_config_res = dumper.save_load(
        "all_config_result"+str(base)+".pydump.result",
        data=None,
        mode=None,
        datagen=all_config_process,
        param={"base":base},
        dataname="All config results",
    )
