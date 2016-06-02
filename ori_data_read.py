#!/usr/bin/python
print "importing...",
import os
import cPickle
from collections import Iterable
import numpy
print "ok"

## This script provides a function which
## walks through 3 datasets and reads all 
## CartPos*.dat and States*.dat files
## Call `data_read()` to read.

## Directory containing all datasets
root = '~/temp/apriltest/data'
root = os.popen('echo '+root).read().strip() # get the absolute path

## keys: datasets
## values: subsets with different suffixes
tasktype_tree = {
    'data_003_SIM_HIRO_SA_Success': ['2012'],
    'data_004_SIM_HIRO_SA_ErrorCharac_Prob': ['FC','exp'],
    'data_008_HIRO_SideApproach_SUCCESS':['2012','x']
}

filenames_save = "HIRO_filenames.cpk"

def dumpdata(dat, filename):
    ftemp = open(filename,"w")
    cPickle.dump(dat,ftemp)
    ftemp.close()
    return dat

def loaddata(filename):
    ftemp = open(filename,"r")
    dat = cPickle.load(ftemp)
    ftemp.close()
    return dat


## list CartPos*.dat and States*.dat file names and paths in different directories. 
## Run `findfilenames(root)` to see the result
def findfilenames(root):
    filename = tasktype_tree.fromkeys(tasktype_tree.keys())
    for i in filename:
        filename[i] = {}
        for j in tasktype_tree[i]:
            path = root+'/'+i+'/'
            print path

            ## list all CartPos*.dat files
            CartFileList = os.popen('cd '+path+' && find . -maxdepth 3 -name "CartPos*" |grep "\./'+j+'.*/CartPos*"').read().strip('./').strip().split('\n./') 

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



def file_reader(filename):
    lines = file(filename,'r').readlines();
    data = numpy.array(
        map(
            lambda x:map(
                lambda x:float(x),
                x[:len(x)-2].split('\t'))
            ,lines))
    return data



def data_read(redump=True, process=[]):
    ## `process` is a list of functions to process data already read.
    ## All processing functions should record the results in the `data` structure
    if callable(process):
        process = [process]
    elif isinstance(process,list):
        process = filter(callable,process)
    else:
        process = []

    ## IF `redump` is true, then the datasets
    ## will be walked through once again
    ## and save var `filename`, 
    ## ELSE load var `filename`. 
    if redump:
        data = dumpdata(findfilenames(root),filenames_save)
        print "Filenames saved"
    else:
        data = loaddata(filenames_save)
        print "Filenames loaded"

    filearray = []
    global_counter = -1
    for i in data: ## dataset 
        for j in data[i]: ## subsets
            for k in data[i][j]: ## path to subsset
                fullpath = root+'/'+i+'/'+k
                global_counter += 1
                filearray.append((i,j,k,fullpath,global_counter))
                ## data reading
                dataijk = data[i][j][k]

                #timestamp = map(float,open('/'.join([root,i,k,data[i][j][k]['State']]),"r").read().strip().split())
                #alldata =file_reader('/'.join([root,i,k,data[i][j][k]['CartPos']]))
                timestamp = map(float,open(fullpath+'/'+dataijk['State'],"r").read().strip().split())
                alldata =file_reader(fullpath+'/'+dataijk['CartPos'])

                r = alldata[:,1:4]

                ## Actually I did the correct here
                #correctR(r)
                #alldata[:,1:4] = r

                timestamp_i = map(lambda x:alldata[:,0].searchsorted(x),timestamp)
                #print type(timestamp_i)
                dataijk['state_stamp'] = timestamp_i
                dataijk['r'] = r
                dataijk['length'] = r.shape[0]
                #print i,j,k

                splitr = []
                timestamp_i.append(r.shape[0])
                for tsi in xrange(1,len(timestamp_i)):
                    splitr.append(r[timestamp_i[tsi-1]:timestamp_i[tsi]])

                data[i][j][k]['r_split'] = splitr

                ## Further process (potential)
                for p in process:
                    p(data,i,j,k,dataijk,fullpath,alldata,timestamp)

    return data,filearray







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




def save_correct_r(data,i,j,k,dataijk,fullpath,alldata,timestamp):
    r = alldata[:,1:4]
    correctR(r)
    alldata[:,1:4] = r
    dataijk['r'] = r
    fCartPosCorrected = file(fullpath+'/'+'CartPosCorrected.dat','w')
    for dataline in alldata:
        fCartPosCorrected.write('\t'.join([str(a) for a in dataline])+'\n')
    fCartPosCorrected.close()


# return: data,filearray
def data_read_correct_r():
    return data_read(redump=False, process=[save_correct_r])
    
    
