import os
import numpy
import vector_sympy
reload(vector_sympy)
from vector_sympy import save_load, dumper, dcc_base



# get the absolute path
def get_abs_path(p):
    return os.popen('echo '+root).read().strip()



def findfilenames(root):
    '''
@param root [string]: directory containing all datasets

Get all filenames of CartPos*.dat & State*.dat (most are the same, some outliers exist)
And return the dict-tree structure to store data.
'''
    filename = tasktype_tree.fromkeys(tasktype_tree.keys())
    for i in filename:
        filename[i] = {}
        for j in tasktype_tree[i]:
            path = root+'/'+i+'/'
            # find all CartPos*.dat files
            CartFileList = os.popen(
                'cd '+path+
                ' && find . -maxdepth 3 -name "CartPos*" |grep "\./'+j+
                '.*/CartPos*"').read().strip('./').strip().split('\n./') 
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



#def savecodefile(data):
def gencodedata(data, codedir=''):
    #falldata = open(alldatafile,"w")
    if(codedir):
        os.system('mkdir '+codedir)
    for i in data:
        if codedir:
            os.system('mkdir '+codedir+'/'+i)
        for j in data[i]:
            if codedir:
                os.system('mkdir '+codedir+'/'+i+'/'+j)
            for k in data[i][j]:
                if codedir:
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

                if codedir:
                    f.write(tcode)
                    f.close()
    #cPickle.dump(data,falldata)
    #falldata.close()

    if codedir:
        print "code saved to ./"+codedir+". This is for intuitive analysis."

    #print "all data saved to ./"+alldatafile+". data will be loaded from this file."
    return data






root = '~/temp/data'
root = get_abs_path(root)

filenames_save = "filenames.pydump"
codedir = 'trajcode'
alldatafile = "alldata.pydump"

tasktype_tree = {
    'data_003_SIM_HIRO_SA_Success': ['2012'],
    'data_004_SIM_HIRO_SA_ErrorCharac_Prob': ['FC','exp'],
    'data_008_HIRO_SideApproach_SUCCESS':['2012','x']
}

filenames = dumper.save_load(
    filenames_save,
    data=None,
    mode=None,
    datagen=findfilenames,
    param={'root':root},
    dataname="Filenames",
)


data = dumper.save_load(
    alldatafile,
    data=None,
    mode=None,
    datagen=gencodedata,
    param={'data':filenames, 'codedir': codedir},
    dataname="Trajectory data",
)
