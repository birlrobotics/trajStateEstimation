import os
import numpy
#import vector_sympy
#reload(vector_sympy)
#from vector_sympy import dumper, dcc_base
import basevectornew_numpy
reload(basevectornew_numpy)
from basevectornew_numpy import dumper, dcc_base, normalize


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
            print j
            path = root+'/'+i+'/'
            # find all CartPos*.dat files
            CartFileList = os.popen(
                'cd '+path+
                ' && find . -maxdepth 3 -name "R_CartPos*" |grep "\./'+j+
                '.*/R_CartPos*"').read().strip('./').strip().split('\n./') 
            #        CartFileList = os.popen('find '+path+' -maxdepth 3 -name "CartPos*" |grep "'+i+'/'+j+'.*/CartPos*"').read() # find all CartPos*.dat files

            print path
            CartFileList.sort(reverse=True)
            #print CartFileList
            PathList = dict(map(lambda x:(x[:x.find("R_CartPos")],x[x.find("R_CartPos"):]),CartFileList))
            CartFileList = PathList.values()
            PathList = PathList.keys()
            len_filelist = len(PathList)
            StateFileList = [None]*len_filelist
            for li in xrange(len_filelist):
                StateFileList[li] = os.popen('cd "'+path+PathList[li]+'" && find . -name "R_State*.dat"|grep "\./R_State"').read().strip('./').strip().split()[0]
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


def basegen19One(t, n, b):
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


def basegenOneRot(t, n, b, it=1): # default for DCC19
    assert len(t.shape) is 1
    assert len(n.shape) is 1
    assert len(b.shape) is 1
    rot = numpy.array([t,n,b]).T
    return numpy.dot(rot,dcc_base[it].T).T

def basegenRot(t, n, b, it=1): # Only for quick-generation of FF
    assert len(t.shape) is 2
    assert len(n.shape) is 2
    assert len(b.shape) is 2
    base_ = numpy.transpose(numpy.array([t,n,b]),axes=(1,2,0))
    base = numpy.transpose(numpy.dot(base_, dcc_base[it].T),axes=(0,2,1))

    naxis = len(t.shape)
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


# T == 0 : keep base
# T != 0 :
#     N == 0 : keep N
#     N != 0 : nice
# B = T x N
def basegen19(t, n, b):
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

# cover all zeros with nearby values
# | 0 |=====| 0 |=====| 0 |=====| 0 |
# |<--|=====|-->|=====|-->|=====|-->|
def cover_zeros(a_):
    assert len(a_.shape) is 2
    assert a_.shape[1] is 3

    a = a_.copy()

    norm = lambda x:numpy.linalg.norm(x,axis=1)
    z = (norm(a)==0)
    c = a.shape[0]
    i = 0
    while i<c:
        if not z[i]:
            for j in xrange(i-1,-1,-1):
                a[j] = a[j+1]
            break
        i += 1

    while i<c:
        if z[i]:
            a[i] = a[i-1]
        i += 1

    return a


# Discrete fenet frame B (binormal vector)
def getb(t): # t is the tangent vector
    assert len(t.shape) is 2
    assert t.shape[1] is 3
    
    c = t.shape[0]
    b = numpy.zeros(t.shape)
    b[1:] = numpy.cross(t[:c-1],t[1:]) #
    b[0] = b[1]
    
    normalize(b)
    
    #b[norm(t)==0] = numpy.zeros((3,))
    return b





# [ WARNING ] THIS IS Frenet Frame encoding, not Accumulated Frenet Frame
# code.shape == r.shape[0] - 1
def encode(r,t,base):
    assert t.shape[0] == base.shape[0]
    t_ = numpy.transpose(numpy.tile(t,(base.shape[1],1,1)),axes=(1,0,2))
    c = base.shape[0]
    nbase = base.shape[1]
    dbase = numpy.zeros((c-1,nbase))
    dbase = numpy.sum(t_[1:]*base[:c-1],axis=2)
    # dbase0: absolute
    # base[0][1] is T vector
    #dbase[0] = numpy.zeros((19,))
    return dbase.argmax(axis=1)


def FFencode(r,it=1):
    orit = gett(r)
    t = cover_zeros(orit)
    b = getb(t)
    b = cover_zeros(b)
    n = numpy.cross(b,t)
    ## DCC19 base directions
    #base = basegen19(t,n,b)
    base = basegenRot(t,n,b,it)
    ## DFF uses difference as tangent vector directly.
    code = encode(r,t,base)
    #return dict(zip("t,n,b,base,code".split(','),(t,n,b,base,code)))
    return code


def AFFencode(r,it=1):
    norm = numpy.linalg.norm
    code = numpy.zeros((r.shape[0]-1,))
    orit = gett(r)
    t = cover_zeros(orit)
    b = getb(t)
    b = cover_zeros(b)
    n = numpy.cross(b,t)
    base_ff = basegenRot(t,n,b,it)

    #b0 = numpy.cross(r[0],r[1])
    #normalize(b0.reshape(1,3))
    #n0 = numpy.cross(b0,t[0])
    #normalize(n0.reshape(1,3))
    #base0 = basegen19One(t[0],n0,b0)
    ##base0 = basegenOneRot(t[0],n0,b0,it)

    #base0 = basegen19One(t[0],n0,b0)
    #base0 = basegenOneRot(t[0],n[0],b[0],it) #should not be zero
    base0 = base_ff[0]
    assert norm(t[0])

    #pi_d_2 = numpy.pi / 2.0
    
    npoint = r.shape[0]
    bound = numpy.pi/numpy.power(2,it+1)
    for i in xrange(1,npoint):
        n_i_1 = norm(base0[1])
        n_i = norm(t[i])
        if (not (n_i_1 and n_i)):
        #if False:
            angle = 0
            print n_i_1, n_i,"="
        else:
            cos_ = numpy.dot(base0[1],t[i])/(n_i_1*n_i)
            if(numpy.abs(cos_)>=1):
                cos_ = 1 if cos_>0 else -1
            angle = numpy.arccos(cos_)
            assert angle >= 0
            #print angle,bound,angle>bound

        if(angle > bound):
            #b[i] = numpy.cross(r[i-1],r[i])
            #normalize(b[i].reshape(1,3))
            #n[i] = numpy.cross(b[i],t[i])
            #normalize(n[i].reshape(1,3))
            #base[i] = basegen19One(t[i],n[i],b[i])

            #b1 = numpy.cross(t[i-1],t[i])
            #normalize(b1.reshape(1,3))
            #n1 = numpy.cross(b1,t[i])
            #normalize(n1.reshape(1,3))
            #base1 = basegenOneRot(t[i],n1,b1)
            base1 = base_ff[i]
            code[i-1] = numpy.argmax(numpy.dot(base0,t[i]))
            #print i,
        else:
            base1 = base0
            code[i-1] = 1
            #print "="*10

        base1,base0 = base0,base1
    print ""
    return code.astype(int)



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
def gencodedata(data, process=[]):
    process_keys = set(['func','param','final-info'])
    assert isinstance(process,list)
    assert all(map(lambda x:isinstance(x,dict),process))
    assert all(map(lambda x:process_keys.issuperset(set(x.keys())),process))
    assert all(map(lambda x:callable(x['func']),process))
    assert all(map(lambda x:isinstance(x['param'],dict) if x.has_key('param') else True,process))
    assert all(map(lambda x:isinstance(x['final-info'],str) if x.has_key('final-info') else True,process))

    #falldata = open(alldatafile,"w")
    for i in data:
        for j in data[i]:
            for k in data[i][j]:
                dataijk = data[i][j][k]
                fullpath = '/'.join([root,i,k])

                timestamp = map(float,open(fullpath+dataijk['State'],"r").read().strip().split())
                alldata =reader(fullpath+dataijk['CartPos'])

                r = alldata[:,1:4]

                correctR(r)

                timestamp_i = map(lambda x:alldata[:,0].searchsorted(x),timestamp)
                dataijk['state_stamp'] = timestamp_i
                dataijk['r'] = r
                print i,j,k
                dataijk['length'] = r.shape[0]


                splitr = []
                timestamp_i.append(r.shape[0])
                for tsi in xrange(1,len(timestamp_i)):
                    splitr.append(r[timestamp_i[tsi-1]:timestamp_i[tsi]])

                dataijk['r_split'] = splitr
                    
                datapack = [data,i,j,k,dataijk,fullpath,alldata,timestamp]
                for p in process:
                    pparam_ = p['param'] if p.has_key('param') else {}
                    p['func'](datapack, **pparam_)
    #cPickle.dump(data,falldata)
    #falldata.close()
    for p in process:
        if p.has_key('final-info'):
            print p['final-info']

    #print "all data saved to ./"+alldatafile+". data will be loaded from this file."
    return data


filename_cartpose_corrected = 'R_CartPos_Corrected.dat'
def poscorrectedsave(datapack):
    data,i,j,k,dataijk,fullpath,alldata,timestamp = datapack
    global filename_cartpose_corrected
    r = dataijk['r']
    alldata[:,1:4] = r
    fCartPosCorrected = file(fullpath+filename_cartpose_corrected,'w')
    for dataline in alldata:
        fCartPosCorrected.write('\t'.join([str(a) for a in dataline])+'\n')
    fCartPosCorrected.close()
    

def codesave(datapack,codedir='./codedir',codetype=[]):
    data,i,j,k,dataijk,fullpath,alldata,timestamp = datapack
    for code_name in codetype:
        try:
            proccode = dataijk[code_name]
            tcode = ','.join([str(pc) for pc in proccode])
        except Exception as e:
            print "Exception in codesave(): ",type(e), e.message
            continue
        codedir_fullpath = '/'.join([codedir,i,j,''])
        codefilepath = codedir_fullpath+k.replace('/','__')+code_name+'.code'
        try:
            f = open(codefilepath,'w')
        except:
            os.system('mkdir -p '+codedir_fullpath)
            f = open(codefilepath,'w')
        f.write(tcode)
        f.close()


def save_code_FF(datapack,it):
    data,i,j,k,dataijk,fullpath,alldata,timestamp = datapack
    r = dataijk['r']
    proccode = FFencode(r,it)
    DCC_BASE_NUM = dcc_base[it].shape[0]    
    dataijk['FF_code_'+str(DCC_BASE_NUM)] = proccode
    splitc = []
    timestamp_i = dataijk['state_stamp']
    for tsi in xrange(1,len(timestamp_i)):
        splitc.append(proccode[timestamp_i[tsi-1]:timestamp_i[tsi]])
    dataijk['FF_code_split_'+str(DCC_BASE_NUM)] = splitc

def save_code_AFF(datapack,it):
    data,i,j,k,dataijk,fullpath,alldata,timestamp = datapack
    r = dataijk['r']
    proccode = AFFencode(r,it)
    DCC_BASE_NUM = dcc_base[it].shape[0]    
    dataijk['AFF_code_'+str(DCC_BASE_NUM)] = proccode
    splitc = []
    timestamp_i = dataijk['state_stamp']
    for tsi in xrange(1,len(timestamp_i)):
        splitc.append(proccode[timestamp_i[tsi-1]:timestamp_i[tsi]])
    dataijk['AFF_code_split_'+str(DCC_BASE_NUM)] = splitc





root = '~/temp/data'
root = get_abs_path(root)

filenames_save = "filenames.pydump"
codedir = 'trajcode'
alldatafile = "alldata.pydump"

tasktype_tree = {
    'data_003_SIM_HIRO_SA_Success': ['2016',"Trial","Test"],
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
    param={
        'data':filenames, 
        'process':[
            {'func': poscorrectedsave },
            {'func': save_code_AFF,
             'param' : {'it':0}},
            {'func': save_code_FF,
             'param' : {'it':0}},
            {'func': save_code_AFF,
             'param' : {'it':1}},
            {'func': save_code_FF,
             'param' : {'it':1}},
            {'func': save_code_AFF,
             'param' : {'it':2}},
            {'func': save_code_FF,
             'param' : {'it':2}},
            {'func': codesave, 
             'param':{
                 'codedir':codedir,
                 'codetype':[
                     'FF_code_7',
                     'FF_code_19',
                     'FF_code_91',
                     'AFF_code_7',
                     'AFF_code_19',
                     'AFF_code_91',
                 ]}, 
             'final-info':"code saved to ./"+codedir+". This is for intuitive analysis." },
        ]
    },
    dataname="Trajectory data",
)


def sampleviewone():
    global data
    a = data
    a = a[a.keys()[1]]
    a = a[a.keys()[0]]
    a = a[a.keys()[15]]
    print "FF code\n",''.join(map(str,a['FF_code_91'])),"\n"
    print "AFF code\n",''.join(map(str,a['AFF_code_91'])),"\n"
    return a

#sample91a = sampleviewone()
