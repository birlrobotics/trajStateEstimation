import sympy
from sympy.physics.vector import *
import numpy
import datadumper
reload(datadumper)
from datadumper import dumper

dcc_base_dump_file_name = "dcc_base.pydump"
N = ReferenceFrame('N')

def extend_base(a_,it=1):
    a = []
    for i in a_:
        a.append(i)
    lista = list(a)
    lla = len(lista)
    for i in xrange(lla):
        for j in xrange(i+1,lla):
            if it>1:
                print "\r",i,j,
            to_add = lista[i]+lista[j]
            #print "[[",to_add.to_matrix(N).norm(),"]]"
            #assert not any(map(lambda x:x is sympy.nan,lista[i].to_matrix(N).evalf())),repr(lista[i])+"\n"+repr(lista[i].to_matrix(N))
            #assert not any(map(lambda x:x is sympy.nan,lista[j].to_matrix(N).evalf())),repr(lista[j])+"\n"+repr(lista[j].to_matrix(N))
            if True:
                assert not any(map(lambda x:x is sympy.nan,to_add.to_matrix(N).evalf())),repr(to_add)+"\n"+repr(to_add.to_matrix(N))
                if to_add.to_matrix(N).norm() != 0:
                    to_add = to_add.normalize()
                assert not any(map(lambda x:x is sympy.nan,to_add.to_matrix(N).evalf())),repr(to_add)+"\n"+repr(to_add.to_matrix(N))
            if it<=1 or True:
                to_add = to_add.simplify()
            a.append(to_add)
    if it>1:
        print "\r            \r",
    return set(a)

def remove_overlapping(a,show=False):
    b_ = set()
    for ni,i in enumerate(list(a)):
        if show:
            print "\r",ni,
        newone = i.to_matrix(N).evalf()
        #assert not any(map(lambda x:x is sympy.nan,newone)),repr(newone)+"\n"+repr(i)+"\n"+repr(i.to_matrix(N))
        b_.add( newone )
    b = []
    for i in b_:
        newone2 = numpy.array(i).astype(float)
        #assert not numpy.isnan(newone2).any()
        b.append(newone2)
    if show:
        print "\r            \r",
    b = numpy.array(b)
    return b.reshape(b.shape[:2])


def base_sort_cmp(x,y):
    for i in xrange(3):
        if x[i]>y[i]:
            return -1
        elif x[i]<y[i]:
            return 1
    return 0

def gen_base(iter=1):
    a = [set()]
    b = []

    for i in [N.x, N.y, N.z]:
        a[0].add(i)
        a[0].add(-i)
    a[0].add((i-i).normalize())
    print N.x*0

    print "0 | a:",len(a[0]),
    b.append(remove_overlapping(a[0]).tolist())
    b[0].sort(base_sort_cmp)
    b[0] = numpy.array(b[0])
    print "  b:",b[0].shape[0]

    for i in xrange(iter):
        a.append(extend_base(a[i],i+1));print str(i+1)+" | a:",len(a[i+1]),
        b.append(remove_overlapping(a[i+1],i>1).tolist());
        b[i+1].sort(base_sort_cmp)
        b[i+1]=numpy.array(b[i+1])
        #print "  b:",len(b[i+1])
        print "  b:",b[i+1].shape[0]

    #return {'a':a,'b':b}

    return b



def get_dcc_basis():
    global dcc_base_dump_file_name
    return dumper.save_load( 
        dcc_base_dump_file_name, 
        data=None,
        mode=None,
        datagen=gen_base,
        param={'iter':3},
        dataname="DCC basis",
    )

dcc_base = get_dcc_basis()

def test_dcc_basis():
    global dcc_base
    dcc_base_tuple = map(lambda x:map(tuple,x),dcc_base)
    fail_count = 0
    for i in xrange(1,len(dcc_base)):
        prev_ = dcc_base_tuple[i-1]
        next_ = set(dcc_base_tuple[i])
        for j in prev_:
            try:
                assert(j in next_)
            except:
                print "FAIL: ",i-1, i, "   ", j
                fail_count += 1
    if fail_count:
        print "TEST FAILED: ",fail_count,"failed test cases"
    else: 
        print "TEST PASSED"

if __name__ == "__main__":
    # cPickle can't save sympy data
    test_dcc_basis()
    print "DCC base size:", map(lambda x:x.shape[0],dcc_base)
