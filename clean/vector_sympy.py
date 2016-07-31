from sympy.physics.vector import *
import numpy
import datadumper
reload(datadumper)
from datadumper import dumper

dcc_base_dump_file_name = "dcc_base.pydump"
N = ReferenceFrame('N')

def extend_base(a_,show=False):
    a = []
    for i in a_:
        a.append(i)
    lista = list(a)
    lla = len(lista)
    for i in xrange(lla):
        for j in xrange(i+1,lla):
            if show:
                print "\r",i,j,
            to_add = (lista[i]+lista[j]).normalize()
            if not show:
                to_add = to_add.simplify()
            a.append(to_add)
    if show:
        print "\r            \r",
    return set(a)

def remove_overlapping(a,show=False):
    b_ = set()
    for ni,i in enumerate(list(a)):
        if show:
            print "\r",ni,
        b_.add( i.to_matrix(N).evalf() )
    b = []
    for i in b_:
        b.append(numpy.array(i).astype(float))
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
    b.append(remove_overlapping(a[0]));print "  b:",len(b[0]);

    for i in xrange(iter):
        a.append(extend_base(a[i],i>1));print str(i+1)+" | a:",len(a[i+1]),
        b.append(remove_overlapping(a[i+1],i>1).tolist());b[i+1].sort();b[i+1]=numpy.array(b[i+1]);
        #print "  b:",len(b[i+1]);
        print "  b:",b[i+1].shape[0];

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
