from sympy.physics.vector import *
import numpy
import datadumper
reload(datadumper)
from datadumper import save_load

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
        b.append(remove_overlapping(a[i+1],i>1));print "  b:",len(b[i+1]);

    #return {'a':a,'b':b}
    return b


if __name__ == "__main__":
    # cPickle can't save sympy data
    data = save_load("dcc_base.pydump",data=None,mode=None,datagen=gen_base,param={'iter':3})
    print map(lambda x:x.shape[0],data)
