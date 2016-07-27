from sympy.physics.vector import *
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
    b = set()
    for ni,i in enumerate(list(a)):
        if show:
            print "\r",ni,
        b.add( i.to_matrix(N).evalf() )
    if show:
        print "\r            \r",
    return b


def gen_base():
    a = [set()]
    b = []

    for i in [N.x, N.y, N.z]:
        a[0].add(i)
        a[0].add(-i)

    print "0 | a:",len(a[0]),
    b.append(remove_overlapping(a[0]));print "  b:",len(b[0]);

    a.append(extend_base(a[0]));print "1 | a:",len(a[1]),
    b.append(remove_overlapping(a[1]));print "  b:",len(b[1]);

    a.append(extend_base(a[1]));print "2 | a:",len(a[2]),
    b.append(remove_overlapping(a[2]));print "  b:",len(b[2]);

    a.append(extend_base(a[2],True));print "0 | a:",len(a[3]),
    b.append(remove_overlapping(a[3],True));print "  b:",len(b[3]);

    return {'a':a,'b':b}

if __name__ == "__main__":
    data = save_load("dcc_base.pydump",data=None,mode=None,datagen=gen_base)


#a = extend_base(a)
#print len(a)
