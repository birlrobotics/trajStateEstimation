#!/usr/bin/python
import fastdtw
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from load_data import data

x = [1,2,3,3,3,3,4,5,6]
y = [1,2,2,2,3,3,3,4,5,5,5,5,6]

def testfdtw(x,y,color):
    D,dist,path = fastdtw.fastdtw(x,y)
    mat = np.zeros((len(x),len(y),4))
    maxcost = max(filter(lambda x:x!=np.inf,map(lambda x:x[0],D.values())))
    print "\t",maxcost,(len(x)+len(y))/2
    #mat.fill(maxcost+1)
    mat.fill(np.inf)
    for i,j in D:
        mat[i-1,j-1] = D[i,j]
    smat = mat[:,:,0]
    #smat[np.isinf(smat)] = maxcost+1
    #smat = maxcost+1-smat
    #print smat



    #plt.imshow(smat,
    #           interpolation="none",
    #           cmap=plt.cm.gray,
    #           #norm=matplotlib.colors.LogNorm()
    #)

    plotp = np.array(path).T
    plt.plot(plotp[1],plotp[0],color=color)

def testondata(sa,sb,color):
    global data
    a = data
    print a.keys()[sa[0]],"(",
    a = a[a.keys()[sa[0]]]
    print a.keys()[sa[1]],")",
    a = a[a.keys()[sa[1]]]
    a = a[a.keys()[sa[2]]]
    b = data
    print b.keys()[sb[0]],"(",
    b = b[b.keys()[sb[0]]]
    print b.keys()[sb[1]],")",
    b = b[b.keys()[sb[1]]]
    b = b[b.keys()[sb[2]]]
    #print "FF code a\n",''.join(map(str,a['FF_code_91'])),"\n"
    #print "FF code b\n",''.join(map(str,b['FF_code_91'])),"\n"
    #print "AFF code\n",''.join(map(str,a['AFF_code_91'])),"\n"
    a_ = a['AFF_code_91']
    b_ = b['AFF_code_91']
    #a_ = filter(lambda x:x,a_)
    #b_ = filter(lambda x:x,b_)
    

    print "  ",len(a_),len(b_),a_==b_
    print '['+','.join(map(str,a_))+']','\n\n'
    testfdtw(a_,b_,color)

times = 1
for i in xrange(times):
    testondata([0,0,0],
               [0,0,i],
               (1,0,0,1))
for i in xrange(times):
    testondata([1,0,0],
               [0,0,i],
               (0,0,1,1))

plt.show()
