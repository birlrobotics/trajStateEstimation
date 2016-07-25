from basevector import *

v = numpy.array([1,0,0])
u = numpy.array([0,1,0])
v = v/numpy.linalg.norm(v)
u = u/numpy.linalg.norm(u)
base = [set()]
base[0].add((0,0,0))

for i in map(tuple,[v, u,numpy.cross(v,u),numpy.cross(u,v),-v,-u]):
    base[0].add(i)

base.append(baseEnhance(base[0]))
base.append(baseEnhance(base[1]))
base.append(baseEnhance(base[2]))

g = numpy.array(list(base[3]))
k = [tuple(g[0])]

for i in g:
    flag = False
    for j in k:
        if not flag:
            if numpy.linalg.norm(i-numpy.array(j)) > 0.01:
                k.append(tuple(i))
                flag = True


k = numpy.array(k)

l = numpy.zeros((6607,6607))
for i in xrange(6607):
    for j in xrange(6607):
        l[i,j] = numpy.linalg.norm(g[i]-g[j])

l[l>0.01] = 1000
print (numpy.count_nonzero((l<0.01)) - 6607)/2,"overlaps"

from matplotlib import pyplot as plt
plt.matshow(l,cmap='gray')

print k.shape
