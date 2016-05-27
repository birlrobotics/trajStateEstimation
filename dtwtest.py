import wave
import struct
import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib.image as mpimg
import hashlib
import os.path


root = '/home/horisun/temp/apriltest/audio/'
recorded_words = [
    ('introduction-to-algorithms',19),
    ('cantonese-did-you-have-supper',20),
]

def get_wav_filename(a,b):
    type_ = recorded_words[a]
    b = b % type_[1]
    return root+type_[0]+'/wav/'+type_[0]+'_'+str(b)+'.wav'
    
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y))


wf = get_wav_filename(0,1)
w = wave.open(wf,'r')
frm = w.readframes(w.getnframes())

f = numpy.array(map(
    lambda x:struct.unpack('=h',x)[0], 
    [frm[i:i+2] for i in xrange(0,len(frm),2)]
))

s = []
datapick = [
    (40000,1000,1000),
    (40000,1000,500),
    (40500,1000,500),
    (40000,1080,14),
]
select = 0
x0,xl = datapick[select][0],datapick[select][2]
y0,yl = x0+datapick[select][1],xl

x1 = x0+xl
y1 = y0+yl
s.append(f[x0:x1])
s.append(f[y0:y1])

#plt.plot(s[0])
#plt.plot(s[1])

x = s[0]
y = s[1]
ratio = numpy.max(numpy.abs([x.min(),x.max(),y.min(),y.max()]))
x = x/float(ratio)
y = y/float(ratio)


if 0:
    fig, ax = plt.subplots(2,2)
    m = numpy.abs(numpy.tile(x,(yl,1))-numpy.tile(y,(xl,1)).T)

    ax[0,0].imshow(m, cmap=plt.cm.gray, interpolation='nearest')

    ax[1,0].plot(x)
    ax[1,0].plot(y)

    x[1:] = x[1:] - x[:xl-1]
    y[1:] = y[1:] - y[:yl-1]
    x[0] = x[1]
    y[0] = y[1]

    m = numpy.abs(numpy.tile(x,(yl,1))-numpy.tile(y,(xl,1)).T)

    ax[0,1].pcolormesh(X,Y,m, cmap=plt.cm.gray, 
                   interpolation='nearest')

    ax[1,1].plot(x)
    ax[1,1].plot(y)

else:
    m = numpy.abs(numpy.tile(x,(yl,1))-numpy.tile(y,(xl,1)).T)
    x.flags.writeable = False
    y.flags.writeable = False
    amfile = root+hashlib.md5(x).hexdigest()+hashlib.md5(x).hexdigest()+'.npy'
    x.flags.writeable = True
    y.flags.writeable = True
    if os.path.isfile(amfile):
        am = numpy.load(amfile)
    else:
        am = m.copy()
        for i in xrange(1,xl):
            am[0,i] = am[0,i] + am[0,i-1]
        for i in xrange(1,yl):
            am[i,0] = am[i,0] + am[i-1,0]
        for i in xrange(1,yl):
            for j in xrange(1,xl):
                am[i,j] = am[i,j] + numpy.min([am[i-1,j], am[i-1,j-1], am[i,j-1]])
        numpy.save(amfile,am)
    p = []
    pc = numpy.array([yl-1,xl-1])
    p.append(pc.tolist())
    # x : 0
    # y : 1
    # xy : 2
    while(pc.any()):
        xis0 = pc[1] == 0
        yis0 = pc[0] == 0
        if(xis0 and yis0):
            break
        if(xis0):
            pc[0] -= 1
        elif(yis0):
            pc[1] -= 1
        else:
            i = pc[1]
            j = pc[0]
            dirmin = numpy.argmin([am[i-1,j], am[i,j-1], am[i-1,j-1]])
            print dirmin,am[i-1,j], am[i,j-1], am[i-1,j-1],
            if(dirmin == 0):
                pc[1] -= 1
            elif(dirmin == 1):
                pc[0] -= 1
            else:
                pc[1] -= 1
                pc[0] -= 1
        print pc
        p.append(pc.tolist())

    p = numpy.array(p).T

    fig, axScatter = plt.subplots(1,2,figsize=(10,10))

    im = axScatter[0].imshow(
        m, cmap=plt.cm.gray, interpolation='nearest',
        #norm=MidpointNormalize(midpoint=(m.max()-m.min())*0.2+m.min(), 
        #norm=colors.LogNorm(
        #norm=colors.SymLogNorm(linthresh=0.3, linscale=0.03,
        #norm=colors.PowerNorm(gamma=1./.10000,
#                              vmin=m.min(), vmax=m.max()),
    )

    im1 = axScatter[1].imshow(
    #im1 = axScatter[1].matshow(
        am[::,::], cmap=plt.cm.gray, interpolation='nearest',
        norm=MidpointNormalize(midpoint=(am.max()-am.min())*0.3+am.min(), 
        #norm=colors.LogNorm(
        #norm=colors.SymLogNorm(linthresh=0.3, linscale=0.03,
        #norm=colors.PowerNorm(gamma=1./.10000,
                              vmin=am.min(), vmax=am.max()),
    )
    fig.canvas.draw()
    axScatter[1].autoscale(False)
    #axScatter[1].invert_yaxis()
    axScatter[1].plot(*p,color='r')
    


    fig.colorbar(im, ax=axScatter[0], extend='both')
    axScatter[0].set_aspect(1.)
    fig.colorbar(im, ax=axScatter[1])
    axScatter[1].set_aspect(1.)

    divider = make_axes_locatable(axScatter[0])

    axHistx = divider.append_axes("top", 1, pad=0.0, sharex=axScatter[0])
    axHisty = divider.append_axes("left", 1, pad=0.0, sharey=axScatter[0])

    #plt.setp(
    #    axHistx.get_xticklabels() + axHisty.get_yticklabels(),
    #    visible=False
    #)
    
    axHisty.set_xticks(())
    axHisty.set_yticks(())
    axHistx.set_xticks(())
    axHistx.set_yticks(())
    axHisty.plot(-y,numpy.arange(0,yl,1))
    axHisty.fill_betweenx(numpy.arange(0,yl,1),-y,-numpy.min(y),facecolor='k')
    axHistx.plot(x)
    axHistx.fill_between(numpy.arange(0,xl,1),x,numpy.min(x),facecolor='k')

    #ax[1].plot(x)
    #ax[1].plot(y)
    plt.savefig('dtw-original_'+'-'.join([str(i) for i in [x0,x1,y0,y1]])+'.png')
plt.show()
