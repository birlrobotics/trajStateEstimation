#   hiro_cartpos_process_all_traj.py (show trajs)
# - hiro_cartpos_code_visualize.py (colormaps)
# - hiro_cartpos_git_updater.py (frames on path)


#hiro_cartpos_code_visualize.py
def codeshow(sa,sb):
    s = code[sa][sb]
    ss = []
    minlen = numpy.inf
    for i in s:
        ss.append(deepcopy(s[i]))
        if len(s[i])<minlen:
            minlen = len(s[i])
        #plt.plot(s[i],'o-')
    
    for i in xrange(len(ss)):
        ss[i] = ss[i][:minlen]

    nss = numpy.array(ss)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    cax = ax.imshow(nss, interpolation='none', extent=[0,minlen,0,len(ss)], aspect='auto', cmap='afmhot')
    ax.set_title(sa+' ['+sb+']')
    fig.savefig('/home/horisun/'+sa+'['+sb+'].png')



#hiro_cartpos_git_updater.py
def drawframes(r,frames,ax):
    assert r.shape[0] == frames.shape[0]
    #totallen = numpy.linalg.norm(numpy.diff(r,axis=0),axis=1).min()
    seglen = numpy.linalg.norm(numpy.diff(r,axis=0),axis=1)
    #seglenm = seglen.argmax()
    #seglen[seglenm] = seglen[seglenm-1]
    #minlen = seglen.min()
    totallen = seglen.sum()
    c = r.shape[0]
    numbers = 50
    assert c > numbers
    step = c/numbers
    #ax2 = fig.add_subplot(1,2,2)
    #ax2.set_ylim((0,0.0003))
    #ax2.scatter(numpy.arange(c-1),seglen)
    r_ = numpy.transpose(numpy.tile(r,(3,1,1)),axes=(1,0,2))
    frames_ = r_ + frames*(totallen/(c-1)*(step/2))#/3)#1000)
    
    tnb = numpy.transpose(
        numpy.concatenate([[r_],[frames_]],axis=0),
        axes = (2,1,3,0))

    t = tnb[0]
    n = tnb[1]
    b = tnb[2]

    color = 'rgb'

    for i in xrange(0,c,step):
        for j in xrange(3):
            ax.plot(
                tnb[j][i][0],
                tnb[j][i][1],
                tnb[j][i][2],
                color=color[j])
