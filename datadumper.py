import cPickle

def load_data(filename):
    ftemp = open(filename,"r")
    data = cPickle.load(ftemp)
    ftemp.close()
    print "Data loaded from "+filename
    return data

def save_data(filename,data):
    ftemp = open(filename,"w")
    cPickle.dump(data,ftemp)
    ftemp.close()
    print "Data saved to "+filename
    return data


def save_load(filename,data=None,mode=None,datagen=None,param={}):
    '''
Save/load data. 
@param data 
@param filename (str)
@param mode (str): ["save","load"]
'''
    data = None
    if not mode:
        try:
            data = save_load(filename,data,"load",datagen,param)
        except Exception as e:
            print e,e.message
            data = save_load(filename,data,"save",datagen,param)
    elif mode=="save":
        if data:
            data = save_data(filename,data)
        elif datagen:
            data = save_data(filename,datagen(**param))
    elif mode=="load":
        data = load_data(filename)
    return data
