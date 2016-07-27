import dill

def load_data(filename):
    ftemp = open(filename,"r")
    data = dill.load(ftemp)
    ftemp.close()
    print "Data loaded from "+filename
    return data

def save_data(filename,data):
    ftemp = open(filename,"w")
    dill.dump(data,ftemp)
    ftemp.close()
    print "Data saved to "+filename
    return data


def save_load(filename,data=None,mode=None,datagen=None):
    '''
Save/load data. 
@param data 
@param filename (str)
@param mode (str): ["save","load"]
'''
    data = None
    if not mode:
        try:
            save_load(filename,data,"load",datagen)
        except Exception as e:
            print e,e.message
            save_load(filename,data,"save",datagen)
    elif mode=="save":
        if data:
            save_data(filename,data)
        elif datagen:
            data = save_data(filename,datagen())
    elif mode=="load":
        data = load_data(filename)
    return data
