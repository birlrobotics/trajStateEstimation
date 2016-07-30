import cPickle

class _dumper():
    def __init__(self):
        self._dataname_default = "Data"
        self._dataname = self._dataname_default
        
    def load_data(self,filename):
        ftemp = open(filename,"r")
        data = cPickle.load(ftemp)
        ftemp.close()
        print self._dataname+" loaded from "+filename
        return data

    def save_data(self,filename,data):
        ftemp = open(filename,"w")
        cPickle.dump(data,ftemp)
        ftemp.close()
        print self._dataname+" saved to "+filename
        return data


    def save_load(self,filename,data=None,mode=None,datagen=None,param={},dataname=""):
        '''
Save/load data. 
@param data 
@param filename (str)
@param mode (str): ["save","load"]
'''
        self._dataname = dataname if dataname else self._dataname_default
        if not mode:
            try:
                data = self.save_load(filename,data,"load",datagen,param,dataname)
            except Exception as e:
                print e,e.message
                data = self.save_load(filename,data,"save",datagen,param,dataname)
        elif mode=="save":
            if data:
                data = self.save_data(filename,data)
            elif datagen:
                data = self.save_data(filename,datagen(**param))
        elif mode=="load":
            data = self.load_data(filename)
        return data


dumper = _dumper()
