#!/usr/bin/python
import os
from hiro_cartpot_clean import dumper, all_config_process, clfname
import numpy

all_config_func = lambda base:dumper.save_load(
    "all_config_result"+str(base)+".pydump.result",
    data=None,
    mode=None,
    datagen=all_config_process,
    param={"base":base},
    dataname="All config results",
)

def add_statistics_func(base):
    res = all_config_func(base)
    for rconf in res:
        for r in rconf:
            acc = r['accuracy']
            for ker in acc:
                accker = acc[ker]
                acclist = [accker[i] for i in xrange(2,21)]
                accker['avg'] = numpy.average(acclist)
                accker['min'] = numpy.min    (acclist)
                accker['max'] = numpy.max    (acclist)
    return res

add_statistics = lambda base:dumper.save_load(
    "all_config_result"+str(base)+".ext.pydump.result",
    data=None,
    mode=None,
    datagen=add_statistics_func,
    param={"base":base},
    dataname="All config results with average, min, max calculated",
)

add_statistics(1)
add_statistics(2)

os.system('mkdir -p ./csv/all')
os.system('mkdir -p ./csv/svc')
os.system('mkdir -p ./csv/lin_svc')

def dict_expand(keylist):
    if(not keylist):
        return None
    ret = dict.fromkeys(keylist[0])
    if(len(keylist)==1):
        for i in ret:
            ret[i] = None
    else:
        for i in ret:
            ret[i] = dict_expand(keylist[1:])
    return ret

def save_to_csv(base_i):
    str_base_i = str(base_i)
    res = add_statistics(base_i)
    print len(res), map(len,res)
    
    svc_total_f = open('./csv/svc_total.csv','w')
    lin_svc_total_f = open('./csv/linsvc_total.csv','w')
    
    datasets = [x['dataset'] for x in res[1]]    

    svc_total_f.write("base,"+str_base_i+'\n'+
                      'implementation,svc\n'+
                      ',,,,'+','.join(datasets)+'\n')

    lin_svc_total_f.write("base,"+str_base_i+'\n'+
                            'implementation,lin_svc\n'+
                            ',,,,'+','.join(datasets)+'\n')
    
    stat = ['avg','min','max']
    strt = [['FF','AFF'],
            ['task','behavior'],
            ['extend','cut','interp'],
            stat,
            datasets,
            clfname]

    dtree = dict_expand(strt)

    for r_config in res:
        encode_ = r_config[0]["encode"]
        level_ = r_config[0]["level"]
        align_ = r_config[0]["align"]
        dtp_ = dtree[encode_][level_][align_]
        l_r_config_6 = (len(r_config)==6)
        for k in r_config:
            ds_ = k['dataset']
            for i in stat:
                if l_r_config_6 and ds_=='A':
                    for clf in clfname:
                        dtp_[i][ds_][clf] = 'NULL'
                else:
                    for clf in clfname:
                        dtp_[i][ds_][clf] = k['accuracy'][clf][i]
             
    svc_total_str = ''
    for enc in ['FF','AFF']:
        for lv in ['task','behavior']:
            for aln in ['extend','cut','interp']:
                for st in stat:
                    svc_total_str += ','.join([enc,lv,aln,st]+[str(dtree[enc][lv][aln][st][ds]['svc']) for ds in datasets])+'\n'
    svc_total_f.write(svc_total_str)

    lin_svc_total_str = ''
    for enc in ['FF','AFF']:
        for lv in ['task','behavior']:
            for aln in ['extend','cut','interp']:
                for st in stat:
                    lin_svc_total_str += ','.join([enc,lv,aln,st]+[str(dtree[enc][lv][aln][st][ds]['lin_svc']) for ds in datasets])+'\n'
    lin_svc_total_f.write(lin_svc_total_str)

    svc_total_f.close()
    lin_svc_total_f.close()

    for r_config in res:
        assert len(set(map(lambda x:(x['encode'],x['level'],x['align']),r_config))) is 1

        filename_gen = '_'.join([str_base_i,
                                 r_config[0]["encode"],
                                 r_config[0]["level"],
                                 r_config[0]["align"]])

        all_f = open('./csv/all/'+filename_gen+".csv",'w')
        svc_f = open('./csv/svc/'+filename_gen+'_svc'+".csv",'w')
        lin_svc_f = open('./csv/lin_svc/'+filename_gen+'_linsvc'+".csv",'w')

        datasets = [x['dataset'] for x in r_config]

        all_f.write("base,"+str_base_i+'\n'+
                    ',,'+','.join(clfname)+'\n')

        svc_f.write("base,"+str_base_i+'\n'+
                    'implementation,svc\n'+
                    ','+','.join(datasets)+'\n')

        lin_svc_f.write("base,"+str_base_i+'\n'+
                        'implementation,lin_svc\n'+
                        ','+','.join(datasets)+'\n')        

        svc_f.write('\n'.join([','.join([str(i)]+
                                        [str(r_['accuracy']['svc'][i]) for r_ in r_config])
                               for i in list(xrange(2,21))+['min','max','avg']]))

        lin_svc_f.write('\n'.join([','.join([str(i)]+
                                            [str(r_['accuracy']['lin_svc'][i]) for r_ in r_config])
                                   for i in list(xrange(2,21))+['min','max','avg']]))
        
        all_f.write(''.join(['\n'.join([','.join([r_['dataset'],str(i)]+
                                                 [str(r_['accuracy'][j][i]) for j in clfname]) 
                                        for i in xrange(2,21)]+
                                       ["",""])
                             for r_ in r_config]))

        
            
        all_f.close()
        svc_f.close()
        lin_svc_f.close()


save_to_csv(1)
save_to_csv(2)
