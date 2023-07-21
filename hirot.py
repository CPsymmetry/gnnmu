import uproot
import numpy as np
import pandas as pd
import awkward as ak
import os

folder_path_input = '/home/kali/sim/data'
path_input = "~/sim/lct_BIB.root"

lam_max = 75/360 * 2*np.pi


def multi_lkth(folder_path, max_events=10, step=1):
    multi_log = []
    num = 0
    for file in os.listdir(folder_path):
        path = f'{folder_path}/{file}'
        trees = [k for k in uproot.open(path).keys() if k.startswith('MyLCTuple')]
        for ttree in trees:
            for lctuple in uproot.iterate({f'{path}':ttree}, step_size=step):
                if num<max_events:
                    num+=1
                    multi_log.append(lkth(lctuple))
                else:
                    return multi_log
            
    return multi_log

def lkth(lctuple):
      
    log = {'mcmox':[],  #px 
           'mcmoy':[],  #py
           'mcmoz':[],  #pz
           'mccha':[],  #charge
           'mcgst':[],  #generation
           'mcpdg':[],  #pdg
           }
    
    ltr = {'tstnl':[],  #tan(lambda)
           'tsome':[],  #omega
           'tsphi':[],  #phi
           'tsdze':[],  #d0
           'tszze':[],  #z0
           }
    
    lti = {'trch2':[],  #chi2
           'trndf':[],  #degrees of freedom
           'trsip':[],
           'trsfh':[],
           'trslh':[], 
           'trsca':[],
           'tr2mcf':[],
           }
    
    #'tof':[],    #time of flight
    #MC particle parameters
    rf = 0
    tn = 0
    
    for i in log:
        params = lctuple[i]
        log[i] = params
    
    #parameters['p'] = pp
    
    #Track Parameters
    
    for i in ltr:
        params = lctuple[i]
        ltr[i] = params
        
    omega = ltr['tsome']
    
    #Track Information Parameters
    for i in lti:       
        params = lctuple[i]
        lti[i] = params
        
    trsip = lti['trsip']
    
    covariance = lctuple['tscov'] #covariance matrix  
    
    #calculations  
    gen = log['mcgst']
    cha = log['mccha']
    
    pre = (gen==1) & (cha!=0)
    
    for i in log:
        log[i] = log[i][pre]
    
    mx = log['mcmox']
    my = log['mcmoy']
    mz = log['mcmoz']

    log.update({'mcppt':ak.Array(np.sqrt(mx**2 + my**2 + mz**2))})
    log.update({'mclambda':np.arctan2(mz,log['mcppt'])})
    log.update({'mcphi':np.arctan2(my,mx)})
    
    ltr.update({'trppt':(.3 * 3.57)/(omega[trsip])})
    
    return {'log':log,'ltr':ltr,'lti':lti,'cov':covariance,'rf':rf}
