import uproot
import numpy as np
import pandas as pd
import awkward as ak
import os

folder_path_input = '/home/kali/sim/data'
path_input = "~/sim/lct_BIB.root"

lam_max = 75/360 * 2*np.pi


def multi_lkth(folder_path, max_events=1, step=1):
    """
    Loops over all LCTuple files and retrieves its data
    
    Parameters
    ----------
    folder_path : string
        path to the folder containing all the files.
    max_events : int, optional
        the max number of events to load through. The default is 10.
    step : int, optional
        The step size for looping over chunks of data. The default is 1.

    Returns
    -------
    multi_log : list
        A list of all the event data stored in dictionaries.
    """
    multi_log = []
    num = 0
    for file in os.listdir(folder_path):
        path = f'{folder_path}/{file}'
        #retrieves all keys in a root file and finds those labelled MYLCTuple.
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
    """
    Processes data that is needed from the file and compiles it into a dictionary for
    ease of use.
    
    Parameters
    ----------
    lctuple : list
        array of data to be processed.
        
    Returns
    -------
    dict
        {log: dict of mc particle arrays,
         ltr: dict of all track state arrays,
         lti: dict of all track information arrays,
         cov: array of the covariance matrix,
         rf: fraction of data that passes gen==1 cha!=0 selection
         }
    """
     
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
           'trsip':[],  #Index of track state (ts branches) at interaction point
           'trsfh':[],  #Index of track state (ts branches) at the first hit
           'trslh':[],  #Index of track state (ts branches) at the last hit
           'trsca':[],  #Index of track state (ts branches) at the calorimeter
           'tr2mcf':[], #Track index from tr branches.
           'tr2mct':[], #To matched MC particle in mc branches
           }
    
    #'tof':[],    #time of flight
    #MC particle parameters
    rf = 0
    
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
    
    #selection that particles must be stable and must not have neutral charge
    pre = (gen==1) & (cha!=0)
    
    rf = ak.count(pre[pre])
    
    for i in log:
        log[i] = log[i][pre]
    
    mx = log['mcmox']
    my = log['mcmoy']
    mz = log['mcmoz']
    
    #updates the dictionary with additional arrays of important variables

    log.update({'mcppt':ak.Array(np.sqrt(mx**2 + my**2 + mz**2))})
    log.update({'mcl':np.arctan2(mz,log['mcppt'])})
    log.update({'mcphi':np.arctan2(my,mx)})
    
    ltr.update({'tsppt':(.3 * 3.57)/(omega)})
    
    return {'log':log,'ltr':ltr,'lti':lti,'cov':covariance,'rf':rf}
