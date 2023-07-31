import uproot
import numpy as np
import awkward as ak
import os

folder_path_input = '/home/kali/sim/data'
path_input = "~/sim/lct_BIB.root"

def multi_lkth(folder_path, max_events=1, step=1, track_perf=False):
    """
    Loops over all LCTuple files and retrieves its data
    
    Parameters
    ----------
    folder_path : string
        path to the folder containing all the files.
    max_events : int, optional
        the max number of events to load through. The default is 1.
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
        print(file)
        path = f'{folder_path}/{file}'
        trees = []
        
        if not track_perf:
            trees = [k for k in uproot.open(path).keys() if k.startswith('MyLCTuple')]
        else:
            super_trees = uproot.open(path).keys()
            trees = [k for k in super_trees if 'MyOutput' in k]
            
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
           'mc2trf':[], #Track index from tr branches.
           'mc2trt':[], #To matched MC particle in mc branches
           }
    
    #'tof':[],    #time of flight
    #MC particle parameters
    rf = 0
    
    for i in log:
        log[i] = lctuple[i]
    
    #parameters['p'] = pp
    
    #Track Parameters
    
    for i in ltr:
        ltr[i] = lctuple[i]
    
    #Track Information Parameters
    for i in lti:       
        lti[i] = lctuple[i]

    mx = log['mcmox']
    my = log['mcmoy']
    mz = log['mcmoz']
    
    #updates the dictionary with additional arrays of important variables

    log.update({'mcppt':ak.Array(np.sqrt(mx**2 + my**2 + mz**2))})
    log.update({'mclam':np.arctan2(mz,log['mcppt'])})
    log.update({'mcphi':np.arctan2(my,mx)})
    
    omega = ltr['tsome']
    ltr.update({'tsppt':(.3 * 3.57)/(omega)})
    
    for i in range(15):
        ltr.update({f'cov{i}':lctuple['tscov'][:,:,i]})
        
    return {'log':log,'ltr':ltr,'lti':lti,'rf':rf}
