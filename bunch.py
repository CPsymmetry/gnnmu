import awkward as ak
import numpy as np
import uproot
import matplotlib.pyplot as plt
from hirot import multi_lkth

"""
    log = {'mcmox':[],  #px 
           'mcmoy':[],  #py
           'mcmoz':[],  #pz
           'mccha':[],  #charge
           'mcgst':[],  #generation
           'mcpdg':[],  #pdg
           'mcppt':[],   #momentum
           'mcphi':[],  #phi
           'mcl':[]     #lambda
           }
    
    ltr = {'tstnl':[],  #tan(lambda)
           'tsome':[],  #omega
           'tsphi':[],  #phi
           'tsdze':[],  #d0
           'tszze':[],  #z0
           'tsppt':[],  #track state momentum
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
"""

class bunch:
    def __init__(self, sys):
        self.log = sys['log']
        self.ltr = sys['ltr']
        self.lti = sys['lti']
        self.covariance = sys['cov']
        self.rf = sys['rf']
        self.trsip = self.lti['trsip']
        self.events = []
        self.match_tracks()   
    
    def match_tracks(self):
        tr2mcf = self.lti['tr2mcf']
        tr2mct = self.lti['tr2mct']
        
        tsphi = self.ltr['tsphi'][tr2mcf]
        tsl = np.arctan(self.ltr['tstnl'][tr2mcf])
        tsppt = self.ltr['trppt'][tr2mcf]
        
        mcppt = self.log['mcppt'][tr2mct]
        mcphi = self.log['mcphi'][tr2mct]
        mcl = self.log['mcl'][tr2mct]
        
        self.dif = {'dif_ppt':mcppt-tsppt,'dif_lambda':mcl-tsl,'dif-phi':mcphi-tsphi}
        
        self.plot_bunch({'dif_ppt':None,'dif_lambda':None, 'dif_phi':None}, maximum=1)
    
    def plot_bunch(self, args, maximum):
        for arg in args.keys():
            x = arg         
            val = args[arg]
            xlabel = ''
            flt = []
            oper = lambda x : x
            if val is not None:
                oper = val
                
            if 'mc' in x:
                flt = ak.flatten(self.log[x], axis=None)
            elif 'dif' in x:
                flt = np.array(self.dif[x])
            else:
                flt = ak.flatten(self.ltr[x], axis=None)
            
            if 'phi' in x or 'lambda' in x:
                xlabel = 'radians'
            
            print(flt)
            plt.hist(oper(flt[(flt<maximum) & (flt>-maximum)]), bins=1000)
            plt.title(label=x)
            plt.yscale('log')
            plt.xlabel(xlabel)
            plt.show()
            
    """
    @staticmethod
    def find_minimum(arg1,arg2):
          cross_prod = ak.cartesian([arg1,arg2])
          a,b = ak.unzip(cross_prod)
          dif = np.abs(a-b)
          return ak.min(dif), dif   
    """

def assign_bunches(folder_path, max_events=1, step=1):
    mlkth = multi_lkth(folder_path, max_events=max_events, step=step)
    bunches = []
    for i in mlkth:
        bunches.append(bunch(i))
        
    return bunches

bunches = assign_bunches('/home/kali/sim/data')
