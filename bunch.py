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
           'mcpp':[],   #momentum
           'mcphi':[],  #phi
           'mclambda':[]#lambda
           }
    
    ltr = {'tstnl':[],  #tan(lambda)
           'tsome':[],  #omega
           'tsphi':[],  #phi
           'tsdze':[],  #d0
           'tszze':[],  #z0
           'trppt':[],  #track momentum
           }
    
    lti = {'trch2':[],  #chi2
           'trndf':[],  #degrees of freedom
           'trsip':[],  #index of track states in interaction point
           'trsfh':[],  #first hit index
           'trslh':[],  #last hit index
           'trsca':[],  #calorimeter hit index
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
    
    @staticmethod
    def find_minimum(arg1,arg2):
          cross_prod = ak.cartesian([arg1,arg2])
          a,b = ak.unzip(cross_prod)
          dif = np.abs(a-b)
          return ak.min(dif), dif      
    
    def match_tracks(self):
        ppt_min, ppt_dif = self.find_minimum(self.ltr['trppt'],self.log['mcppt'])
        phi_min, phi_dif = self.find_minimum(self.ltr['tsphi'][self.trsip], self.log['mcphi'])
        lambda_min, lambda_dif = self.find_minimum(np.arctan(self.ltr['tstnl'][self.trsip]), self.log['mclambda'])
        
        self.dif = {'dif_ppt':ppt_dif, 'dif_phi':phi_dif, 'dif_lambda':lambda_dif}
        
    
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

def assign_bunches(folder_path, max_bunches=10, step=1):
    mlkth = multi_lkth(folder_path, max_events=max_bunches, step=step)
    bunches = []
    for i in mlkth:
        bunches.append(bunch(i))
        
    return bunches

bunches = assign_bunches('/home/kali/sim/data', step=1, max_bunches=1)
bunches[0].plot_bunch({'dif_ppt':None,'dif_lambda':None, 'dif_phi':None}, maximum=1)
