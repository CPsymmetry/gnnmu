import awkward as ak
import numpy as np
import pandas as pd
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
           'mclam':[]     #lambda
           }
    
    ltr = {'tstnl':[],  #tan(lambda)
           'tsome':[],  #omega
           'tsphi':[],  #phi
           'tsdze':[],  #d0
           'tszze':[],  #z0
           'tsppt':[],  #track state momentum
           'tscov':[],  #covariance
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
"""

lam_max = 75/360 * 2*np.pi

class bunch:
    def __init__(self, sys):
        self.log = sys['log']
        self.ltr = sys['ltr']
        self.lti = sys['lti']
        self.rf = sys['rf']
        print(len(self.lti['trndf'][0]))
        print(len(self.ltr['tstnl'][0]))
        self.ltg = {'trch2':ak.flatten(sys['lti']['trch2']),
                    'trndf':ak.flatten(sys['lti']['trndf'])}
        self.events = pd.DataFrame([])
        
    def graphs_net_setup(self):
        mc2trt = self.lti['mc2trt'][0]
        mc2trf = self.lti['mc2trf'][0]
        
        self.globals = ak.to_numpy(ak.zip(self.ltg.values())).view(('<f4',2))
        self.nodes = ak.to_numpy(ak.zip(self.ltr.values())).view(('<f4',6))[0]
        #self.globals = self.nodes
        self.senders = ak.to_numpy(ak.concatenate([*self.lti['trsip'],*self.lti['trsfh'],*self.lti['trslh']]))
        self.receivers = ak.to_numpy(ak.concatenate([*self.lti['trsfh'],*self.lti['trslh'],*self.lti['trslh']]))
        self.edges = np.array([[]]*len(self.senders))
        
        self.labels = np.array([[0.,1.]]*len(self.globals))
        
        trsip = self.lti['trsip']
        inter,n1,n2 = np.intersect1d(ak.flatten(trsip),mc2trt, return_indices=True)
        self.labels[mc2trf[n2]] = [1.,0.]
        
        """
        print(self.globals)
        print(self.nodes)
        print(self.senders)
        print(self.receivers)
        print(self.edges)
        """
    
    
    
    def match_tracks(self, selection = True):
        mc2trt = self.lti['mc2trt']
        mc2trf = self.lti['mc2trf']
        
        track_state_info = {'tsppt':'ppt','tsphi':'phi','tstnl':'lam','tsome':'omega','tsdze':'d0','tszze':'z0'}
        track_info = {'trch2':'ch2','trndf':'ndf'}
        
        pre = self.log['mcgst'][mc2trf] != None
        
        if selection:
            gen = self.log['mcgst'][mc2trf]
            cha = self.log['mccha'][mc2trf]
            lam = self.log['mclam'][mc2trf]
        #selection that particles must be stable and must not have neutral charge + lambda < 75 degrees
            pre = (gen==1) & (cha!=0) & (lam<lam_max)  
        
        for x in track_state_info:
            col = track_state_info[x]
            self.events[col] = ak.flatten(self.ltr[x][mc2trt][pre])
            
        for x in track_info:
            col = track_info[x]
            self.events[col] = ak.flatten(self.lti[x][mc2trt][pre])
        
    def plot_bunch(self, labels, globels={}, bunches=None):
        
        bunches = self or bunches        
        for x in labels.keys():   
            flt = []
            local = labels[x]
            
            local_keys = local.keys()
            global_keys = globels.keys()
            
            xlabel = ''
            ylabel = ''
            ran = (-10000,10000)
            bins = 1000
            operation = lambda x : x
            
            for key in local_keys:
                if 'xlabel' in key:
                    xlabel = local[key]
                elif 'ylabel' in key:
                    ylabel = local[key]
                elif 'range' in key:
                    ran = local[key]
                elif 'bins' in key:
                    bins = local[key]
            
            for key in global_keys:
                if 'xlabel' in key:
                    xlabel = globels[key]
                elif 'ylabel' in key:
                    ylabel = globels[key]
                elif 'range' in key:
                    ran = globels[key]
                elif 'bins' in key:
                    bins = globels[key]

            if 'mc' in x:
                flt = ak.flatten(self.log[x], axis=None)
            else:
                flt = np.array(self.events[x])
            
            if 'phi' in x or 'lambda' in x:
                xlabel = 'radians'
            
            mini, maxi = ran
            
            plt.hist(operation(flt[(flt<maxi) & (flt>mini)]), bins=bins)
            plt.title(label=x)
            plt.yscale('log')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()
     
    @staticmethod
    def assign_bunches(folder_path, max_events=1, step=1):
        mlkth = multi_lkth(folder_path, max_events=max_events, step=step)
        bunches = []
        for i in mlkth:
            bunches.append(bunch(i))
            
        return bunches
    
    @staticmethod 
    def trackPerf_to_bunch(folder_path, max_events=1, step=1):
        mlkth = multi_lkth(folder_path, max_events=max_events, step=step, track_perf=True)
        bunches = []
        for i in mlkth:
            bun = bunch(i)
            bunches.append(bun)
            bun.match_tracks()
            bun.graphs_net_setup()
            
        return bunches


if __name__ == '__main__': 
    samples = bunch.trackPerf_to_bunch('/home/kali/sim/data')

    """
    bunches[0].plot_bunch(labels = 
                          {'ppt':{'xlabel':'true - reconstructed $p_T$ [MeV]'},
                           'lambda':{'xlabel':'true - reconstructed lambda [radians]'}, 
                           'phi':{'xlabel':'true - reconstructed phi [radians]'}
                           }, 
                          globels = {'range':(-1,1), 'ylabel':'count', 'bins':100})
    """
