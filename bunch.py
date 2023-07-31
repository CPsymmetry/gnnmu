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
        #print(len(self.lti['trndf'][0]))
        #print(len(self.ltr['tstnl'][0]))
        self.ltg = {'trch2':ak.flatten(sys['lti']['trch2']),
                    'trndf':ak.flatten(sys['lti']['trndf'])}
        self.events = pd.DataFrame([])
        
    def graphs_net_setup(self):
        mc2trt = ak.flatten(self.lti['mc2trt'])
        mc2trf = ak.flatten(self.lti['mc2trf'])
        
        trsip = ak.flatten(self.lti['trsip'])
        trsfh = ak.flatten(self.lti['trsfh'])
        trslh = ak.flatten(self.lti['trslh'])
        
        #self.senders = ak.to_numpy(ak.concatenate([trsip,trsfh]))
        #self.receivers = ak.to_numpy(ak.concatenate([trsfh,trslh]))
        
        self.senders = [[0,0],[0,1]]
        self.receivers = [[0,1],[0,2]]
        
        self.edges = np.array([[],[]])
        
        glob = ak.to_numpy(ak.zip(self.ltg.values()))
        nodes = ak.to_numpy(ak.zip(self.ltr.values()))
        
        glob = glob.view(('<f4',len(glob.dtype.names)))
        nodes = nodes.view(('<f4',len(nodes.dtype.names)))[0]
        
        ip = nodes[trsip]
        fh = nodes[trsfh]
        lh = nodes[trslh]
        nodes = np.transpose(np.dstack((ip,fh,lh)),(0,2,1))
        
        self.globals = glob
        self.nodes = nodes
        
        self.labels = np.array([[0.,1.]]*len(self.globals))
        
        #Finds matching values in trsip and mc2trt
        inter,n1,n2 = np.intersect1d(trsip,mc2trt, return_indices=True)
        print(inter)
        print(mc2trf[n2])
        self.labels[mc2trf[n2]] = [1.,0.]
        
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
    """  
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
    def assign_bunches(folder_path, max_events=1, step=1, track_perf=False):
        """
        Creates bunch objects
        
        Parameters
        ----------
        folder_path : TYPE
            DESCRIPTION.
        max_events : int, optional
            the max number of events to load through.. The default is 1.
        step : int, optional
            The step size for looping over chunks of data. The default is 1.
        track_perf: boolean, optional
            Whether a track_perf will be used as the input. The default is False.

        Returns
        -------
        bunches : array
            array of bunch objects.
        """
        mlkth = multi_lkth(folder_path, max_events=max_events, step=step)
        bunches = []
        for i in mlkth:
            bunches.append(bunch(i))
            
        return bunches


if __name__ == '__main__': 
    samples = bunch.assign_bunches('/home/kali/sim/data', track_perf=True)

    """
    bunches[0].plot_bunch(labels = 
                          {'ppt':{'xlabel':'true - reconstructed $p_T$ [MeV]'},
                           'lambda':{'xlabel':'true - reconstructed lambda [radians]'}, 
                           'phi':{'xlabel':'true - reconstructed phi [radians]'}
                           }, 
                          globels = {'range':(-1,1), 'ylabel':'count', 'bins':100})
    """
