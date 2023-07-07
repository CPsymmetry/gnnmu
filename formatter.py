import numpy as np

import graph_nets as gn
import tensorflow as tf

def format_graph(node_sys):
    """
    Takes a node system and formats it into a graph_net compatible dictionary
    """
    globals_ = []
    edge_list = []
    senders = []
    receivers = []
    
    for edge in node_sys.edges:
        s = edge[0]
        r = edge[1]
        dist = edge[2]
        senders.append(s.un)
        receivers.append(r.un)
        #converts floats to tf.float32
        rd = tf.cast(dist, tf.float32)
        edge_list.append(rd)
            
    nodes = [[]]*len(senders)*2
    
    edges = np.array([weight for weight in edge_list], ndmin=2).T
    
    return {'globals':globals_, 'nodes':nodes, 'edges':edges, 'senders':senders, 
            'receivers':receivers}

def format_graphs(dgraphs):
    """
    Takes an array of formatted node systems and formats it ready to be used
    by Graph_net
    """
    return gn.utils_tf.data_dicts_to_graphs_tuple(dgraphs)

def formatter(num_samples, mimax=[3,5], size=[10,10], ef=0.0, truth_value=False):
    """
    
    Parameters
    ----------
    num_samples : int
        Number of node systems to sample
    mimax : array, optional
        Minimum and maximum values of nodes to sample from
    size : array, optional
        X and Y size of the canvas where the nodes lie. The default is [10,10].
    ef : float, optional
        Varies the number of node connections made. 0 - minimum connections,
        1 - maximum connections. The default is 0.
    truth_value : Boolean, optional
        Whether to create the true path for each node system sampled. 
        The default is False.

    Returns
    -------
    dict
        [formatted dgraphs, node_systems for each graph, true path for each graph].

    """
    from nodal import nodal
    
    node_systems = []
    dgraphs = []
    path_systems = []
    labels = []
    
    for i in range(num_samples):
        max_nodes = 3#np.random.randint(mimax[0],mimax[1])
        node_sys = nodal(max_nodes, size, ef)
        node_systems.append(node_sys)
        
        dgraphs.append(format_graph(node_sys))
        
        if truth_value:
            path_systems.append(node_sys.shortestpath()) 
        else:
            path_systems.append({'dist':0, 'labels':[], 'path':[]})
     
    for path_system in path_systems:
        labels+=path_system['labels']
                
    dgraphs = format_graphs(dgraphs)
    
    return {'dgraphs':dgraphs, 'node_systems':node_systems, 'path_systems':path_systems, 'labels':labels}