import numpy as np
import graph_nets as gn
import tensorflow as tf
import itertools

def format_graph(system):
    """
    Takes the system and formats it into a graph_net compatible dictionary
    """
    globals_ = system.globals
    #chi2, ndf
    nodes = system.nodes
    
    #pt, tanlambda, phi, d0, z0, cov, time_of_flight
    edges = [[]]*system.len
    
    i=itertools.product(range(nodes.shape[0]),range(nodes.shape[0]))
    senders=[]
    receivers=[]
    for s,r in i:
        if s==r: continue
        senders.append(s)
        receivers.append(r)
    edges=[[]]*len(senders)
    
    return {'globals':globals_, 'nodes':nodes, 'edges':edges, 'senders':senders, 
            'receivers':receivers}

def format_graphs(dgraphs):
    """
    Takes an array of formatted systems and formats it ready to be used
    by Graph_net
    """
    return gn.utils_tf.data_dicts_to_graphs_tuple(dgraphs)

def formatter(samples):
    dgraphs = []
    
    for i in samples:
        dgraphs.append(format_graph(i)) 
                
    dgraphs = format_graphs(dgraphs)
    
    return {'dgraphs':dgraphs}
