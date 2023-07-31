import graph_nets as gn
import tensorflow as tf
import numpy as np

def format_graph(glob, nodes, edges, senders, receivers):
    """
    Takes the system and formats it into a graph_net compatible dictionary
    """
    
    return {'globals':tf.convert_to_tensor(glob,dtype=tf.float32), 
            'nodes':tf.convert_to_tensor(nodes,dtype=tf.float32), 
            'edges':tf.convert_to_tensor(edges,dtype=tf.float32), 
            'senders':tf.convert_to_tensor(senders,dtype=tf.int32), 
            'receivers':tf.convert_to_tensor(receivers,dtype=tf.int32)
            }

def format_graphs(dgraphs):
    """
    Takes an array of formatted systems and formats it ready to be used
    by Graph_net
    """
    return gn.utils_tf.data_dicts_to_graphs_tuple(dgraphs)

def formatter(samples, sample_size=1, graph_size=1):
    """
    Formats the data into GraphTuples for the GNN
    
    Parameters
    ----------
    samples : array
        a list of bunches.
    sample_size : int, optional
        The maximum number of tracks to be used. The default is 1.
    graph_size : TYPE, optional
        The maximum number of tracks to be used for a single graph. The default is 1.

    Returns
    -------
    dgraph data : {GraphTuple, array, array}
        All the graph data that is needed for the GNN, including the original samples.

   """
    dgraphs = []
    n = 0   #num. of tracks added
    
    for v in samples:
        ng = 0
        ns = 0
        senders = v.senders
        receivers = v.receivers
        arg_len = len(v.globals)
        
        for i in np.arange((sample_size)/graph_size):
            difng = (graph_size+ns)-arg_len
            print()
            edges = np.array([[0],[1]])*graph_size
            
            labels = np.empty((0,2),dtype=float)
            nodes = np.empty((0,3,21),dtype=float)
            glob = np.empty((0,2),dtype=float)
            
            if difng < 0:
                ran = np.arange(ns, (graph_size+ns))  
                
                labels = np.append(labels,v.labels[ran],axis=0)
                nodes = np.append(nodes,v.nodes[ran],axis=0)
                glob = np.append(glob,v.globals[ran],axis=0)
                
                ng+=graph_size
                ns+=ng
                n+=ng  
                print(f'n:{n},  ns:{ns}')
                print(glob)
                print(nodes)
                print(labels)
                
                dgraphs.append(format_graph(glob, nodes, edges, senders, receivers)) 
            else:
                break
            
        
    print(dgraphs)   
    dgraphs = format_graphs(dgraphs)
    
    return {'dgraphs':dgraphs, 'sys':samples, 'labels':labels}
