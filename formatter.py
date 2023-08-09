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

def formatter(samples, sample_size=100):
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
    labels = np.empty((0,2),dtype=float)
    n = 0   #num. of tracks added
    
    for v in samples:
        senders = v.senders
        receivers = v.receivers
        arg_len = len(v.globals)
        truth_len = len(v.truth_indexes)
        
        ns = 0
        p = -1
        i = 0
        
        for i in np.arange(sample_size):
            difng = (1+ns)-arg_len
             
            if difng < 0:
                edges = np.array([[],[]])
                
                nodes = np.empty((0,21),dtype=float)
                glob = np.empty((0),dtype=float)
                
                index = [ns]
                p=p*-1
                if p == 1 and i < truth_len:
                   index =  [v.truth_indexes[i]]
                   i+=1
                
                labels = np.append(labels,v.labels[index],axis=0)
                nodes = np.append(nodes,*v.nodes[index],axis=0)
                glob = np.append(glob,*v.globals[index],axis=0)
                
                ns+=1
                n+=1
                
                dgraphs.append(format_graph(glob, nodes, edges, senders, receivers)) 
                
            else:
                break
      
    dgraphs = format_graphs(dgraphs)
    print(dgraphs) 
    
    return {'dgraphs':dgraphs, 'sys':samples, 'labels':labels}
