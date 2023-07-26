import graph_nets as gn
import tensorflow as tf
import numpy as np

def format_graph(system):
    """
    Takes the system and formats it into a graph_net compatible dictionary
    """
    
    return {'globals':tf.convert_to_tensor(system.globals), 
            'nodes':tf.convert_to_tensor(system.nodes), 
            'edges':tf.convert_to_tensor(system.edges), 
            'senders':tf.convert_to_tensor(system.senders), 
            'receivers':tf.convert_to_tensor(system.receivers)}

def format_graphs(dgraphs):
    """
    Takes an array of formatted systems and formats it ready to be used
    by Graph_net
    """
    return gn.utils_tf.data_dicts_to_graphs_tuple(dgraphs)

def formatter(samples):
    dgraphs = []
    labels = None
    for i in samples:
        dgraphs.append(format_graph(i)) 
        if labels is None:
            labels = i.labels
        else:
            labels = np.concatenate((labels,i.labels))
    
    dgraphs = format_graphs(dgraphs)
    print(dgraphs)
    
    return {'dgraphs':dgraphs, 'sys':samples, 'labels':labels}
