import sonnet as snt
import graph_nets as gn

class INModel(snt.Module):
    def __init__(self):
        """
        A graph neural network. Sets up network function.
        """
        # Create the graph network.
        super(INModel, self).__init__()
       
        size = 128
        
        self.gn_module = gn.modules.GraphNetwork(
           edge_model_fn=lambda: snt.nets.MLP([size, size, 0]),
           node_model_fn=lambda: snt.nets.MLP([size, size, ]),
           global_model_fn=lambda: snt.nets.MLP([size, size,6])        
           )          
       
    def __call__(self, data):
        """
        Takes data and parses it through the graph network
        
        data: dictionary, [formatted_graphs, node_system, true_path]
        """
        #print(f"dgraph data {data['dgraphs']}")
        print(data['dgraphs'])
        gnn = self.gn_module(data['dgraphs'])
        
        
        return gnn
    
    
        
#input_graphs, node_systems, path = formatter(10, truth_value = True)
