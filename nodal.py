import numpy as np
import pandas as pd
import networkx as nx

import formatter

"""
edge = [[connection, distance]]
"""

class nodal:
    """
    Creates a system of nodes
    """

    def __init__(self, max_nodes, size, ef=0):
        """
        Parameters
        ----------
        max_nodes : int
            number of nodes to generate

        size : array [x, y]
            The effective size of the node system from [0<xi<x,0<yi<y]

        ef : float
            The edge factor which determines the likelyhood of a new edge being 
            made for a node. 0 - no nodes will be connected, 1 - all nodes will
            be connected.

        Returns
        -------
        A list of nodes and connections

        """
        self.large = 9999.0
        self.nodes = []
        self.ef = ef
        self.size = size
        self.edges = []
        
        self.sn = 0
        self.en = max_nodes-1

        for i in range(max_nodes):
            nodeA = self.add_node()
            for nodeB in self.nodes:
                cont = True
                if cont and ef>0:
                    rn = np.random.rand()
                    if rn <= self.ef and nodeA != nodeB:
                        dist = self.get_magnitude(nodeA, nodeB)
                        self.connect_nodes(nodeA, nodeB, dist)
                    elif rn > self.ef:
                        cont = False
            if len(nodeA.edges) == 0 and len(self.nodes) > 1:
                node_bin = self.nodes
                node_bin = np.delete(node_bin, nodeA.un)
                nodeB = np.random.choice(node_bin)

                dist = self.get_magnitude(nodeA, nodeB)
                self.connect_nodes(nodeA, nodeB, dist)
                
        
        self.graph = formatter.format_graph(self)
        

    def get_magnitude(self, nodeA, nodeB):
        posA = nodeA.pos
        posB = nodeB.pos

        x = posA[0] - posB[0]
        y = posA[1] - posB[1]

        return np.sqrt(x**2 + y**2)

    def add_node(self):
        x = np.random.rand()*self.size[0]
        y = np.random.rand()*self.size[1]

        nodeA = nodec(nsys=self, pos=(x, y), un=len(self.nodes))
        self.nodes.append(nodeA)
        return nodeA

    def connect_nodes(self, nodeA, nodeB, distance):
        nodeB.add_edge(nodeA, distance)
        self.edges.append([nodeB, nodeA, distance])
    
    """
    def se_nodes(self):
        arr = np.arange(0, len(self.nodes))
        rnd1 = np.random.choice(arr)
        arr = np.delete(arr, rnd1)
        rnd2 = np.random.choice(arr)

        nodeA = self.nodes[rnd1]
        nodeB = self.nodes[rnd2]

        return nodeA, nodeB
    """
    
    def labels(self, path):
        """
        Creates an array of labels of 0 or 1. 0 if edge is not on true path or
        1 if edge is on true path.
        """
        labels = []
        
        edges = self.graph['edges']
        senders = self.graph['senders']
        receivers = self.graph['receivers']
        print(path)
        pathun = np.array([node.un for node in path])
        print(pathun)
        
        for i, edge in enumerate(self.edges):
            sender = senders[i]
            receiver = receivers[i]
            
            nn = np.where(pathun == sender)[0]

            if pathun.size>0 and (pathun.size-1)>nn and pathun[nn+1] == receiver:
                a = [0.,1.]
                #if pathun[0]==sender or pathun[len(pathun)-1] == receiver:
                    #a = [1.,1.]
            else:
                a = [1.,0.]
                
            labels.append(a)
            
        return labels


    def shortestpath(self):
        """
        Creates the shortest path from start_node to end_node using the
        Dijkstra's algorithm'
        """
        start_node = self.nodes[self.sn]
        end_node = self.nodes[self.en]
        
        uns = self.sn
        unf = self.en
        lno = len(self.nodes)
        
        
        uvis = np.full(shape=lno, fill_value=0.0)
        uvis[uns] = 1
        tentvis = np.full(shape=lno, fill_value=self.large)
        tentvis[uns] = 0

        path = [[start_node]]*lno
        
        def process(current_node, path):
            #print(f'in: {current_node}')
            
            uc = current_node.un
            if current_node != end_node:
                min_node = [None, self.large]
                uvis[uc] = 1
                for index, data in current_node.edges.iterrows():
                    node = data[0]
                    edge = data[1]
                    uv = uvis[node.un]
                    if uv == 0:
                        #print(f'      {node}')
                        tentv = tentvis[node.un]
                        dist = edge + tentvis[uc]
                        unn = node.un
                        if tentv > dist:
                            tentvis[unn] = dist
                            path[unn] = [] + path[uc]
                            path[unn].append(node)
                        if min_node[1] > tentvis[unn]:
                            min_node = [node, dist]

                if min_node[0] is not None:
                    #print(f'out: {min_node[0]}')
                    return process(min_node[0], path)
                else:
                    tab = np.array([[self.large, self.large]])
                    for ind, i in enumerate(uvis):
                        l = tentvis[ind]
                        if i == 0 and l<self.large:
                            tab = np.append(tab, [[ind,l]], axis=0)
                    imin = tab[np.argmin(tab[:,1])][0]
                    node = self.nodes[int(imin)]
                    return process(node, path)

            else:
                """
                #print("end node reached")
                min_node = [None, self.large, []]
                for index, data in current_node.edges.iterrows():
                    node = data[0]
                    edge = data[1]
                    uv = uvis[node.un] 
                    tentv = tentvis[unf]
                    unn = node.un
                    dist = edge + tentvis[unn]
                    if tentv > dist:
                        tentvis[unf] = dist
                    if min_node[1]>tentvis[unn]:
                        min_node = [node, dist, path[uc]]
                """
                return [0, 0, path[len(path)-1]]
        
        path_dist = process(start_node, path)
        dist_raw = path_dist[1]
        path_raw = path_dist[2]
        labels = self.labels(path_raw)
        
        return {'dist': dist_raw, 'labels': labels, 'path': path_raw}
    
    def draw(self):
        """
        Creates a graph of the nodes and edges utilising networkx
        """
        g=nx.DiGraph()
        edges = []
        for node in self.nodes:
            g.add_node(node.un, pos=node.pos)
        for nodeA in self.nodes:
            for i, nodeB in enumerate(nodeA.edges['nodes'].values):
                dist = nodeA.edges['edges'].values[i]
                edges.append([nodeA.un,nodeB.un,dist])
                g.add_edge(nodeA.un,nodeB.un)
        pos = nx.get_node_attributes(g,'pos')
        
        edge_labels = [[n1.un, n2.un, '%.3f' % dist1] for n1, n2, dist1 in self.edges]
        
        g.add_weighted_edges_from(edge_labels)
        
        nx.draw(g, pos, edge_color='black', width=1, linewidths=1,
                node_size=500, node_color='pink', alpha=0.9,
                labels={node: node for node in g.nodes()})  

        return g
            

class nodec:
    """
    Creates node objects
    """

    def __init__(self, nsys, pos, un):
        self.edges = pd.DataFrame(columns=["nodes", "edges"])
        self.parent = nsys
        self.pos = pos
        self.un = un
        return

    def add_edge(self, nodeB, dist):
        d1 = pd.DataFrame([[nodeB, dist]], columns=["nodes", "edges"])
        self.edges = pd.concat([self.edges, d1])

    def __repr__(self):
        return f"node{self.un}"
    
    
"""
node_sys = nodal(10, size=[10, 10], ef=0)
path = node_sys.shortestpath()
node_sys.draw()
print(path)
"""