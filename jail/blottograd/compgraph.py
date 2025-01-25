import numpy as np

class CompGraphNode:
    def __init__(self, fct, name, weights):
        self.parents = [] # comp graph nodes that call this function
        self.children = [] # comp graph nodes that are called by this function
        self.fct = fct
        self.name = name
        self.input = None
        self.weights = weights

def topo_sort(node):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v.children:
                build_topo(child)
            topo.append(v)
    build_topo(node)
    return topo

def prop_forward(head_node, x):
    node = head_node
    y = np.copy(x)
    while node.children:
        children = node.children
        for child in children:
            child.input = np.copy(y)
            y = child.fct.forward_fct(y, child.weights)
            # print(f"node {child.name}, input {child.input}, output {y} ")
            node = child
    return y

def prop_backward(head_node, y):
    # get the topologically sorted nodes
    nodes = topo_sort(head_node)
    # init the VJP
    vjp = np.array([1.])
    # for each node, prop the VJP 
    for node in nodes:
        # print(f"node {node.name}, input {node.input}, vec {vjp}")
        vjp = node.fct.vjp_fct(node.input, node.weights, vjp)
    return vjp