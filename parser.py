import igraph as ig
import numpy as np

def parse_nodes(nodes_filepath):
    with open(nodes_filepath, 'r') as f:
        lines = f.readlines()
    nodes = [line.strip() for line in lines if line[0] == ' ']
    #print(nodes)
    return nodes
 # type: ignore

def parse_edges(edges_filepath):
    edges = []
    
    with open(edges_filepath, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.startswith("NetDegree"):
            num_edges = int(line.strip().split(":")[1])
            edge_line=lines[i+1:i+num_edges+1]
            
            for edge in edge_line:
                edge = edge.strip()
                edge_data = edge.split(None, 2)
                edges.append(edge_data)
            
            continue
    #print(edges)
    return edges
 # type: ignore


# edges = parse_edges('ibm01/ibm01 1.nets')
# for edge in edges:
#     print(edge)

# nodes = parse_nodes('ibm01/ibm01 1.nodes')
# for node in nodes:
#     node_data = node.split(None, 3)
#     print(node_data)
# #     #print(node)