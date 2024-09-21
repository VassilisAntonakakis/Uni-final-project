import igraph as ig
import numpy as np
import re
import sys
import Node
import Shelf

def parse_nodes(nodes_filepath):
    with open(nodes_filepath, 'r') as f:
        lines = f.readlines()
    node_strings = [line.strip() for line in lines if line[0] == ' ']
    #print(nodes)
    nodes = []
    i = 0
    for node in node_strings:
        node_data = node.split(None, 3)
        nodes.append(Node.Node(node_data[0], int(node_data[1]), int(node_data[2])))
        #print(f"Node {i}: {nodes[i].getName()} with width {nodes[i].getWidth()} and height {nodes[i].getHeight()}")
        i += 1
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

def parse_scl_file(filename):
    shelves = []
    with open(filename, 'r') as file:
        lines = file.readlines()

    current_shelf = None
    for line in lines:
        line = line.strip()
        if line.startswith("CoreRow Horizontal"):
            #print("Shelf found")
            current_shelf = {}
        elif line.startswith("End"):
            if current_shelf:
                shelf = Shelf(current_shelf['coordinate'],
                              current_shelf['height'],
                              current_shelf['sitewidth'],
                              current_shelf['sitespacing'],
                              current_shelf['siteorient'],
                              current_shelf['sitesymmetry'],
                              current_shelf['subrow_origin'],
                              current_shelf['num_sites']
                            )
                shelves.append(shelf)
                current_shelf = None
        elif line.startswith("#") or line == "": # Skip comments and empty lines
            continue
        else:
            print(f"Line: {line}")
            match = re.match(r'\s*(\w+)\s*:\s*([^\s]+)', line)
            
            if match:
                key, value = match.group(1), match.group(2)
                print(f"Key: {key}, Value: {value}")
                if key.lower() == "Coordinate":
                    current_shelf[key.lower()] = int(value)
                elif key.lower() == "Height":
                    current_shelf[key.lower()] = int(value)
                elif key.lower() == "Sitewidth":
                    current_shelf[key.lower()] = int(value)
                elif key.lower() == "Sitespacing":
                    current_shelf[key.lower()] = int(value)
                elif key.lower() == "Siteorient":
                    current_shelf[key.lower()] = value
                elif key.lower() == "Sitesymmetry":
                    current_shelf[key.lower()] = value
                elif key.lower() == "Subrow_origin":
                    current_shelf[key.lower()] = int(value)
                elif key.lower() == "Numsites":
                    current_shelf[key.lower()] = int(value)
                else:
                    print(f"Unknown key: {key}")
    
    return shelves
# edges = parse_edges('ibm01/ibm01 1.nets')
# for edge in edges:
#     print(edge)

# nodes = parse_nodes('ibm01/ibm01 1.nodes')
# for node in nodes:
#     node_data = node.split(None, 3)
#     print(node_data)
# #     #print(node)