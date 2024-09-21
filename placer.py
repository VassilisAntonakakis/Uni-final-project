import random
import string
import parser
import gene
import Shelf

N = 4

# Get the nodes from the parser.py file
nodes = parser.parse_nodes('ibm01/ibm01 1.nodes')
# Create n bins based on their width
n = 4

# go through each entry's and find the min and max width
min_width = 100000
max_width = -1
i = 0

#print(nodes)
for node in nodes:
    if node.getWidth() < min_width:
        min_width = node.getWidth()
    if node.getWidth() > max_width:
        max_width = node.getWidth()
        

# Calculate the width range for each bin
bin_width_range = (max_width - min_width) / n

# Create n empty bin lists
bins = [[] for _ in range(n)]

# Sort the nodes to the correct bin list based on their width
# for node in nodes:
#     node_data = node.split(None, 3)
#     width = int(node_data[1])
#     bin_index = int((width - min_width) / bin_width_range)
#     print("index", bin_index)
#     print(f"Adding node {node_data[0]} to bin #{bin_index}")
#     bins[bin_index].append(node)
    
#Print the nodes in each bin
# for i, bin_list in enumerate(bins):
#     print(f"Bin #{i}: {bin_list}")

# Calculate N bins based on the width of the nodes
for node in nodes:
    width = int(node.getWidth())
    bin_index = min(int((width - min_width) / bin_width_range), n - 1)
    #print("index", bin_index)
    #print(f"Adding node {node_data[0]} to bin #{bin_index}")
    bins[bin_index].append(node)

# Print the number of nodes in each bin
for i, bin_list in enumerate(bins):
    print(f"Bin #{i}: {len(bin_list)} nodes")
    
# Create a gene with N genomes where each genome is the posibility of choosing a node from this bin
# g = gene.gene(n, N)
# print(f"Gene: {g.getGene()}")
# print(f"Genomes: {g.getGenomes()}")
# print(f"Specific genome 0: {g.getSpecificGenome(0)}")
# print(f"Specific genome 1: {g.getSpecificGenome(1)}")
# print(f"Specific genome 2: {g.getSpecificGenome(2)}")
# print(f"Specific genome 3: {g.getSpecificGenome(3)}")
# print(f"Gene value: {g.getGeneValue(0)}")
# print(f"Gene value: {g.getGeneValue(1)}")
# print(f"Gene value: {g.getGeneValue(2)}")
# print(f"Gene value: {g.getGeneValue(3)}")

# # For all available shelves, try to place the nodes from the bins. The choice of the bin is based on the gene value
# shelves = []
# for i in range(N):
#     shelf = Shelf(10, 16)
#     for j in range(n):
#         node = bins[j][g.getGeneValue(i) % len(bins[j])]
#         shelf.addNode(node)
#     shelves.append(shelf)
    
# for i, shelf in enumerate(shelves):
#     print(f"Shelf #{i} has {len(shelf.nodes)} nodes left")
#     for node in shelf.nodes:
#         print(node)

shelves = parser.parse_scl_file('ibm01/ibm01 1.scl')

#count the number of shelves found
print(shelves)