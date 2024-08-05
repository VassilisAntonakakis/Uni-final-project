import random
import string
import parser

class Genome:
    def __init__(self):
        N = 10  # Number of triplets

        triplets = []
        for _ in range(N):
            triplet = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
            triplets.append(triplet)

        self.chromosomes = triplets

class Shelf:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.spaceLeft = width
        self.nodes = []
    
    def addNode(self, node):
        self.nodes.append(node)
        self.spaceLeft -= node.width
        
class Bookshelf:
    def __init__(self, rows):
            self.shelves = [rows]
    
    def addShelf(self, shelf):
        self.shelves.append(shelf)
    
    def addNode(self, node):
        pass


# Get the nodes from the parser.py file
nodes = parser.parse_nodes('ibm01/ibm01 1.nodes')
# Create n bins based on their width
n = 4

# go through each entrys and find the min and max width
min_width = 100000
max_width = -1
i = 0

print(nodes)
for node in nodes:
    node_data = node.split(None, 3)
    print(f"Parsing node #{i} with data: {node_data}")
    width = int(node_data[1])
    if width < min_width:
        min_width = width
    if width > max_width:
        max_width = width
    i += 1

# Calculate the width range for each bin
bin_width_range = (max_width - min_width) / n

# Create n empty bin lists
bins = [[] for _ in range(n)]

# Sort the nodes to the correct bin list based on their width
for node in nodes:
    node_data = node.split(None, 3)
    width = int(node_data[1])
    bin_index = int((width - min_width) / bin_width_range)
    print("index", bin_index)
    print(f"Adding node {node_data[0]} to bin #{bin_index}")
    bins[bin_index].append(node)
    
# Print the nodes in each bin
for i, bin_list in enumerate(bins):
    print(f"Bin #{i}: {bin_list}")