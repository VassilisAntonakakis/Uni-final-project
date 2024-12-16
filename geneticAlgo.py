import configparser
from datetime import datetime
import pandas as pd
import openpyxl
from pathlib import Path
import random
import os
from copy import deepcopy
import string
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import numpy as np

class Module:
    def __init__(self, name, width, height, terminal=False):
        self.name = name
        self.width = width
        self.height = height
        self.x = 0
        self.y = 0
        self.terminal = terminal

    def __repr__(self):
        return f"Module(name={self.name}, width={self.width}, height={self.height}, x={self.x}, y={self.y}, terminal={self.terminal})"

class Shelf:
    def __init__(self, height, max_width, name):
        self.name = name
        self.height = height
        self.max_width = max_width
        self.used_width = 0
        self.contained_modules = []
        
    def add_module(self, module):
        self.contained_modules.append(module)
        self.used_width += module.width
    
    def reset_shelf(self):
        self.used_width = 0
        self.contained_modules.clear()
    
    def add_module_to_shelf(self, module):
        # If there is space available in the shelf
            x = self.find_space(module.width)
            if x is not None:
                #print(f"Found space at {x} in shelf {shelves.index(self)} for module {module.name}")
                module.x = x
                module.y = sum([s.height for s in shelves if s != self]) + (self.height - module.height) / 2
                self.insert_module(module, x)
                self.check_module_overlap()
                #print(f"Placed module {module.name} at ({module.x}, {module.y}) on shelf {self.name}\n {self.contained_modules}")
                return True
                #shelf.check_module_overlap()
            else:
                # If no space was found in any shelf
                #print(f"Warning: Could not place module {module.name} due to size constraints.")
                return False
    
    def insert_module(self, module, x):
        # Insert a module in the contained modules list at the correct position based on x-coordinate
        index = 0
        while index < len(self.contained_modules) and ((self.contained_modules[index].x + self.contained_modules[index].width) <= x):
            index += 1
        if index < len(self.contained_modules):
            self.contained_modules.insert(index, module)
        elif index == len(self.contained_modules):
            self.contained_modules.append(module)
        else:
            pass#print("Warning: Module not inserted at correct index!")
        
    def check_module_overlap(self):
        # Check if any modules in the shelf overlap with each other
        previous_x = 0
        for module in self.contained_modules:
            if module.x < previous_x:
                #print(f"Warning: Module {module.name} overlaps with previous module(s) in the shelf!\n")
                return True
            previous_x = module.x + module.width
        return False
    
    def find_space(self, size_needed):
        # If there are no modules in the shelf, start from 0 if size_needed fits
        if not self.contained_modules:
            return 0 if size_needed <= self.max_width else None

        # Check if there’s space before the first module in the shelf
        first_module = self.contained_modules[0]
        if first_module.x >= size_needed:
            return 0  # Place at the start of the shelf

        # Check spaces between consecutive modules
        try:
            previous_x = first_module.x + first_module.width
            for index in range(1, len(self.contained_modules)):
                current_module = self.contained_modules[index]
                space = current_module.x - previous_x
                if space >= size_needed:
                    return previous_x
                previous_x = current_module.x + current_module.width
                
            # Check space after the last module in the shelf
            last_module = self.contained_modules[-1]
            if (self.max_width - (last_module.x + last_module.width)) >= size_needed:
                return last_module.x + last_module.width

            # No space found
            return None                 
        except IndexError:
            return None

    def __repr__(self):
        return f"Shelf(height={self.height}, max_width={self.max_width}, used_width={self.used_width})"

class Rail:
    def __init__(self, core_width, core_height, buffer_distance):
        self.buffer_distance = buffer_distance
        self.continuous_spaces = {
            "bottom": [0, core_width, -buffer_distance - 1],
            "right": [0, core_height, core_width + buffer_distance],  # Offset right rail
            "top": [0, core_width, core_height + buffer_distance],
            "left": [0, core_height, -buffer_distance - 1]
        }
        self.occupied_rects = []  # Track occupied rectangles to avoid overlaps

    def can_place(self, x, y, width, height):
        """Check if a terminal can be placed without overlapping."""
        for rect in self.occupied_rects:
            rx, ry, rw, rh = rect
            if not (x + width <= rx or x >= rx + rw or y + height <= ry or y >= ry + rh):
                return False
        return True

    def add_terminal(self, x, y, width, height):
        """Add a terminal to the rail."""
        self.occupied_rects.append((x, y, width, height))

def place_terminals_on_rails(terminals, rail, modules, nets, core_width, core_height):
    """
    Place terminals evenly around the rails of the core area.

    Parameters:
        terminals (list): List of terminal nodes (Module objects).
        rail (Rail): Rail object defining the continuous rail spaces.
        modules (list): List of all modules in the design.
        nets (list): List of nets connecting the modules.
        core_width (float): Width of the core area.
        core_height (float): Height of the core area.
    """
    if not terminals:
        print("No terminals to place.")
        return

    num_terminals = len(terminals)
    rail_perimeter = 2 * (core_width + core_height)  # Total rail perimeter
    spacing = rail_perimeter / num_terminals  # Spacing between terminals

    # Initialize placement variables
    current_position = 0  # Tracks position along the rail perimeter

    for terminal in terminals:
        placed = False
        remaining_position = current_position  # Current position relative to rail segments

        # Iterate through the rails in clockwise order: bottom, right, top, left
        for side, (start, end, fixed) in [
            ("bottom", (0, core_width, -rail.buffer_distance - terminal.height)),
            ("right", (0, core_height, core_width + rail.buffer_distance)),
            ("top", (0, core_width, core_height + rail.buffer_distance)),
            ("left", (0, core_height, -rail.buffer_distance - terminal.width)),
        ]:
            rail_length = end - start  # Length of the current rail

            # Check if remaining_position is within this rail
            if remaining_position < rail_length:
                if side in ["bottom", "top"]:  # Horizontal rails
                    x = start + remaining_position
                    y = fixed  # Fixed y-coordinate
                elif side in ["left", "right"]:  # Vertical rails
                    x = fixed  # Fixed x-coordinate
                    y = start + remaining_position

                # Check if the terminal can be placed
                if rail.can_place(x, y, terminal.width, terminal.height):
                    terminal.x, terminal.y = x, y
                    rail.add_terminal(x, y, terminal.width, terminal.height)
                    placed = True
                    #print(f"Placed terminal {terminal.name} at ({x:.2f}, {y:.2f}) on the {side} rail.")
                    break

            # Update remaining_position for the next rail
            remaining_position -= rail_length

        # Move to the next terminal position
        current_position += spacing

        if not placed:
            print(f"Warning: Unable to place terminal {terminal.name}.")
    
def filenames_parser(aux_filepath):
    filenames = []
    with open(aux_filepath, "r") as file:
        for line in file:
            filelist = line.split(":")[1].strip()
            filenames = filelist.split(" ")
    
    for i in range(len(filenames)):
        filenames[i] = filenames[i].strip()
    
    print("Filenames parsed: ", filenames)
    return filenames

def parse_nodes(file_path):
    nodes = []
    num_nodes = None
    num_terminals = None
    start_parsing = False
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue

            if "NumNodes" in line:
                num_nodes = int(line.split(":")[1].strip())
            elif "NumTerminals" in line:
                num_terminals = int(line.split(":")[1].strip())
                print(f"Expected NumNodes: {num_nodes}, Expected NumTerminals: {num_terminals}")
                start_parsing = True
                continue

            if start_parsing:
                terminalNode = False
                if "terminal" in line: terminalNode = True 
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    width = float(parts[1])
                    height = float(parts[2])
                    nodes.append(Module(name, width, height, terminalNode))
    print("Node parsing complete: Found", len(nodes), "modules")
    return nodes

def parse_shelves(file_path):
    shelves = []
    start_parsing = False

    with open(file_path, 'r') as file:
        height = None
        width = None
        shelf_parsed = 0
        for line in file:
            #print("Parsing line: ", line)
            # Ignore comment lines and empty lines
            if line.startswith("#") or not line.strip():
                continue
            
            if "NumRows" in line:
                num_rows = int(line.split(":")[1].strip())
                print(f"Expected NumRows: {num_rows}")
                continue
            
            if "CoreRow" in line:
                #print(f"CoreRow found! Parsing shelf data for {shelf_parsed}")
                start_parsing = True
                data_row = 0
                continue
            
            if "End" in line:
                start_parsing = False
                shelf_parsed += 1
                shelves.append(Shelf(height, width, shelf_parsed))
                continue
            
            if start_parsing:
                data_row += 1
                if data_row == 7:
                    parts = line.split(":")
                    #print("Parts: ", parts)
                    for i in range(len(parts)):
                        #print("Part: ", parts[i])
                        parts[i].strip()
                    #print("Parts after: ", parts)
                    width = float(parts[2])
                    
                elif data_row == 2:
                    parts = line.split(":")
                    for part in parts:
                        part.strip()
                    height = float(parts[1])
                else:
                    continue
    print("Shelf parsing complete! Found", len(shelves), "shelves")
    return shelves

# New function to parse nets
def parse_nets(file_path):
    nets = []
    start_parsing = False
    
    with open(file_path, 'r') as file:
        current_net = []
        for line in file:
            if line.startswith("#") or not line.strip() or "NumPins" in line:
                continue

            if "NumNets" in line:
                num_nets = int(line.split(":")[1].strip())
                print(f"Expected NumNets: {num_nets}")
                start_parsing = True
                continue

            if start_parsing:
                if "NetDegree" in line:
                    #print("NetDegree found!")
                    if current_net:
                        nets.append(current_net)
                    current_net = []
                else:
                    parts = line.split()
                    current_net.append(parts[0])

        if current_net:
            nets.append(current_net)

    print(f"Net parsing complete! Found {len(nets)} nets")
    return nets

def parse_custom_nets(file_path):
    nets = []  # List to hold each parsed net as a list of nodes
    current_net = []  # Temporary list to collect nodes for the current net
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove any extra whitespace
            if line.startswith("NetDegree"):
                # If there's an existing net, save it before starting a new one
                if current_net:
                    nets.append(current_net)
                    current_net = []  # Reset for the new net
            elif line:  # Add the line content (node) to the current net if it's not empty
                current_net.append(line)
        
        # Append the last net if there was one
        if current_net:
            nets.append(current_net)
    
    return nets

# HPWL calculation function
def calculate_hpwl(modules, nets):
    total_hpwl = 0
    
    for net in nets:
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        # For each module connected by this net
        for module_name in net:
            module = next((m for m in modules if m.name == module_name), None)
            if module:
                min_x = min(min_x, module.x)
                max_x = max(max_x, module.x)
                min_y = min(min_y, module.y)
                max_y = max(max_y, module.y)
            else:
                pass#print(f"Warning: Module {module_name} not found in module list!")

        # Calculate HPWL for this net
        hpwl = (max_x - min_x) + (max_y - min_y)
        total_hpwl += hpwl

    return total_hpwl

# Reset all shelves for the next individual
def reset_shelves(shelves):
    for shelf in shelves:
        shelf.reset_shelf()

# Modified fitness function to use HPWL
def fitness(chromosome, nets, shelves):
    """
    Calculate the fitness of a chromosome by placing modules and terminals
    and then computing the HPWL.
    """
    # Reset shelves to start fresh for each chromosome evaluation
    #reset_shelves(shelves)

    # Step 1: Place non-terminal modules based on the chromosome
    individual = create_individual(non_terminals, shelves, chromosome)
    if individual is None:
        return float('inf')  # Penalize if placement failed

    # Step 2: Separate terminals and place them on the rails
    #terminals = [module for module in modules if module.terminal]
    #place_terminals_on_rails(terminals, rail, individual, nets, core_width, core_height)

    # Step 3: Calculate HPWL for the combined placement
    hpwl = calculate_hpwl(individual + terminals, nets)

    return hpwl

def mutate(chromosome, mutation_rate=0.2):
    """Applies mutation to the chromosome based on the mutation rate."""
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            swap_idx = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[swap_idx] = chromosome[swap_idx], chromosome[i]
    return chromosome

def plot_fitness_progress(fitness_progress, xOp):
    plt.figure()
    plt.plot(fitness_progress, marker='o')
    plt.title(F"HPWL Fitness Over Generations ({xOp})")
    plt.xlabel("Generation")
    plt.ylabel("Best HPWL (Lower is Better)")
    plt.grid(True)
    plt.show()

# Genetic algorithm functions
def create_individual(modules, shelves, chromosome):
    modules_copy = deepcopy(non_terminals)
    reset_shelves(shelves)
    individual = []

    for module_index in chromosome:
        #print(f"Placing module {module_index}\n")
        module = modules_copy[module_index]
        placed = False

        for shelf in shelves:
            if shelf.add_module_to_shelf(module):
                individual.append(module)
                #print(f"Placed module {module.name} at ({module.x}, {module.y}) on shelf {shelves.index(shelf)}")
                placed = True
                break

        if not placed:
            #print(f"Failed to place module {module.name}")
            return None
    
    return individual

def starting_placement(modules, shelves):
    PERCENTAGE = float(config['OPTIONS']['PERCENTAGE'])
    #select a random 80% of the modules to place excluding the terminals
    #modulesToPlace = random.sample([module for module in modules if not module.terminal], int(len(modules) * PERCENTAGE))
    
    # Simple placement algorithm to place modules row by row
    for module in modules:
        placed = False
        for shelf in shelves:
            if shelf.used_width + module.width <= shelf.max_width and module.height <= shelf.height:
                # Place the module
                module.x = shelf.used_width
                #module.y = sum(s.height for s in shelves if s != shelf)
                #shelf.used_width += module.width
                shelf.add_module(module)
                placed = True
                #print(f"Placed module {module.name} at ({module.x}, {module.y}) on shelf {shelves.index(shelf)}")
                break
        if not placed:
            #print(f"Warning: Could not place module {module.name} due to size constraints.")
            #return None  # Return None if placement fails
            pass    
    
    #for each shelf in the shelves, print the space utilization percentage
    for shelf in shelves:
        print(f"Shelf {shelves.index(shelf)} utilization: {shelf.used_width / shelf.max_width * 100:.2f}%")
    
    #print("Initial placement HPWL: ", )
    #visualize_row_based_design(shelves, "Initial Placement")
    return calculate_hpwl(modules + terminals, nets)

def generate_string_mapping(modules):
    # Determine the number of mappings needed
    num_modules = len(modules)
    alphabet = string.ascii_uppercase
    mapping = []
    letters_per_mapping = 1
    
    # Calculate required number of letters per mapping
    while 26 ** letters_per_mapping < num_modules:
        letters_per_mapping += 1

    # Generate unique strings for each module
    for i in range(num_modules):
        chars = []
        n = i
        # Convert the index into a base-26 representation for the required letters per mapping
        for _ in range(letters_per_mapping):
            chars.append(alphabet[n % 26])
            n //= 26
        mapping.append(''.join(reversed(chars)))

    # Return both the mapping and the number of letters per mapping
    return mapping, letters_per_mapping

def create_initial_population(values, population_size):
    # Calculate the total number of possible unique permutations
    joins = []
    for _ in range(population_size):
        # Create a copy of the list and shuffle it to get a random join
        shuffled = values[:]
        random.shuffle(shuffled)
        # Join the shuffled list into a single string and add it to the result
        joins.append(''.join(shuffled))
    return joins

def place_modules_in_shelves(modules, shelves, chromosome, letters_per_mapping):
    # Split the chromosome into module names based on letters_per_mapping
    mapped_modules = split_string(chromosome, letters_per_mapping)

    # Iterate over each mapped module to place it in shelves
    for module_name in mapped_modules:
        # Find the module with the given name
        module = modules[gene_decoding(module_name)]
        placed = False

        # Try to place module in each shelf in order
        for shelf in shelves:
            if not placed:
                placed = shelf.add_module_to_shelf(module)
                #visualize_row_based_design(shelves)
            else:
                # If no space was found in any shelf
                #print(f"Warning: Could not place module {module.name} due to size constraints.")
                pass
    
    # Return updated modules with placement
    return modules

def pmx_crossover(parent1, parent2):
    size = len(parent1)

    # Step 1: Select two crossover points
    point1, point2 = sorted(random.sample(range(size), 2))

    # Step 2: Initialize children with None to mark empty spots
    child1, child2 = [None] * size, [None] * size

    # Step 3: Copy the crossover segment from each parent to the respective child
    child1[point1:point2 + 1] = parent1[point1:point2 + 1]
    child2[point1:point2 + 1] = parent2[point1:point2 + 1]

    # Step 4: Map and fill in the missing genes
    def fill_child(child, other_parent, point1, point2):
        for i in range(size):
            if child[i] is None:  # Only fill if slot is empty
                gene = other_parent[i]

                # Resolve any conflicts using mapping
                while gene in child[point1:point2 + 1]:  # Check if gene is already in crossover segment
                    gene = other_parent[child[point1:point2 + 1].index(gene) + point1]

                child[i] = gene

    # Fill the rest of the children
    fill_child(child1, parent2, point1, point2)
    fill_child(child2, parent1, point1, point2)

    return child1, child2

def mpmx_crossover(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))
    #print(f"Crossover points: {point1}, {point2}")
    #print(f"Parent1: {parent1}")
    #print(f"Parent2: {parent2}")
    
    child1, child2 = [None] * size, [None] * size

    # Copy the segment from parent1 to child1 and parent2 to child2
    child1[point1:point2 + 1] = parent1[point1:point2 + 1]
    child2[point1:point2 + 1] = parent2[point1:point2 + 1]
    #print(f"Child1 after copying segment: {child1}")
    #print(f"Child2 after copying segment: {child2}")

    def resolve_mapping(child, parent, segment):
        seen_genes = set(child[segment[0]:segment[1] + 1])  # Genes already in the segment
        for i in range(len(child)):
            if child[i] is None:
                gene = parent[i]
                visited = set()  # Track visited genes for the current resolution
                while gene in seen_genes:
                    if gene in visited:
                        # Break infinite loop by assigning a random unused gene
                        unused_genes = set(range(len(parent))) - set(child) - visited
                        if unused_genes:
                            gene = unused_genes.pop()
                        else:
                            raise ValueError("Unable to resolve mapping due to cyclic dependencies.")
                        break
                    visited.add(gene)
                    gene = parent[parent.index(gene)]
                child[i] = gene
                seen_genes.add(gene)



    resolve_mapping(child1, parent2, (point1, point2))
    resolve_mapping(child2, parent1, (point1, point2))

    #print(f"Final Child1: {child1}")
    #print(f"Final Child2: {child2}")
    return child1, child2

def csex_crossover(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))
    subtour = parent1[point1:point2 + 1]

    child1 = [None] * size
    child1[point1:point2 + 1] = subtour

    def fill_remaining(child, parent):
        index = 0
        for gene in parent:
            if gene not in child:
                while child[index] is not None:
                    index += 1
                child[index] = gene

    child2 = [None] * size
    child2[point1:point2 + 1] = parent2[point1:point2 + 1]

    fill_remaining(child1, parent2)
    fill_remaining(child2, parent1)

    return child1, child2

def px_crossover(parent1, parent2):
    size = len(parent1)
    selected_positions = sorted(random.sample(range(size), size // 2))  # Select half positions randomly

    child1, child2 = [None] * size, [None] * size

    for pos in selected_positions:
        child1[pos] = parent1[pos]
        child2[pos] = parent2[pos]

    def fill_child(child, parent):
        index = 0
        for gene in parent:
            if gene not in child:
                while child[index] is not None:
                    index += 1
                child[index] = gene

    fill_child(child1, parent2)
    fill_child(child2, parent1)

    return child1, child2

def split_string(s, length):
    parts = []
    for i in range(0, len(s), length):
        parts.append(s[i:i+length])
    return parts

def visualize_row_based_design(shelves, xOp, rail=None, terminals=None):
    fig, ax = plt.subplots()

    # Draw shelves and modules
    for shelf_index, shelf in enumerate(shelves):
        y_offset = sum([s.height for s in shelves[:shelf_index]])
        for module in shelf.contained_modules:
            y = y_offset + (shelf.height - module.height) / 2
            color = 'lightblue' if module.terminal else 'gray'
            rect = patches.Rectangle(
                (module.x, y), module.width, module.height,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.8
            )
            ax.add_patch(rect)

        # Draw shelf borders
        ax.plot([0, shelf.max_width], [y_offset, y_offset], color='black', linewidth=1)  # Top border
        ax.plot([0, shelf.max_width], [y_offset + shelf.height, y_offset + shelf.height], color='black', linewidth=1)  # Bottom border

    # Draw rails
    if rail:
        core_width = max([s.max_width for s in shelves])
        core_height = sum([s.height for s in shelves])

        # Draw continuous rail strips
        rail_rects = [
            (0, -rail.buffer_distance - 1, core_width, 1),  # Bottom rail
            (core_width + rail.buffer_distance, 0, 1, core_height),  # Right rail
            (0, core_height + rail.buffer_distance, core_width, 1),  # Top rail
            (-rail.buffer_distance - 1, 0, 1, core_height)  # Left rail
        ]
        for x, y, w, h in rail_rects:
            rect = patches.Rectangle((x, y), w, h, linewidth=0.5, edgecolor='black', facecolor='lightgray', alpha=0.5)
            ax.add_patch(rect)

        # Draw terminals on rails
        if terminals:
            for terminal in terminals:
                rect = patches.Rectangle(
                    (terminal.x, terminal.y), terminal.width, terminal.height,
                    linewidth=1, edgecolor='black', facecolor='lightblue', alpha=0.8
                )
                ax.add_patch(rect)

    # Draw boundary around the core
    core_width = max([s.max_width for s in shelves])
    core_height = sum([s.height for s in shelves])
    core_rect = patches.Rectangle(
        (0, 0), core_width, core_height,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(core_rect)

    # Set plot limits and grid
    buffer_distance = rail.buffer_distance if rail else 0
    ax.set_xlim(-buffer_distance - 2, core_width + buffer_distance + 2)
    ax.set_ylim(-buffer_distance - 2, core_height + buffer_distance + 2)
    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.title(f"Row-Based Placement Visualization ({xOp})")
    plt.grid(visible=True, which='both', color='lightgray', linestyle='--', linewidth=0.5)
    plt.show()
   
def gene_decoding(s):
    base = 26
    result = 0
    for char in s:
        result = result * base + (ord(char) - ord('A'))
    return result

def worker_task(thread_id, modules, shelves, chromosome, nets):
    """
    Worker function for fitness evaluation in each thread.
    """
    # Create a deep copy of shelves for this thread
    local_shelves = deepcopy(shelves)
    
    # Create an individual and evaluate placement
    individual = create_individual(modules, local_shelves, chromosome)
    if individual is None:
        #print(f"Thread-{thread_id}: Placement failed.\n")
        return float('inf')  # Penalize if placement failed
    
    # Calculate HPWL for this placement
    hpwl = calculate_hpwl(individual, nets)
    #print(f"Thread-{thread_id}: HPWL = {hpwl}\n")
    return hpwl

def parallel_fitness_evaluation(population, modules, shelves, nets, max_threads=4):
    fitness_scores = []
    with ThreadPoolExecutor(max_threads) as executor:
        futures = [
            executor.submit(fitness, chromosome, nets, shelves)
            for chromosome in population
        ]
        fitness_scores = [future.result(timeout=30) for future in futures]
    return fitness_scores

def genetic_algorithm_pmx(modules, shelves, nets):
    """
    Main genetic algorithm with multithreaded fitness evaluation.
    """
    
    # Initialize population
    population = [random.sample(range(len(modules)), len(modules)) for _ in range(POPULATION_SIZE)]
    best_fitness_per_generation = []

    for generation in range(GENERATIONS):
        print(f"========================== Generation {generation} (PMX) ==========================\n")
        
        # Parallel fitness evaluation
        fitness_values = parallel_fitness_evaluation(population, modules, shelves, nets, max_threads=MAX_THREADS)
        #print(f"Fitness Values Before sort (Generation {generation}): {fitness_values}")
        # Sort population by fitness
        population = [chrom for _, chrom in sorted(zip(fitness_values, population))]
        fitness_values.sort()

        #print(f"Fitness Values After sort (Generation {generation}): {fitness_values}")
        
        # Maintain elite individuals
        elite_size = int(ELITISM_RATE * POPULATION_SIZE)
        next_generation = population[:elite_size]

        # Generate new individuals
        while len(next_generation) < POPULATION_SIZE:
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)
            child1, child2 = pmx_crossover(parent1, parent2)
            
            if random.random() < MUTATION_RATE:
                child1 = mutate(child1)
            if random.random() < MUTATION_RATE:
                child2 = mutate(child2)
            
            next_generation.append(child1)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(child2)

        # Track the best fitness
        best_fitness = fitness_values[0]
        best_fitness_per_generation.append(best_fitness)
        print(f"Generation {generation}, Best Fitness (HPWL): {best_fitness}\n Pop Size: {len(population)}")
        #print(f"best individual fit: {fitness(population[0], modules, nets, deepcopy(shelves), rail, core_width, core_height)}\t Best fitness value: {min(fitness_values)}")
        # Track best individual explicitly
        best_individual = population[0]
        best_fitness = fitness(best_individual, nets, deepcopy(shelves))
        #best_fitness_recalc = calculate_hpwl(create_individual(modules, shelves, best_individual) + terminals, nets)
        #min_fit_index_chrom_fitness = fitness(population[fitness_values.index(min(fitness_values))], nets, deepcopy(shelves))
        #print(f"Best individual: {best_individual}")
        #print(f"Best fitness value: {best_fitness}")
        #print(f"Minimum fitness value in the generation: {min(fitness_values)}\t Minimum fitness value index: {fitness_values.index(min(fitness_values))}")
        #print(f"min_fit_index_chrom_fitness value: {min_fit_index_chrom_fitness}")
        
        #print(f"Recalculated fitness value: {best_fitness_recalc}")

        # Update population
        population = next_generation
        
    return create_individual(modules, shelves, population[0]), best_fitness_per_generation

def genetic_algorithm_mpmx(modules, shelves, nets):

    population = [random.sample(range(len(modules)), len(modules)) for _ in range(POPULATION_SIZE)]
    best_fitness_per_generation = []

    for generation in range(GENERATIONS):
        print(f"========================== Generation {generation} (MPMX) ==========================\n")
        
        fitness_values = parallel_fitness_evaluation(
        population, modules, shelves, nets, max_threads=MAX_THREADS
        )
        #print(f"Fitness Values (Generation {generation}): {fitness_values}")

        population = [chrom for _, chrom in sorted(zip(fitness_values, population))]

        elite_size = int(ELITISM_RATE * POPULATION_SIZE)
        next_generation = population[:elite_size]

        while len(next_generation) < POPULATION_SIZE:
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)
            child1, child2 = mpmx_crossover(parent1, parent2)

            if random.random() < MUTATION_RATE:
                child1 = mutate(child1)
            if random.random() < MUTATION_RATE:
                child2 = mutate(child2)

            next_generation.append(child1)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(child2)

        population = next_generation
        best_fitness = fitness(population[0], nets, deepcopy(shelves))
        best_fitness_per_generation.append(best_fitness)
        print(f"Generation {generation}, Best Fitness (HPWL): {best_fitness}")

    return create_individual(modules, shelves, population[0]), best_fitness_per_generation

def genetic_algorithm_csex(modules, shelves, nets):

    population = [random.sample(range(len(modules)), len(modules)) for _ in range(POPULATION_SIZE)]
    best_fitness_per_generation = []

    for generation in range(GENERATIONS):
        print(f"========================== Generation {generation} (CSEX) ==========================\n")
        
        fitness_values = parallel_fitness_evaluation(
        population, modules, shelves, nets, max_threads=MAX_THREADS
        )
        population = [chrom for _, chrom in sorted(zip(fitness_values, population))]

        elite_size = int(ELITISM_RATE * POPULATION_SIZE)
        next_generation = population[:elite_size]

        while len(next_generation) < POPULATION_SIZE:
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)
            child1, child2 = csex_crossover(parent1, parent2)

            if random.random() < MUTATION_RATE:
                child1 = mutate(child1)
            if random.random() < MUTATION_RATE:
                child2 = mutate(child2)

            next_generation.append(child1)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(child2)

        population = next_generation
        best_fitness = fitness(population[0], nets, deepcopy(shelves))
        best_fitness_per_generation.append(best_fitness)
        print(f"Generation {generation}, Best Fitness (HPWL): {best_fitness}")

    return create_individual(modules, shelves, population[0]), best_fitness_per_generation

def genetic_algorithm_px(modules, shelves, nets):

    population = [random.sample(range(len(modules)), len(modules)) for _ in range(POPULATION_SIZE)]
    best_fitness_per_generation = []

    for generation in range(GENERATIONS):
        print(f"========================== Generation {generation} (PX) ==========================\n")
        
        fitness_values = parallel_fitness_evaluation(
        population, modules, shelves, nets, max_threads=MAX_THREADS
        )
        population = [chrom for _, chrom in sorted(zip(fitness_values, population))]

        elite_size = int(ELITISM_RATE * POPULATION_SIZE)
        next_generation = population[:elite_size]

        while len(next_generation) < POPULATION_SIZE:
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)
            child1, child2 = px_crossover(parent1, parent2)

            if random.random() < MUTATION_RATE:
                child1 = mutate(child1)
            if random.random() < MUTATION_RATE:
                child2 = mutate(child2)

            next_generation.append(child1)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(child2)

        population = next_generation
        best_fitness = fitness(population[0], nets, deepcopy(shelves))
        best_fitness_per_generation.append(best_fitness)
        print(f"Generation {generation}, Best Fitness (HPWL): {best_fitness}")

    return create_individual(modules, shelves, population[0]), best_fitness_per_generation

def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(1.0 / f for f in fitness_values if f != float('inf') and f != 0)
    if total_fitness == 0:
        return random.choice(population)  # Fallback in case of all invalid fitness values

    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitness_values):
        if fitness != float('inf'):
            current += 1.0 / fitness
            if current > pick:
                return individual
    return random.choice(population)  # Fallback in case of rounding issues

def save_results_to_excel(filename, circuit, xOp, fitness_progress, placement_time, fit_prog_list):
    if not LOG_RESULTS:
        return None
    
    df = pd.DataFrame({
        'Circuit': [circuit],
        '# Of Modules': [len(modules)],
        '# Of Nets': [len(nets)],
        'Generations': [GENERATIONS],
        'Initial Population': [POPULATION_SIZE],
        'Initial HPWL': [initial_HPWL],
        'Final HPWL': [fitness_progress],
        'Fitness Progress': [fit_prog_list],
        'Improvement Percentage': [(initial_HPWL - fitness_progress) / initial_HPWL * 100],
        'Elitism': [ELITISM_RATE],
        'Mutation Rate': [MUTATION_RATE],
        'Placement Time(s)': [placement_time],
        'CrossOver Operator': [xOp]
    })
    
    # Ensure the 'results' directory exists
    os.makedirs('results', exist_ok=True)
    
    file_path = f'results/{filename}.xlsx'
    
    if os.path.isfile(file_path):
        # If file exists, append data to the existing sheet
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # Find the starting row by getting the existing data length
            try:
                existing_data = pd.read_excel(file_path)
                startrow = len(existing_data) + 1
            except Exception as e:
                # If the file is corrupted or cannot be read, overwrite it
                print(f"Error reading {file_path}, overwriting: {e}")
                startrow = None
            df.to_excel(writer, index=False, header=False if startrow else True, startrow=startrow)
    else:
        # Create a new file
        df.to_excel(file_path, index=False)

def join_excel_files():
    if not LOG_RESULTS and not COMBINE_RESULTS:
        return None
    
    # Join all Excel files in the results folder into a single file
    files = [f for f in os.listdir('results') if f.endswith('.xlsx')]
    combined = pd.concat([pd.read_excel(f"results/{f}") for f in files], ignore_index=True)
    # Save the combined results to a new file with the current timestamp
    #combined.to_excel(f"results/{}_combined_results.xlsx", index=False)
    combined.to_excel(f"results/{datetime.now().strftime('%Y%m%d_%H%M%S')}_combined_results.xlsx", index=False)

# Enqueue visualization requests
def queue_visualization(shelves, xOp):
    visualization_queue.put((shelves, xOp))

# Dequeue and visualize in the main thread
def process_visualizations():
    while not visualization_queue.empty():
        shelves, xOp = visualization_queue.get()
        visualize_row_based_design(shelves, xOp)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Constants
    LOG_RESULTS = config['OPTIONS'].getboolean('LOG_RESULTS')
    PMX_RUN = config['OPTIONS'].getboolean('PMX_RUN')
    MPMX_RUN = config['OPTIONS'].getboolean('MPMX_RUN')
    CSEX_RUN = config['OPTIONS'].getboolean('CSEX_RUN')
    PX_RUN = config['OPTIONS'].getboolean('PX_RUN')
    POPULATION_SIZE = int(config['OPTIONS']['POPULATION_SIZE'])
    GENERATIONS = int(config['OPTIONS']['GENERATIONS'])
    MUTATION_RATE = float(config['OPTIONS']['MUTATION_RATE'])
    ELITISM_RATE = float(config['OPTIONS']['ELITISM_RATE'])
    MR_CHANGE_RATE = float(config['OPTIONS']['MR_CHANGE_RATE'])
    ELITISM_CHANGE_RATE = float(config['OPTIONS']['ELITISM_CHANGE_RATE'])
    MAX_THREADS = int(config['OPTIONS']['MAX_THREADS'])
    SMALL_DESIGNS_PATH = config['PATHS']['small_designs']
    COMBINE_RESULTS = config['OPTIONS'].getboolean('COMBINE_RESULTS')
    RESULT_VISUALIZATION = config['OPTIONS'].getboolean('RESULT_VISUALIZATION')
    directory_path = Path(SMALL_DESIGNS_PATH)
    # Queue for visualization tasks
    visualization_queue = Queue()

    print(f"#################################### Elitism Rate: {ELITISM_RATE:.2f} ####################################")
    print(f"#################################### Mutation Rate: {MUTATION_RATE:.2f} ####################################")
    for subdir in directory_path.iterdir():
        if subdir.is_dir():
            subdirectory = SMALL_DESIGNS_PATH + subdir.name + "/"
            for file in os.listdir(subdirectory):
                if file.endswith(".aux"):
                    aux_file = file
                elif file.endswith(".nodes"):
                    nodes_file = file
                elif file.endswith(".nets"):
                    nets_file = file
                elif file.endswith(".pl"):
                    pl_file = file
                elif file.endswith(".scl"):
                    shelves_file = file
            
            # Parsing files
            parsing_start_time = time.time()
            filenames = filenames_parser(subdirectory + aux_file)
            modules = parse_nodes(subdirectory + nodes_file)
            shelves = parse_shelves(subdirectory + shelves_file)
            nets = parse_custom_nets(subdirectory + nets_file)
            parsing_end_time = time.time()
            
            # Create rail based on core dimensions and buffer distance
            buffer_distance = int(config['OPTIONS']['BUFFER_DISTANCE'])
            core_width = max([s.max_width for s in shelves])
            core_height = sum([s.height for s in shelves])
            rail = Rail(core_width, core_height, buffer_distance)
            # Separate terminal nodes from regular modules
            terminals = [m for m in modules if m.terminal]
            non_terminals = [m for m in modules if not m.terminal]
            place_terminals_on_rails(terminals, rail, modules, nets, core_width, core_height)
            #visualize_row_based_design(shelves, "", rail=rail, terminals=terminals)
            initial_HPWL = starting_placement(non_terminals, shelves)
            print(f"Initial HPWL: {initial_HPWL}")
            visualize_row_based_design(shelves, 'Initial Placement', rail=rail, terminals=terminals)
            # Run genetic algorithms
            
            #=============================================PMX==========================================================            
            if PMX_RUN:
                # Starting placement                
                pmx_placement_start_time = time.time()
                #mapping, letters_per_mapping = generate_string_mapping(modules_removed)
                best_solution_pmx, fitness_progress_pmx = genetic_algorithm_pmx(non_terminals, shelves, nets)
                #visualize_row_based_design(shelves, 'PMX', rail=rail, terminals=terminals)
                pmx_placement_end_time = time.time()
                
                filename = (subdir.name + "_MR" +
                            str(config['OPTIONS']['MUTATION_RATE']) +
                            "_ER" + str(config['OPTIONS']['ELITISM_RATE']) +
                            "_POP" + str(config['OPTIONS']['POPULATION_SIZE']) +
                            "_GEN" + str(config['OPTIONS']['GENERATIONS']) + "_PMX")
                
                # Save results to Excel
                save_results_to_excel(filename, subdir.name, 'PMX', fitness_progress_pmx[-1], pmx_placement_end_time - pmx_placement_start_time, fitness_progress_pmx)
                
                # Timing stats
                # print(f"Parsing runtime: {parsing_end_time - parsing_start_time:.2f}s")
                # print(f"PMX Placement runtime: {placement_end_time - placement_start_time:.2f}s"
                if RESULT_VISUALIZATION:
                    # Visualization
                    visualize_row_based_design(shelves, 'PMX', rail=rail, terminals=terminals)
                    plot_fitness_progress(fitness_progress_pmx, "PMX")
                    #queue_visualization(shelves, 'PMX')
            else:
                print("PMX is not enabled.")
            
            #=============================================MPMX==========================================================
            if MPMX_RUN:
                mpmx_placement_start_time = time.time()
                #mapping, letters_per_mapping = generate_string_mapping(modules_removed)
                best_solution_mpmx, fitness_progress_mpmx = genetic_algorithm_mpmx(non_terminals, shelves, nets)
                mpmx_placement_end_time = time.time()
                
                filename = (subdir.name + "_MR" +
                            str(config['OPTIONS']['MUTATION_RATE']) +
                            "_ER" + str(config['OPTIONS']['ELITISM_RATE']) +
                            "_POP" + str(config['OPTIONS']['POPULATION_SIZE']) +
                            "_GEN" + str(config['OPTIONS']['GENERATIONS']) + "_MPMX")
                
                # Save results to Excel
                save_results_to_excel(filename, subdir.name, 'MPMX', fitness_progress_mpmx[-1], mpmx_placement_end_time - mpmx_placement_start_time, fitness_progress_mpmx)
                
                # Timing stats
                #print(f"MPMX Placement runtime: {placement_end_time - placement_start_time:.2f}s"
                if RESULT_VISUALIZATION:
                    # # Visualization
                    visualize_row_based_design(shelves, 'MPMX', rail=rail, terminals=terminals)
                    plot_fitness_progress(fitness_progress_mpmx, "MPMX")
                    # queue_visualization(shelves, 'MPMX')
            else:
                print("MPMX is not enabled.")
        
            #=============================================CSEX==========================================================
            if CSEX_RUN:
                csex_placement_start_time = time.time()
                #mapping, letters_per_mapping = generate_string_mapping(modules_removed)
                best_solution_csex, fitness_progress_csex = genetic_algorithm_csex(non_terminals, shelves, nets)
                csex_placement_end_time = time.time()
                
                filename = (subdir.name + "_MR" +
                            str(config['OPTIONS']['MUTATION_RATE']) +
                            "_ER" + str(config['OPTIONS']['ELITISM_RATE']) +
                            "_POP" + str(config['OPTIONS']['POPULATION_SIZE']) +
                            "_GEN" + str(config['OPTIONS']['GENERATIONS']) + "_CSEX")
                
                # Save results to Excel
                save_results_to_excel(filename, subdir.name, 'CSEX', fitness_progress_csex[-1], csex_placement_end_time - csex_placement_start_time, fitness_progress_csex)
                
                # Timing stats
                #print(f"CSEX Placement runtime: {placement_end_time - placement_start_time:.2f}s"
                if RESULT_VISUALIZATION:
                    # Visualization
                    visualize_row_based_design(shelves, 'CSEX', rail=rail, terminals=terminals)
                    plot_fitness_progress(fitness_progress_csex, "CSEX")
                    #queue_visualization(shelves, 'CSEX')
            else:
                print("CSEX is not enabled.")
            
            #=============================================PX==========================================================            
            if PX_RUN:
                px_placement_start_time = time.time()
                #mapping, letters_per_mapping = generate_string_mapping(modules_removed)
                best_solution_px, fitness_progress_px = genetic_algorithm_px(non_terminals, shelves, nets)
                px_placement_end_time = time.time()
                
                filename = (subdir.name + "_MR" +
                            str(config['OPTIONS']['MUTATION_RATE']) +
                            "_ER" + str(config['OPTIONS']['ELITISM_RATE']) +
                            "_POP" + str(config['OPTIONS']['POPULATION_SIZE']) +
                            "_GEN" + str(config['OPTIONS']['GENERATIONS']) + "_PX")
                
                # Save results to Excel
                save_results_to_excel(filename, subdir.name, 'PX', fitness_progress_px[-1], px_placement_end_time - px_placement_start_time, fitness_progress_px)
                
                if RESULT_VISUALIZATION:
                    visualize_row_based_design(shelves, 'PX', rail=rail, terminals=terminals)
                    plot_fitness_progress(fitness_progress_px, "PX")
                # Timing stats
                #print(f"PX Placement runtime: {placement_end_time - placement_start_time:.2f}s")
            else:
                print("PX is not enabled.")
            
            print(f"========================== {subdir.name} ==========================\n")
            print(f"Parsing runtime: {parsing_end_time - parsing_start_time:.2f}s")
            if PMX_RUN: print(f"PMX Placement runtime: {pmx_placement_end_time - pmx_placement_start_time:.2f}s") 
            else: None
            if MPMX_RUN: print(f"MPMX Placement runtime: {mpmx_placement_end_time - mpmx_placement_start_time:.2f}s")
            else: None
            if CSEX_RUN: print(f"CSEX Placement runtime: {csex_placement_end_time - csex_placement_start_time:.2f}s")  
            else: None
            if PX_RUN: print(f"PX Placement runtime: {px_placement_end_time - px_placement_start_time:.2f}s")
            else: None                    
            
            # # Visualization
            
            # #queue_visualization(shelves, 'PX')
            # plot_fitness_progress(fitness_progress_pmx, "PMX")
            # plot_fitness_progress(fitness_progress_csex, "CSEX")
            # #plot_fitness_progress(fitness_progress_mpmx, "MPMX")
            # plot_fitness_progress(fitness_progress_px, "PX")
    
    # After the algorithm is done
    #process_visualizations()
    join_excel_files()
