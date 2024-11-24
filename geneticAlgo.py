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
            print("Warning: Module not inserted at correct index!")
        
    def check_module_overlap(self):
        # Check if any modules in the shelf overlap with each other
        previous_x = 0
        for module in self.contained_modules:
            if module.x < previous_x:
                print(f"Warning: Module {module.name} overlaps with previous module(s) in the shelf!\n")
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


    def __repr__(self):
        return f"Shelf(height={self.height}, max_width={self.max_width}, used_width={self.used_width})"

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
def fitness(chromosome, modules, nets, shelves):
    # Reset shelves to start fresh for each chromosome evaluation
    reset_shelves(shelves)
    individual = create_individual(modules, shelves, chromosome)
    if individual is None:
        return float('inf')  # Penalize if placement failed
    # Calculate HPWL for this placement
    hpwl = calculate_hpwl(individual, nets)
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
    modules_copy = deepcopy(modules)
    reset_shelves(shelves)
    individual = []

    for module_index in chromosome:
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
    
    return calculate_hpwl(modules, nets)

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

def visualize_row_based_design(shelves, xOp):
    fig, ax = plt.subplots()

    for shelf_index, shelf in enumerate(shelves):
        y_offset = sum([s.height for s in shelves[:shelf_index]])  # Calculate y-position based on previous shelves
        for module in shelf.contained_modules:
            # Draw a rectangle for each module. If it's a terminal, put the rectangle on the y axis border of the shelf
            y = y_offset if module.terminal else y_offset + (shelf.height - module.height) / 2
            rect = patches.Rectangle((module.x, y), module.width, module.height, linewidth=1, edgecolor='#000000', facecolor='#c4c4c4', alpha=1)
            # Add the name of the module in the rectangle and make it so that the name scales with the size of the rectangle
            #ax.text(module.x + module.width / 2, y + module.height / 2, module.name, ha='center', va='center', fontsize=8)
            
            ax.add_patch(rect)
            
            # rect = patches.Rectangle((module.x, y_offset), module.width, module.height, linewidth=1, edgecolor='#000000', facecolor='#c4c4c4', alpha=1)
            # ax.add_patch(rect)

    # Set the limits and display the grid
    ax.set_xlim(0, max([s.max_width for s in shelves]))
    ax.set_ylim(0, sum([s.height for s in shelves]))
    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Invert Y axis to match row layout
    # Display the row names on the left side of the plot with the number of the row (len of shelves - index) 
    for i, shelf in enumerate(shelves):
        plt.text(-50, sum([s.height for s in shelves[:i]]) + shelf.height / 2, f"Row {len(shelves) - i}", ha='center', va='center')
    
    #show a line at the end of each row
    for i, shelf in enumerate(shelves):
        plt.plot([0, shelf.max_width], [sum([s.height for s in shelves[:i]]), sum([s.height for s in shelves[:i]])], color='black', linewidth=1)
    
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    plt.title(f"Row-Based Placement Visualization ({xOp})")
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
    """
    Evaluates fitness for the entire population using multithreading.
    """
    fitness_scores = []
    with ThreadPoolExecutor(max_threads) as executor:
        # Submit a fitness evaluation task for each chromosome
        futures = [
            executor.submit(worker_task, i, modules, shelves, chromosome, nets)
            for i, chromosome in enumerate(population)
        ]
        # Collect results as threads complete
        try:
            fitness_scores = [future.result(timeout=30) for future in futures]  # Add a timeout
        except Exception as e:
            print(f"Thread execution error: {e}")
            fitness_scores = [float('inf')] * len(population)
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
        
        # Sort population by fitness
        population = [chrom for _, chrom in sorted(zip(fitness_values, population))]

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

        # Update population
        population = next_generation

        # Track the best fitness
        best_fitness = fitness(population[0], modules, nets, deepcopy(shelves))
        best_fitness_per_generation.append(best_fitness)
        print(f"Generation {generation}, Best Fitness (HPWL): {best_fitness}")

    return create_individual(modules, shelves, population[0]), best_fitness_per_generation

def genetic_algorithm_mpmx(modules, shelves, nets):

    population = [random.sample(range(len(modules)), len(modules)) for _ in range(POPULATION_SIZE)]
    best_fitness_per_generation = []

    for generation in range(GENERATIONS):
        print(f"========================== Generation {generation} (MPMX) ==========================\n")
        
        fitness_values = parallel_fitness_evaluation(population, modules, shelves, nets, max_threads=MAX_THREADS)
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
        best_fitness = fitness(population[0], modules, nets, deepcopy(shelves))
        best_fitness_per_generation.append(best_fitness)
        print(f"Generation {generation}, Best Fitness (HPWL): {best_fitness}")

    return create_individual(modules, shelves, population[0]), best_fitness_per_generation

def genetic_algorithm_csex(modules, shelves, nets):

    population = [random.sample(range(len(modules)), len(modules)) for _ in range(POPULATION_SIZE)]
    best_fitness_per_generation = []

    for generation in range(GENERATIONS):
        print(f"========================== Generation {generation} (CSEX) ==========================\n")
        
        fitness_values = parallel_fitness_evaluation(population, modules, shelves, nets, max_threads=MAX_THREADS)
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
        best_fitness = fitness(population[0], modules, nets, deepcopy(shelves))
        best_fitness_per_generation.append(best_fitness)
        print(f"Generation {generation}, Best Fitness (HPWL): {best_fitness}")

    return create_individual(modules, shelves, population[0]), best_fitness_per_generation

def genetic_algorithm_px(modules, shelves, nets):

    population = [random.sample(range(len(modules)), len(modules)) for _ in range(POPULATION_SIZE)]
    best_fitness_per_generation = []

    for generation in range(GENERATIONS):
        print(f"========================== Generation {generation} (PX) ==========================\n")
        
        fitness_values = parallel_fitness_evaluation(population, modules, shelves, nets, max_threads=MAX_THREADS)
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
        best_fitness = fitness(population[0], modules, nets, deepcopy(shelves))
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
            
            #=============================================PMX==========================================================
            
            if PMX_RUN:
                # Starting placement
                initial_HPWL = starting_placement(modules, shelves)
                pmx_placement_start_time = time.time()
                #mapping, letters_per_mapping = generate_string_mapping(modules_removed)
                best_solution_pmx, fitness_progress_pmx = genetic_algorithm_pmx(modules, shelves, nets)
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
                # Visualization
                #visualize_row_based_design(shelves, 'PMX')
                #plot_fitness_progress(fitness_progress_pmx, "PMX")
                #queue_visualization(shelves, 'PMX')
            else:
                print("PMX is not enabled.")
            
            #=============================================MPMX==========================================================
            if MPMX_RUN:
                mpmx_placement_start_time = time.time()
                #mapping, letters_per_mapping = generate_string_mapping(modules_removed)
                best_solution_mpmx, fitness_progress_mpmx = genetic_algorithm_mpmx(modules, shelves, nets)
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
                # # Visualization
                # #visualize_row_based_design(shelves, 'MPMX')
                # #plot_fitness_progress(fitness_progress_mpmx)
                # queue_visualization(shelves, 'MPMX')
            else:
                print("MPMX is not enabled.")
        
            #=============================================CSEX==========================================================
            if CSEX_RUN:
                csex_placement_start_time = time.time()
                #mapping, letters_per_mapping = generate_string_mapping(modules_removed)
                best_solution_csex, fitness_progress_csex = genetic_algorithm_csex(modules, shelves, nets)
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
                # Visualization
                #visualize_row_based_design(shelves, 'CSEX')
                #plot_fitness_progress(fitness_progress_csex, "CSEX")
                #queue_visualization(shelves, 'CSEX')
            else:
                print("CSEX is not enabled.")
            
            #=============================================PX==========================================================            
            if PX_RUN:
                px_placement_start_time = time.time()
                #mapping, letters_per_mapping = generate_string_mapping(modules_removed)
                best_solution_px, fitness_progress_px = genetic_algorithm_px(modules, shelves, nets)
                px_placement_end_time = time.time()
                
                filename = (subdir.name + "_MR" +
                            str(config['OPTIONS']['MUTATION_RATE']) +
                            "_ER" + str(config['OPTIONS']['ELITISM_RATE']) +
                            "_POP" + str(config['OPTIONS']['POPULATION_SIZE']) +
                            "_GEN" + str(config['OPTIONS']['GENERATIONS']) + "_PX")
                
                # Save results to Excel
                save_results_to_excel(filename, subdir.name, 'PX', fitness_progress_px[-1], px_placement_end_time - px_placement_start_time, fitness_progress_px)
                
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
            # visualize_row_based_design(shelves, 'PX')
            # # plot_fitness_progress(fitness_progress_px)
            # #queue_visualization(shelves, 'PX')
            # plot_fitness_progress(fitness_progress_pmx, "PMX")
            # plot_fitness_progress(fitness_progress_csex, "CSEX")
            # #plot_fitness_progress(fitness_progress_mpmx, "MPMX")
            # plot_fitness_progress(fitness_progress_px, "PX")
    
    # After the algorithm is done
    #process_visualizations()
    join_excel_files()
