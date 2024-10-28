import random
import os
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Module:
    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height
        self.x = 0
        self.y = 0

    def __repr__(self):
        return f"Module(name={self.name}, width={self.width}, height={self.height}, x={self.x}, y={self.y})"

class Shelf:
    def __init__(self, height, max_width):
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
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[0]
                    width = float(parts[1])
                    height = float(parts[2])
                    nodes.append(Module(name, width, height))
    print("Node parsing complete")
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
                shelves.append(Shelf(height, width))
                shelf_parsed += 1
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
                print(f"Warning: Module {module_name} not found in module list!")

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
    # Create individual based on the chromosome order
    individual = create_individual(modules, shelves, chromosome)
    if individual is None:
        return float('inf')  # Penalize if placement failed
    # Calculate the total HPWL for this individual placement
    hpwl = calculate_hpwl(individual, nets)
    return hpwl  # Lower HPWL is better

# Genetic algorithm functions
def create_individual(modules, shelves, chromosome):
    modules_copy = deepcopy(modules)  # Avoid modifying the original modules list
    reset_shelves(shelves)  # Reset shelves before creating a new individual
    individual = []
    current_shelf = 0
    for module_index in chromosome:
        module = modules_copy[module_index]
        placed = False
        # Try placing the module on existing shelves
        while not placed and current_shelf < len(shelves):
            shelf = shelves[current_shelf]
            if shelf.used_width + module.width <= shelf.max_width and module.height <= shelf.height:
                # Place the module
                module.x = shelf.used_width
                module.y = sum(s.height for s in shelves[:current_shelf])
                shelf.used_width += module.width
                shelf.add_module(module)
                individual.append(module)
                #print(f"Placed module {module.name} at ({module.x}, {module.y}) on shelf {current_shelf}") 
                placed = True
            else:
                # Move to the next shelf
                current_shelf += 1
        
        if not placed:
            print(f"Warning: Could not place module {module.name} due to size constraints.")
            return None  # Return None if placement fails

    return individual

def mutate(chromosome):
    # Select two random genes and swap their positions
    idx1, idx2 = random.sample(range(len(chromosome)), 2)
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

def crossover(parent1, parent2):
    # Perform crossover between two parents to create a child
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    return child

def visualize_row_based_design(shelves):
    fig, ax = plt.subplots()

    for shelf_index, shelf in enumerate(shelves):
        y_offset = sum([s.height for s in shelves[:shelf_index]])  # Calculate y-position based on previous shelves
        for module in shelf.contained_modules:
            # Draw a rectangle for each module
            rect = patches.Rectangle((module.x, y_offset), module.width, module.height, linewidth=1, edgecolor='#000000', facecolor='#c4c4c4', alpha=1)
            ax.add_patch(rect)

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
    plt.show()


def genetic_algorithm(modules, shelves, nets, population_size=50, generations=100):
    # Initialize population with random orders
    population = [random.sample(range(len(modules)), len(modules)) for _ in range(population_size)]
    for generation in range(generations):
        # Sort population by HPWL-based fitness
        population = sorted(population, key=lambda chrom: fitness(chrom, modules, nets, shelves))
        # Create next generation
        next_generation = population[:2]  # Elitism: carry over top 2 individuals
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(population[:5], 2)  # Select from top 5 individuals
            child = crossover(parent1, parent2)
            if random.random() < 0.1:  # 10% chance to mutate
                child = mutate(child)
            next_generation.append(child)
        population = next_generation
        # Print the best fitness (HPWL) in each generation
        best_fitness = fitness(population[0], modules, nets, shelves)
        print(f"Generation {generation}, Best Fitness (HPWL): {best_fitness}")
        
        # Visualize the row-based design after each generation
        #visualize_row_based_design(shelves)
        
    return create_individual(modules, shelves, population[0])

def csex_crossover(parent1, parent2):
    """
    Complete Subtour Exchange Crossover (CSEX) for the placement problem.
    This function exchanges subtours between two parent chromosomes to produce offspring.
    """
    size = len(parent1)

    # Select random subtour indices for both parents
    start1, end1 = sorted(random.sample(range(size), 2))
    start2, end2 = sorted(random.sample(range(size), 2))

    # Extract subtours from both parents
    subtour1 = parent1[start1:end1+1]
    subtour2 = parent2[start2:end2+1]

    # Create offspring
    child1 = [None] * size
    child2 = [None] * size

    # Copy subtours to children
    child1[start1:end1+1] = subtour2
    child2[start2:end2+1] = subtour1

    # Fill remaining positions in child1
    pos = 0
    for gene in parent1:
        if gene not in subtour2:
            while pos < size and child1[pos] is not None:
                pos += 1
            if pos < size:
                child1[pos] = gene

    # Fill remaining positions in child2
    pos = 0
    for gene in parent2:
        if gene not in subtour1:
            while pos < size and child2[pos] is not None:
                pos += 1
            if pos < size:
                child2[pos] = gene

    return child1, child2

def genetic_algorithm(modules, shelves, nets, population_size=50, generations=30):
    # Initialize population with random orders
    population = [random.sample(range(len(modules)), len(modules)) for _ in range(population_size)]
    best_fitness_per_generation = []

    for generation in range(generations):
        # Sort population by HPWL-based fitness
        population = sorted(population, key=lambda chrom: fitness(chrom, modules, nets, shelves))

        # Create next generation
        next_generation = population[:2]  # Elitism: carry over top 2 individuals
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(population[:5], 2)  # Select from top 5 individuals
            child1, child2 = csex_crossover(parent1, parent2)  # Use CSEX crossover
            if random.random() < 0.1:  # 10% chance to mutate
                child1 = mutate(child1)
                child2 = mutate(child2)
            next_generation.append(child1)
            if len(next_generation) < population_size:
                next_generation.append(child2)

        population = next_generation

        # Print the best fitness (HPWL) in each generation
        best_fitness = fitness(population[0], modules, nets, shelves)
        best_fitness_per_generation.append(best_fitness)
        print(f"Generation {generation}, Best Fitness (HPWL): {best_fitness}")

        # Visualize the row-based design after each generation
        visualize_row_based_design(shelves)
        time.sleep(0.5)  # Optional: pause for 0.5 seconds to see the update clearly

    return create_individual(modules, shelves, population[0]), best_fitness_per_generation

# Visualization and other helper functions remain the same...

# Paths to the files
subdirectory = "designs/small_designs/b01/"
for file in os.listdir(subdirectory):
    if file.endswith(".aux"):
        aux_file = file

# Parse filenames, modules, shelves, and nets
filenames = filenames_parser(subdirectory + aux_file)  # .nodes, .nets, .wts, .pl, .scl (Always in this order)
modules = parse_nodes(subdirectory + filenames[0])
shelves = parse_shelves(subdirectory + filenames[4])
nets = parse_nets(subdirectory + filenames[1])

# Run the genetic algorithm with CSEX
start_time = time.time()
best_solution, fitness_progress = genetic_algorithm(modules, shelves, nets)
print(f"Execution time: {time.time() - start_time:.2f}s")

# Visualize the row-based design
visualize_row_based_design(shelves)

# Optionally plot the fitness progress
def plot_fitness_progress(fitness_values):
    plt.figure()
    plt.plot(fitness_values, marker='o')
    plt.title("HPWL Fitness Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Best HPWL (Lower is Better)")
    plt.grid(True)
    plt.show()

# Plot the fitness progress
plot_fitness_progress(fitness_progress)

# Output the best solution
for module in best_solution:
    print(module)
