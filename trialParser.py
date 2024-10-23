import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from geneticAlgo import GeneticAlgorithmPlacer

# Define global variables for colors
edgeColor = "#A9A9A9"
faceColor = "#D3D3D3"

class IBMBookshelfParser:
    def __init__(self, aux_file, subdir):
        self.aux_file = aux_file
        self.subdir = subdir
        self.nodes_file = None
        self.nets_file = None
        self.wts_file = None  # Not used, but included in case needed later
        self.pl_file = None
        self.scl_file = None  # Will be used for row data
        self.nodes = {}
        self.placements = {}
        self.nets = []
        self.rows = []  # Store row data (start_y, height)

    def parse_aux(self):
        """Parse the .aux file to extract the file names."""
        with open(os.path.join(self.subdir, self.aux_file), 'r') as f:
            line = f.readline().strip()

        # Match the specific format in the .aux file
        match = re.match(r'RowBasedPlacement\s*:\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', line)
        if match:
            self.nodes_file = os.path.join(self.subdir, match.group(1))
            self.nets_file = os.path.join(self.subdir, match.group(2))
            self.wts_file = os.path.join(self.subdir, match.group(3))  # Placeholder, not used
            self.pl_file = os.path.join(self.subdir, match.group(4))
            self.scl_file = os.path.join(self.subdir, match.group(5))  # Used for row data
        else:
            raise ValueError("Invalid format in .aux file")

    def parse_nodes(self):
        """Parse the .nodes file to extract cell names, dimensions, and whether they are fixed."""
        with open(self.nodes_file, 'r') as f:
            lines = f.readlines()

        start_parsing = False
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue

            # Skip the header, start parsing after 'NumNodes' line
            if "NumNodes" in line:
                start_parsing = True
                continue

            if start_parsing:
                match = re.match(r'(\w+)\s+(\d+)\s+(\d+)(\s+fixed)?', line)
                if match:
                    node_name = match.group(1)
                    width = int(match.group(2))
                    height = int(match.group(3))
                    fixed = bool(match.group(4))

                    self.nodes[node_name] = {
                        'width': width,
                        'height': height,
                        'fixed': fixed
                    }

    def parse_pl(self):
        """Parse the .pl file to extract cell placement coordinates and orientation."""
        with open(self.pl_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue

            # Match the placement pattern
            match = re.match(r'(\w+)\s+(\d+)\s+(\d+)\s*:\s*(\w+)', line)
            if match:
                node_name = match.group(1)
                x_coord = int(match.group(2))
                y_coord = int(match.group(3))
                orientation = match.group(4)

                self.placements[node_name] = {
                    'x': x_coord,
                    'y': y_coord,
                    'orientation': orientation
                }

    def parse_nets(self):
        """Parse the .nets file to extract net connectivity."""
        with open(self.nets_file, 'r') as f:
            lines = f.readlines()

        start_parsing = False
        current_net = None  # Initialize the current net to None

        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue

            # Start parsing after 'NumNets' line
            if "NumNets" in line:
                start_parsing = True
                continue

            if start_parsing:
                # Handle "NetDegree" line, which indicates the start of a new net
                if line.startswith("NetDegree"):
                    # If we already have a current net, append it to the nets list
                    if current_net:
                        self.nets.append(current_net)

                    # Initialize a new net
                    current_net = {'nodes': []}
                    continue

                # Ensure we are inside a net before trying to append nodes
                if current_net is not None:
                    # Match node lines in the net
                    match = re.match(r'(\w+)\s*(\w*)', line)
                    if match:
                        node_name = match.group(1)
                        current_net['nodes'].append(node_name)

        # After the loop, append the last net if it's not already added
        if current_net:
            self.nets.append(current_net)

    def parse_scl(self):
        """Parse the .scl file to extract row information."""
        with open(self.scl_file, 'r') as f:
            lines = f.readlines()

        current_row_start = None
        current_row_height = None

        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            # Check for CoreRow Horizontal line
            if "CoreRow Horizontal" in line:
                current_row_start = None
                current_row_height = None

            # Check for Coordinate line
            if "Coordinate" in line:
                try:
                    current_row_start = int(line.split(':')[1].strip())
                except (IndexError, ValueError) as e:
                    print(f"Error parsing Coordinate line: {line} - {e}")
                    continue  # Skip this line if there's an error

            # Check for Height line
            if "Height" in line:
                try:
                    current_row_height = int(line.split(':')[1].strip())
                except (IndexError, ValueError) as e:
                    print(f"Error parsing Height line: {line} - {e}")
                    continue  # Skip this line if there's an error

            # Check for End line to finalize the current row
            if current_row_start is not None and current_row_height is not None and "End" in line:
                row_end = current_row_start + current_row_height
                self.rows.append((current_row_start, row_end, current_row_height))

                # Reset current_row variables
                current_row_start = None
                current_row_height = None

    def parse_all(self):
        """Parse the aux file and all the required files."""
        self.parse_aux()  # This will set the file names
        self.parse_nodes()
        self.parse_pl()
        self.parse_nets()
        self.parse_scl()

    def get_nodes(self):
        return self.nodes

    def get_placements(self):
        return self.placements

    def get_nets(self):
        return self.nets

    def get_rows(self):
        return self.rows

    def plot_row_based_design(self, placements):
        # Calculate the maximum x and y coordinates
        max_x = max((placement['x'] + self.nodes[node]['width'] for node, placement in placements.items()), default=0)
        max_y = max((placement['y'] + self.nodes[node]['height'] for node, placement in placements.items()), default=0)

        # Set the figure size based on the maximum coordinates
        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw the chip boundary
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)
        ax.set_aspect('equal', 'box')
        ax.set_title('Row-Based Design Visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Plot the rows as horizontal bars
        for i, (row_start, row_end, row_height) in enumerate(self.rows):
            ax.add_patch(patches.Rectangle((0, row_start), max_x, row_height, 
                                           linewidth=1, edgecolor='black', facecolor='none'))
            ax.text(-10, row_start + row_height / 2, f'Row {i}', ha='right', va='center')

        # Plot nodes (cells) and terminals in rows
        for node_name, node_data in self.nodes.items():
            width = node_data['width']
            height = node_data['height']
            fixed = node_data['fixed']

            # Check if the node has placement information
            if node_name in placements:
                x = placements[node_name]['x']
                y = placements[node_name]['y']

                if fixed:
                    rect = patches.Rectangle((x-0.1, y-0.1), width, height, linewidth=1, edgecolor='r', facecolor='r', label='Terminal')
                    ax.add_patch(rect)
                else:
                    rect = patches.Rectangle((x-0.1, y-0.1), width, height, linewidth=1, edgecolor=edgeColor, facecolor=faceColor, label='Cell')
                    ax.add_patch(rect)
            else:
                print(f"Warning: No placement information for node {node_name}")

        plt.show()

# Example usage:
if __name__ == "__main__":
    subdir = "ibm01"  # Specify the subdirectory containing the bookshelf files
    aux_file = "ibm01.aux"  # The .aux file

    parser = IBMBookshelfParser(aux_file, subdir)
    parser.parse_all()

    # Output parsed data to a text file
    output_file = os.path.join(subdir, "output.txt")
    with open(output_file, 'w') as f:
        # Write Nodes information
        f.write("Nodes:\n")
        for node, data in parser.get_nodes().items():
            f.write(f"{node} {data}\n")

        # Write Placements information
        f.write("\nPlacements:\n")
        for node, placement in parser.get_placements().items():
            f.write(f"{node} {placement}\n")

        # Write Nets information
        f.write("\nNets:\n")
        for net in parser.get_nets():
            f.write(f"{net}\n")

        # Write Row information
        f.write("\nRows:\n")
        for row in parser.get_rows():
            f.write(f"Row: {row}\n")

    print(f"Output written to {output_file}")

    # Run the genetic algorithm to get the best placement
    ga_placer = GeneticAlgorithmPlacer(parser.get_nodes(), parser.get_nets())
    best_placements, best_hwl = ga_placer.run()
    print("Best Placements:", best_placements)
    print("Best HPWL:", best_hwl)

    # Plot the row-based design visualization with the best placements
    parser.plot_row_based_design(best_placements)