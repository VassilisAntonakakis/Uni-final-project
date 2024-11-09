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
                print(f"Found space at {x} in shelf {self.name} for module {module.name}")
                module.x = x
                #module.y = sum([s.height for s in shelves if s != self]) + (self.height - module.height) / 2
                self.insert_module(module, x)
                self.check_module_overlap()
                print(f"Placed module {module.name} at ({module.x}, {module.y}) on shelf {self.name}\n {self.contained_modules}")
                return True
                #shelf.check_module_overlap()
            else:
                # If no space was found in any shelf
                print(f"Warning: Could not place module {module.name} due to size constraints.")
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
                print(f"Warning: Module {module.name} overlaps with previous module(s) in the shelf!")
                return True
            previous_x = module.x + module.width
        return False
    
    def find_space(self, size_needed):
        # Find the first available space in the shelf to place a module of the given size
        previous_x = self.contained_modules[0].width if self.contained_modules else 0
        index = 1
        while index < len(self.contained_modules):
            module = self.contained_modules[index]
            space = module.x - previous_x
            if space >= size_needed:
                return previous_x
            previous_x = module.x + module.width
            index += 1
        
        # Check the remaining space after the last module
        remaining_space = self.max_width - previous_x
        if remaining_space >= size_needed:
            return previous_x
        return None

    def __repr__(self):
        return f"Shelf(height={self.height}, max_width={self.max_width}, used_width={self.used_width})"