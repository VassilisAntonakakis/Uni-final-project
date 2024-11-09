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
