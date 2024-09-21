# class Shelf:
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height
#         self.spaceLeft = width
#         self.nodes = []
    
#     def addNode(self, node):
#         self.nodes.append(node)
#         self.spaceLeft -= node.getWidth()
        
# class Bookshelf:
#     def __init__(self, rows):
#             self.shelves = [rows]
    
#     def addShelf(self, shelf):
#         self.shelves.append(shelf)
    
#     def addNode(self, node):
#         pass

import re
import sys

class Shelf:
    def __init__(self, coordinate, height, sitewidth, sitespacing, siteorient, sitesymmetry, subrow_origin, num_sites):
        self._coordinate = int(coordinate)
        self._height = int(height)
        self._sitewidth = int(sitewidth)
        self._sitespacing = int(sitespacing)
        self._siteorient = siteorient.strip()
        self._sitesymmetry = sitesymmetry.strip() == 'Y'
        self._subrow_origin = int(subrow_origin)
        self._num_sites = int(num_sites)

    # Getter and Setter for Coordinate
    @property
    def coordinate(self):
        return self._coordinate
    
    @coordinate.setter
    def coordinate(self, value):
        self._coordinate = int(value)
    
    # Getter and Setter for Height
    @property
    def height(self):
        return self._height
    
    @height.setter
    def height(self, value):
        self._height = int(value)

    # Getter and Setter for Sitewidth
    @property
    def sitewidth(self):
        return self._sitewidth
    
    @sitewidth.setter
    def sitewidth(self, value):
        self._sitewidth = int(value)
    
    # Getter and Setter for Sitespacing
    @property
    def sitespacing(self):
        return self._sitespacing
    
    @sitespacing.setter
    def sitespacing(self, value):
        self._sitespacing = int(value)
    
    # Getter and Setter for Siteorient
    @property
    def siteorient(self):
        return self._siteorient
    
    @siteorient.setter
    def siteorient(self, value):
        self._siteorient = value.strip()
    
    # Getter and Setter for Sitesymmetry
    @property
    def sitesymmetry(self):
        return self._sitesymmetry
    
    @sitesymmetry.setter
    def sitesymmetry(self, value):
        self._sitesymmetry = value.strip() == 'Y'
    
    # Getter and Setter for SubrowOrigin
    @property
    def subrow_origin(self):
        return self._subrow_origin
    
    @subrow_origin.setter
    def subrow_origin(self, value):
        self._subrow_origin = int(value)
    
    # Getter and Setter for NumSites
    @property
    def num_sites(self):
        return self._num_sites
    
    @num_sites.setter
    def num_sites(self, value):
        self._num_sites = int(value)
    
    def __str__(self):
        return (f"Shelf(Coordinate={self.coordinate}, Height={self.height}, "
                f"Sitewidth={self.sitewidth}, Sitespacing={self.sitespacing}, "
                f"Siteorient='{self.siteorient}', Sitesymmetry={self.sitesymmetry}, "
                f"SubrowOrigin={self.subrow_origin}, NumSites={self.num_sites})")
