"""Classes to aid localization of trunks (nb4)"""

import numpy as np


class Point:
    index = None
    position = None
    paths = []
    network = None
    vec = None
    linked_to = None
    tree_id = None

    def __init__(self, index, position):
        self.index = index
        self.position = position

    def add_path(self, path):
        self.paths = np.append(self.paths, path)


class Path:
    index = None
    points = []
    network = None

    def __init__(self, index):
        self.index = index

    def add_point(self, this_point):
        self.points = np.append(self.points, this_point)


class Network:
    index = None
    paths = []
    points = []
    top = None

    def __init__(self, index):
        self.index = index

    def add_path(self, path):
        self.paths = np.append(self.paths, path)
        path.network = self
        for point in path.points:
            point.network = self
        self.points = np.append(self.points, path.points)
