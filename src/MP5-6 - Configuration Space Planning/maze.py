# maze.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Joshua Levine (joshua45@illinois.edu) and Jiaqi Gun
"""
This file contains the Maze class, which reads in a maze file and creates
a representation of the maze that is exposed through a simple interface.
"""

import copy
from state import MazeState, euclidean_distance
from geometry import does_alien_path_touch_wall, does_alien_touch_wall


class MazeError(Exception):
    pass


class NoStartError(Exception):
    pass


class NoObjectiveError(Exception):
    pass


class Maze:
    def __init__(self, alien, walls, waypoints, goals, move_cache={}, k=5, use_heuristic=True):
        """Initialize the Maze class, which will be navigated by a crystal alien

        Args:
            alien: (Alien), the alien that will be navigating our map
            walls: (List of tuple), List of endpoints of line segments that comprise the walls in the maze in the format
                        [(startx, starty, endx, endx), ...]
            waypoints: (List of tuple), List of waypoint coordinates in the maze in the format of [(x, y), ...]
            goals: (List of tuple), List of goal coordinates in the maze in the format of [(x, y), ...]
            move_cache: (Dict), caching whether a move is valid in the format of
                        {((start_x, start_y, start_shape), (end_x, end_y, end_shape)): True/False, ...}
            k (int): the number of waypoints to check when getting neighbors
        """
        self.k = k
        self.alien = alien
        self.walls = walls

        self.states_explored = 0
        self.move_cache = move_cache
        self.use_heuristic = use_heuristic

        self.__start = (*alien.get_centroid(), alien.get_shape_idx())
        self.__objective = tuple(goals)

        # Waypoints: the alien must move between waypoints (goal is a special waypoint)
        # Goals are also viewed as a part of waypoints
        self.__waypoints = waypoints + goals
        self.__valid_waypoints = self.filter_valid_waypoints()
        self.__start = MazeState(self.__start, self.get_objectives(), 0, self, self.use_heuristic)

        # self.__dimensions = [len(input_map), len(input_map[0]), len(input_map[0][0])]
        # self.__map = input_map

        if not self.__start:
            # raise SystemExit
            raise NoStartError("Maze has no start")

        if not self.__objective:
            raise NoObjectiveError("Maze has no objectives")

        if not self.__waypoints:
            raise NoObjectiveError("Maze has no waypoints")

    def is_objective(self, waypoint):
        """"
        Returns True if the given position is the location of an objective
        """
        return waypoint in self.__objective

    # Returns the start position as a tuple of (row, col, level)
    def get_start(self):
        assert (isinstance(self.__start, MazeState))
        return self.__start

    def set_start(self, start):
        """
        Sets the start state
        start (MazeState): a new starting state
        return: None
        """
        self.__start = start

    # Returns the dimensions of the maze as a (num_row, num_col, level) tuple
    # def get_dimensions(self):
    #     return self.__dimensions

    # Returns the list of objective positions of the maze, formatted as (x, y, shape) tuples
    def get_objectives(self):
        return copy.deepcopy(self.__objective)

    def get_waypoints(self):
        return self.__waypoints

    def get_valid_waypoints(self):
        return self.__valid_waypoints

    def set_objectives(self, objectives):
        self.__objective = objectives

    # TODO VI
    def filter_valid_waypoints(self):
        """Filter valid waypoints on each alien shape

            Return:
                A dict with shape index as keys and the list of waypoints coordinates as values
        """
        valid_waypoints = {i: [] for i in range(len(self.alien.get_shapes()))}
        for i, shape in enumerate(self.alien.get_shapes()):
            for waypoint in self.__waypoints:
                alien = self.create_new_alien(waypoint[0], waypoint[1], i)
                if not does_alien_touch_wall(alien, self.walls):
                    valid_waypoints[i].append(waypoint)
        return valid_waypoints

        # If alien is on same coord then its invalid waypoint

    # TODO VI
    def get_nearest_waypoints(self, cur_waypoint, cur_shape):
        """Find the k nearest valid neighbors to the cur_waypoint from a list of 2D points.
            Args:
                cur_waypoint: (x, y) waypoint coordinate
                cur_shape: shape index
            Return:
                the k valid waypoints that are closest to waypoint
        """
        valid_waypoints = self.filter_valid_waypoints().get(cur_shape, [])
        current_position = (cur_waypoint[0], cur_waypoint[1], cur_shape)

        # Calculate the distance to each valid waypoint.
        def compute_distance(waypoint):
            end_position = (waypoint[0], waypoint[1], cur_shape)
            return euclidean_distance((current_position[0], current_position[1]), (end_position[0], end_position[1]))

        # Filter and compute distances for valid waypoints
        waypoint_distances = [(point, compute_distance(point)) for point in valid_waypoints if cur_waypoint != point and self.is_valid_move(current_position, (point[0], point[1], cur_shape))]

        # Sort waypoints based on distances and retrieve the nearest k
        nearest_waypoints = [item[0] for item in sorted(waypoint_distances, key=lambda x: x[1])[:self.k]]

        return nearest_waypoints

    def create_new_alien(self, x, y, shape_idx):        # Create and return a new alien instance based on provided configuration
        alien_instance = copy.deepcopy(self.alien)
        alien_instance.set_alien_config([x, y, self.alien.get_shapes()[shape_idx]])
        return alien_instance


    # TODO VI
    def is_valid_move(self, start, end): # check for 0, 1, 2
        """Check if the position of the waypoint can be reached by a straight-line path from the current position
            Args:
                start: (start_x, start_y, start_shape_idx)
                end: (end_x, end_y, end_shape_idx)
            Return:
                True if the move is valid, False otherwise
        """
        # Ensure shape integrity
        if not self.is_valid_shape(start[2]) or not self.is_valid_shape(end[2]):
            return False

        # New alien for testing
        test_alien = self.initialize_alien(start[0], start[1], start[2])

        # Check alien-wall collisions
        if self.alien_collides(test_alien):
            return False

        # Verify move based on shape
        if start[2] == end[2]:
            if self.alien_path_collides(test_alien, (end[0], end[1])):
                return False
            return True
        else:
            if self.invalid_shape_move(start, end):
                print('Invalid movement detected with shape change!')
                return False

            if self.shape_change_collides(test_alien, end[2]):
                return False
            return True

    def is_valid_shape(self, shape):        # Helper function
        """Ensure the given shape index is within the valid range."""
        return 0 <= shape <= 2

    def initialize_alien(self, x, y, shape):        # Helper function
        """Create a new alien instance."""
        return self.create_new_alien(x, y, shape)

    def alien_collides(self, alien_instance):       # Helper function
        """Check if the alien touches any walls."""
        return does_alien_touch_wall(alien_instance, self.walls)

    def alien_path_collides(self, alien_instance, destination):     # Helper function
        """Check if the alien's path to a destination touches walls."""
        return does_alien_path_touch_wall(alien_instance, self.walls, destination)

    def invalid_shape_move(self, start, end):       # Helper function
        """Ensure alien doesn't move and change shape simultaneously."""
        is_same_location = start[0] == end[0] and start[1] == end[1]
        is_illegal_transformation = abs(start[2] - end[2]) > 1
        return not is_same_location or is_illegal_transformation

    def shape_change_collides(self, alien_instance, new_shape):     # Helper function
        """Check if the alien's shape change causes a collision."""
        alien_instance.set_alien_shape(alien_instance.get_shapes()[new_shape])
        return self.alien_collides(alien_instance)

    def get_neighbors(self, x, y, shape_idx):
        """Returns list of neighboring squares that can be moved to from the given coordinate
            Args:
                x: query x coordinate
                y: query y coordinate
                shape_idx: query shape index
            Return:
                list of possible neighbor positions, formatted as (x, y, shape) tuples.
        """
        self.states_explored += 1

        nearest = self.get_nearest_waypoints((x, y), shape_idx)
        neighbors = [(*end, shape_idx) for end in nearest]
        for end in [(x, y, shape_idx - 1), (x, y, shape_idx + 1)]:
            start = (x, y, shape_idx)
            if self.is_valid_move(start, end):
                neighbors.append(end)

        return neighbors
