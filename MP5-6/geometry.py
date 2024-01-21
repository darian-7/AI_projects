# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Joshua Levine (joshua45@illinois.edu)
# Inspired by work done by James Gao (jamesjg2@illinois.edu) and Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP5
"""

import numpy as np
import math
from alien import Alien
from typing import List, Tuple
from copy import deepcopy

def does_alien_touch_wall(alien: Alien, walls: List[Tuple[int]]): 
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]

        Return:
            True if touched, False if not
    """
    alien_width = alien.get_width()
    
    # Check for circular alien
    if alien.is_circle():
        alien_center = alien.get_centroid()
        for wall_segment in walls:
            start_vertex, end_vertex = (wall_segment[0], wall_segment[1]), (wall_segment[2], wall_segment[3])
            if point_segment_distance(alien_center, (start_vertex, end_vertex)) <= alien_width:
                return True

    # Check for oblong alien shape
    else:
        alien_endpoints = alien.get_head_and_tail()
        for wall_segment in walls:
            start_vertex, end_vertex = (wall_segment[0], wall_segment[1]), (wall_segment[2], wall_segment[3])
            if segment_distance((start_vertex, end_vertex), alien_endpoints) <= alien_width:
                return True

    # No collision detected
    return False


def is_alien_within_window(alien: Alien, window: Tuple[int]):
    """Determine whether the alien stays within the window

        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
    """
   # width, height = window
   # x, y = alien.get_centroid()
   # al_length = alien.get_length()
   # al_width = alien.get_width()

    #   Boundaries of C-space window
    c_space = [(0, 0, 0, window[1]),(0, window[1], window[0], window[1]), (window[0], window[1], window[0], 0), ( window[0], 0, 0, 0 )]

    if alien.is_circle(): 
        if does_alien_touch_wall(alien, c_space):
            return False    # Alien is entirely inside or outside window boundary
    # Check if centroid of alien is in or out of C-space boundary
        if is_point_in_polygon(alien.get_centroid(), c_space):
            return True
        else:
            return False

    else:
        if does_alien_touch_wall(alien, c_space):
            return False
        entire_oblong = alien.get_head_and_tail()
        oblong_head = entire_oblong[0]
        if is_point_in_polygon (oblong_head, c_space):
            return True
        else:
            return False
    
# Shoelace formula: Split the polygon into triangles formed by the point and two consecutive vertices of the polygon. If the sum of these triangles' areas equals the area of the polygon, then the point lies within.
def is_point_in_polygon(point, polygon):        
    """Determine whether a point is in a polygon.

        Args:
            point (tuple): shape of (2, ). The coordinate (x, y) of the query point.
            polygon (tuple): shape of (4, 2). The coordinate (x, y) of the vertices.
    """
    
    def area(a, b, c):      # Helper function: Calculate area of a triangle given its 3 vertices
        return abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0

    polygon_area = area(polygon[0], polygon[1], polygon[2]) + area(polygon[0], polygon[2], polygon[3])
    triangle_areas = sum([area(point, polygon[i], polygon[(i+1)%4]) for i in range(4)])
    buffer = 1e-10      # Buffer to handle floating point inaccuracies

    return abs(polygon_area - triangle_areas) < buffer


def does_alien_path_touch_wall(alien: Alien, walls: List[Tuple[int]], waypoint: Tuple[int, int]):
    """Determine whether the alien's straight-line path from its current position to the waypoint touches a wall

        Args:
            alien (Alien): the current alien instance
            walls (List of tuple): List of endpoints of line segments that comprise the walls in the maze in the format
                         [(startx, starty, endx, endx), ...]
            waypoint (tuple): the coordinate of the waypoint where the alien wants to move

        Return:
            True if touched, False if not
    """

    alien_centroid = alien.get_centroid()
    alien_width = alien.get_width()
    alien_length = alien.get_length()

    if alien_centroid == waypoint:
        return does_alien_touch_wall(alien, walls)

    segment1 = (alien_centroid, waypoint)

    for segment in walls:
        segment2 = ((segment[0], segment[1]), (segment[2], segment[3]))
        alien_shape = alien.get_shape()

        if alien_shape == 'Ball':
            distance = segment_distance(segment1, segment2)
            if do_segments_intersect(segment1, segment2) or distance <= alien_width:
                return True

        if alien_shape in ['Vertical', 'Horizontal']:
            al_head, al_tail = alien.get_head_and_tail()

            if alien_shape == 'Vertical':
                seg3_endpoint = (waypoint[0], waypoint[1]-(alien_length/2))
                seg4_endpoint = (waypoint[0], waypoint[1]+(alien_length/2))

            else:  # 'Horizontal'
                seg3_endpoint = (waypoint[0] + (alien_length/2), waypoint[1])
                seg4_endpoint = (waypoint[0] - (alien_length/2), waypoint[1])

            segment3 = (al_head, seg3_endpoint)
            segment4 = (al_tail, seg4_endpoint)
            distance3 = segment_distance(segment3, segment2)
            distance4 = segment_distance(segment4, segment2)

            if do_segments_intersect(segment1, segment2) or distance3 <= alien_width or distance4 <= alien_width:
                return True

    return False 



def point_segment_distance(p, s):
    """Compute the distance from the point to the line segment.

        Args:
            p: A tuple (x, y) of the coordinates of the point.
            s: A tuple ((x1, y1), (x2, y2)) of coordinates indicating the endpoints of the segment.

        Return:
            Euclidean distance from the point to the line segment.
    """
    # x, y = p        # Point coord
    # (x1, y1), (x2, y2) = s      # Segment coord
    # x_delta = x2 - x1
    # y_delta = y2 - y1

    # if (x_delta**2 + y_delta**2) == 0:
    #     return np.hypot(x - x1, y - y1)     # Segment == point

    # #  Distance from x1, y1 to the point
    # t = ((x - x1) * x_delta + (y - y1) * y_delta) / (x_delta**2 + y_delta**2)

    # t = max(0, min(1, t))

    # # Distance between projected point and p
    # pro_x = x1 + t * x_delta
    # pro_y = y1 + t * y_delta
    # return np.hypot(x - pro_x, y - pro_y)

    px, py = p
    (sx1, sy1), (sx2, sy2) = s

    # Calculate directional vectors
    vector_point = (px - sx1, py - sy1)
    vector_segment = (sx2 - sx1, sy2 - sy1)

    dot_product = vector_point[0] * vector_segment[0] + vector_point[1] * vector_segment[1]
    magnitude_squared = vector_segment[0]**2 + vector_segment[1]**2
    
    if magnitude_squared == 0:
        # Segment is a point, return distance to the point
        return math.dist(p, s[0])
    
    ratio = dot_product / magnitude_squared
    if ratio <= 0:
        closest_x, closest_y = sx1, sy1
    elif ratio >= 1:
        closest_x, closest_y = sx2, sy2
    else:
        closest_x = sx1 + ratio * vector_segment[0]
        closest_y = sy1 + ratio * vector_segment[1]

    return math.dist(p, (closest_x, closest_y))


def get_line_eq(p1, p2):        # Helper function: Returns a, b and c for the line equation ax + by = c
    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = a*p1[0] + b*p1[1]
    return a, b, c

def point_to_line_dist(point, line):        # Helper function: Returns the shortest distance between a point and a line segment
    a, b, c = get_line_eq(*line)
    return abs(a*point[0] + b*point[1] - c) / math.sqrt(a*a + b*b)

def check_alien_touches_line(alien, line):      # Helper function: Check if the alien touches a particular line
    if alien.is_circle():
        centroid = alien.get_centroid()
        return point_to_line_dist(centroid, line) <= alien.get_width()
    else:
        head, tail = alien.get_head_and_tail()
        return segment_distance(line, (head, tail)) <= alien.get_width()

def orientation(p, q, r):   # Helper function: Check orientation of triplet (p, q, r)
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def on_segment(p, q, r):    # Helper function: Check if point q lies on line segment pr
    return q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])

def do_segments_intersect(s1, s2):
    """Determine whether segment1 intersects segment2.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            True if line segments intersect, False if not.
    """
    p1, q1 = s1
    p2, q2 = s2
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    return False

# Reference: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

def segment_distance(s1, s2):
    """Compute the distance from segment1 to segment2.  You will need `do_segments_intersect`.

        Args:
            s1: A tuple of coordinates indicating the endpoints of segment1.
            s2: A tuple of coordinates indicating the endpoints of segment2.

        Return:
            Euclidean distance between the two line segments.
    """
    if do_segments_intersect(s1, s2):
        return 0

    dists = []

#    dists = [point_to_line_dist(s1[0], s2), point_to_line_dist(s1[1], s2), point_to_line_dist(s2[0], s1), point_to_line_dist(s2[1], s1)]

    # s1 endpoints
    dists.append( point_segment_distance(s1[0], s2))
    dists.append( point_segment_distance(s1[1], s2))

    # s2 endpoints                             
    dists.append( point_segment_distance(s2[0], s1))
    dists.append( point_segment_distance(s2[1], s1))

    return min(dists)


if __name__ == '__main__':

    from geometry_test_data import walls, goals, window, alien_positions, alien_ball_truths, alien_horz_truths, \
        alien_vert_truths, point_segment_distance_result, segment_distance_result, is_intersect_result, waypoints


    # Here we first test your basic geometry implementation
    def test_point_segment_distance(points, segments, results):
        num_points = len(points)
        num_segments = len(segments)
        for i in range(num_points):
            p = points[i]
            for j in range(num_segments):
                seg = ((segments[j][0], segments[j][1]), (segments[j][2], segments[j][3]))
                cur_dist = point_segment_distance(p, seg)
                assert abs(cur_dist - results[i][j]) <= 10 ** -3, \
                    f'Expected distance between {points[i]} and segment {segments[j]} is {results[i][j]}, ' \
                    f'but get {cur_dist}'


    def test_do_segments_intersect(center: List[Tuple[int]], segments: List[Tuple[int]],
                                   result: List[List[List[bool]]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    if do_segments_intersect(a, b) != result[i][j][k]:
                        if result[i][j][k]:
                            assert False, f'Intersection Expected between {a} and {b}.'
                        if not result[i][j][k]:
                            assert False, f'Intersection not expected between {a} and {b}.'


    def test_segment_distance(center: List[Tuple[int]], segments: List[Tuple[int]], result: List[List[float]]):
        for i in range(len(center)):
            for j, s in enumerate([(40, 0), (0, 40), (100, 0), (0, 100), (0, 120), (120, 0)]):
                for k in range(len(segments)):
                    cx, cy = center[i]
                    st = (cx + s[0], cy + s[1])
                    ed = (cx - s[0], cy - s[1])
                    a = (st, ed)
                    b = ((segments[k][0], segments[k][1]), (segments[k][2], segments[k][3]))
                    distance = segment_distance(a, b)
                    assert abs(result[i][j][k] - distance) <= 10 ** -3, f'The distance between segment {a} and ' \
                                                                        f'{b} is expected to be {result[i]}, but your' \
                                                                        f'result is {distance}'


    def test_helper(alien: Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()

        touch_wall_result = does_alien_touch_wall(alien, walls)
        in_window_result = is_alien_within_window(alien, window)

        assert touch_wall_result == truths[
            0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {touch_wall_result}, ' \
                f'expected: {truths[0]}'
        assert in_window_result == truths[
            2], f'is_alien_within_window(alien, window) with alien config {config} returns {in_window_result}, ' \
                f'expected: {truths[2]}'


    def test_check_path(alien: Alien, position, truths, waypoints):
        alien.set_alien_pos(position)
        config = alien.get_config()

        for i, waypoint in enumerate(waypoints):
            path_touch_wall_result = does_alien_path_touch_wall(alien, walls, waypoint)

            assert path_touch_wall_result == truths[
                i], f'does_alien_path_touch_wall(alien, walls, waypoint) with alien config {config} ' \
                    f'and waypoint {waypoint} returns {path_touch_wall_result}, ' \
                    f'expected: {truths[i]}'

            # Initialize Aliens and perform simple sanity check.


    alien_ball = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Ball', window)
    test_helper(alien_ball, alien_ball.get_centroid(), (False, False, True))

    alien_horz = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal', window)
    test_helper(alien_horz, alien_horz.get_centroid(), (False, False, True))

    alien_vert = Alien((30, 120), [40, 0, 40], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical', window)
    test_helper(alien_vert, alien_vert.get_centroid(), (True, False, True))

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Horizontal',
                            window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal', 'Ball', 'Vertical'), 'Vertical',
                            window)

    # Test validity of straight line paths between an alien and a waypoint
    test_check_path(alien_ball, (30, 120), (False, True, True), waypoints)
    test_check_path(alien_horz, (30, 120), (False, True, False), waypoints)
    test_check_path(alien_vert, (30, 120), (True, True, True), waypoints)

    centers = alien_positions
    segments = walls
    test_point_segment_distance(centers, segments, point_segment_distance_result)
    test_do_segments_intersect(centers, segments, is_intersect_result)
    test_segment_distance(centers, segments, segment_distance_result)

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    # Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110, 55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))

    print("Geometry tests passed\n")
