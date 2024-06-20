# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq


# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
def astar(maze):        # From MP4 search.py (A* implementation)

    starting_state = maze.get_start()

    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    #   - you can reuse the search code from mp3...
    # Your code here ---------------
    current_state = starting_state
    
    while frontier:
            current_state = heapq.heappop(frontier)     # Pop and return smallest item from heap

            if current_state.is_goal():
                break

            for neighbor_state in current_state.get_neighbors():    # Check if neighbor state has been visited
                if neighbor_state not in visited_states:
                    visited_states[neighbor_state] = (current_state, neighbor_state.dist_from_start)      # Visited neighbor is flagged as 'visited'
                    heapq.heappush(frontier, neighbor_state)    # Visited neighboring state is added to the frontier
                elif visited_states[neighbor_state][1] > neighbor_state.dist_from_start:
                    visited_states[neighbor_state] = (current_state, neighbor_state.dist_from_start)
    # ------------------------------
    # Backtrack if goal is found
    if len(frontier) > 0:
        return backtrack(visited_states, current_state)
    # Return an empty list if goal not found
    return None


# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI

def backtrack(visited_states, goal_state):      # From MP3 search.py (backtrack path)

    returning_track = []

    def check_starting_state(state):
        return visited_states[state][0] is None

    def parent_state(state):
        return visited_states[state][0]

    current_state = goal_state

    while not check_starting_state(current_state):      # Get from goal state to start state
        returning_track.append(current_state)
        current_state = parent_state(current_state)

    returning_track.append(current_state)

    return returning_track[::-1]      # Gives all states in space in reverse order

