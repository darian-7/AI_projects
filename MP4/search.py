import heapq
# You do not need any other imports

def best_first_search(starting_state):
    '''
    Implementation of best first search algorithm

    Input:
        starting_state: an AbstractState object

    Return:
        A path consisting of a list of AbstractState states
        The first state should be starting_state
        The last state should have state.is_goal() == True
    '''
    # we will use this visited_states dictionary to serve multiple purposes
    # - visited_states[state] = (parent_state, distance_of_state_from_start)
    #   - keep track of which states have been visited by the search algorithm
    #   - keep track of the parent of each state, so we can call backtrack(visited_states, goal_state) and obtain the path
    #   - keep track of the distance of each state from start node
    #       - if we find a shorter path to the same state we can update with the new state 
    # NOTE: we can hash states because the __hash__/__eq__ method of AbstractState is implemented
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
    return []

# TODO(III): implement backtrack method, to be called by best_first_search upon reaching goal_state
# Go backwards through the pointers in visited_states until you reach the starting state
# You can reuse the backtracking code from MP3
# NOTE: the parent of the starting state is None
def backtrack(visited_states, goal_state):
    returning_track = []
    # Your code here ---------------

    while goal_state:
        returning_track.append(goal_state)
        goal_state = visited_states[goal_state][0]
    return returning_track[::-1]


