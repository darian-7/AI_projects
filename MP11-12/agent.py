import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table.

        self.N[state][action] += 1      # State-action pair counter 

        # if state is not None and action is not None:
        #     self.N[state + (action,)]=+1

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 

        alpha = self.C / (self.C + self.N[s][a])        # Learning rate (Q-learning eq)
        # Q val update calculation
        a_star = max(self.Q[s_prime])     # Max Q-val for next state
        self.Q[s][a] += alpha * (r + self.gamma * a_star - self.Q[s][a])

# Reference: https://learning.oreilly.com/library/view/hands-on-q-learning-with/9781789345803/3aa19289-1223-4770-8df2-c713a68aa969.xhtml/ 

# TESTING -----------------------------------

# Test 1: python mp11_12.py --snake_head_x 5 --snake_head_y 5 --food_x 2 --food_y 2 --width 18 --height 10 --rock_x 3 --rock_y 4 --Ne 40 --C 40 --gamma 0.7 
# Test 2: python mp11_12.py --snake_head_x 5 --snake_head_y 5 --food_x 2 --food_y 2 --width 18 --height 10 --rock_x 3 --rock_y 4 --Ne 20 --C 60 --gamma 0.5 
# Test 3: python mp11_12.py --snake_head_x 3 --snake_head_y 4 --food_x 2 --food_y 2 --width 10 --height 18 --rock_x 5 --rock_y 5 --Ne 30 --C 30 --gamma 0.6 

# Old avg points: test1 = 14.51 | test2 = 1.5 | test3 = 9.3 
# -------------------------------------------
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO - MP12: write your function here

        def best_action(q_values):
            return np.argmax(q_values)

        if not self._train:
            # In evaluation mode: choose action with highest Q-value
            self.a = best_action(self.Q[s_prime])
        else:
            # In training mode: update Q-table and choose next action
            if self.a is not None and self.s is not None:
                reward = -1 if dead else 1 if points > self.points else -0.1

                best_future_value = max(self.Q[s_prime])
                self.N[self.s][self.a] += 1
                alpha = self.C / (self.N[self.s][self.a] + self.C)
                self.Q[self.s][self.a] += alpha * (reward + self.gamma * best_future_value - self.Q[self.s][self.a])

            if dead:
                self.reset()
                return 1  # Default action after reset

            self.s = s_prime
            self.points = points

            # Exploration vs Exploitation
            if any(self.N[s_prime][action] < self.Ne for action in range(len(self.Q[s_prime]))):
                self.a = np.random.choice(len(self.Q[s_prime]))
            else:
                self.a = best_action(self.Q[s_prime])

        return self.actions[self.a]

# Reference: https://medium.com/@nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c


    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x (0), snake_head_y (1), snake_body (2), food_x (3), food_y (4), rock_x (5), rock_y (6)] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 

        # Initialization
        snake_head_x, snake_head_y = environment[0], environment[1]
        food_x, food_y = environment[3], environment[4]
        rock_x, rock_y = environment[5], environment[6]
        snake_body_segments = environment[2]
        adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right = 0, 0, 0, 0


        # Wall or rock check
        wall_on_right = snake_head_x+2 == self.display_width
        rock_on_right = rock_x == snake_head_x + 1 and snake_head_y == rock_y
        wall_on_left = snake_head_x-1==0
        rock_on_left = snake_head_x - rock_x == 2 and snake_head_y == rock_y


        if snake_head_x == 1 or rock_on_left or (wall_on_right and rock_on_left):
            adjoining_wall_x = 1
        elif wall_on_right or rock_on_right:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0

        wall_above = snake_head_y+2 == self.display_height
        rock_above = snake_head_y - rock_y == 1 and snake_head_x == rock_x
        wall_below = snake_head_y-1 == 0
        rock_below = (rock_y-snake_head_y==1 and abs(snake_head_x-rock_x)<=1)
        
        if snake_head_y == 1 or rock_above or (wall_above and rock_above):
            adjoining_wall_y = 1
        elif (wall_above) or rock_below:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0


        # Food direction
        if snake_head_x == food_x:
            food_dir_x = 0
        elif food_x - snake_head_x < 0:
            food_dir_x = 1
        else:
            food_dir_x = 2

        if snake_head_y == food_y:
            food_dir_y = 0
        elif food_y - snake_head_y < 0:
            food_dir_y = 1
        else:
            food_dir_y = 2


        # Adjacent bodies
        snake_body_segments = environment[2]
        for segment in snake_body_segments:
            if segment[0] == snake_head_x and snake_head_y - segment[1] == 1:
                adjoining_body_top = 1

            if segment[0] == snake_head_x and snake_head_y - segment[1] == -1:
                adjoining_body_bottom = 1

            if segment[1] == snake_head_y and snake_head_x - segment[0] == 1:
                adjoining_body_left = 1

            if segment[1] == snake_head_y and snake_head_x - segment[0] == -1:
                adjoining_body_right = 1


        state = (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, 
                 adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        return state

