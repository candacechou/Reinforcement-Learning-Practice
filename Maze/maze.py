import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display


# implemented methods
methods = ['DynProg', 'ValIter']

class Maze :
    # Action
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }
    # Reward values
    STEP_REWARD       = -1
    GOAL_REWARD       = 0
    IMPOSSIBLE_REWARD = -100
    TOWARD_MONATOUR   = -10
    DIE_REWARD        = -25
    def __init__(self, maze,Time, weights = None, random_rewards = False, monatour_stand = False):
        """ Constructor of the environment Maze"""
        self.maze = maze
        self.T = Time
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        ## create the monatour_Path
        self.monatour_Path = self.__monatour_path_generation(monatour_stand)
    
    
    
    def __actions(self):
        actions = dict()
        actions[self.STAY]       =(0,0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions
    
    
    def __states(self):
        states = dict()
        map = dict()
        end = False
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i,j] != 1:
                    states[s] = (i,j)
                    map[(i,j)] = s
                    s += 1
        return states, map
    
    
    def __move(self,state,action):

        """Makes a step in the maze, 
            given a current position and an action.
            If the action STAY or an inadmissible action is used, 
            the agent stays in place."""
        """ input value of state """
        """:return tuple next_cell: 
            Position (x,y) on the maze that agent transitions to."""
        # Compute the future position given current (state, action)
       
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1)
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state
        else:
            return self.map[(row,col)];
        
        
        
    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);
        # Compute the transition probabilities. 
        # Note that the transitions are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s  = self.__move(s,a)
                transition_probabilities[next_s,s,a] = 1
        return transition_probabilities
    
    def rewards(self,current_state,a,monatour_state, weights = None, random_rewards = None,monatour_stand = False):
        ### current_state  = (row,col)
        ### action         = a
        ### monatour_state = (mrow, mcol)
        
        # allowed_distance between the next move and the monatour_state
        if monatour_stand == False:
          allowed_dist = 2
        else:
          allowed_dist = 3
        # the next state
        next_s = self.states[self.__move(self.map[current_state],a)]
        # calculate the manhatan distance
        manhattan_dist = abs(next_s[0] - monatour_state[0]) + abs(next_s[1] - monatour_state[1])
        # possible next state of monatour
        ## reward for hitting a wall
        if next_s == current_state and a != self.STAY:
            return self.IMPOSSIBLE_REWARD
        ## reward for reaching the exit 
        elif self.maze[next_s[0],next_s[1]] == 2:
            return self.GOAL_REWARD
        ## every other move
        else:
            ## if the monatour is far away from us
            if manhattan_dist >= allowed_dist :
                return self.STEP_REWARD
            ## if we are caught 
            elif manhattan_dist == 1  :
                return self.DIE_REWARD
            ## if the monatour is around us
            elif manhattan_dist != 1 and manhattan_dist <= allowed_dist and manhattan_dist != 0:
                return self.TOWARD_MONATOUR / manhattan_dist
            else:
                return self.TOWARD_MONATOUR - 0.5 * self.TOWARD_MONATOUR
            
    def __monatour_path_generation(self,monatour_stand = True):
        ### Path is a list of tuple , length = T
        Path = [(5,6)]
        from random import seed
        from random import randint
        # seed random number generator
       
        
        for t in range(self.T):
          foo = True
          while foo :
            value = randint(0, 300) % 5
            #print('value:',value,self.actions_names[value])
            next_s = (0,0)
            ## stand still 
            if value == 0 :
                if not monatour_stand:
                    continue
                else:
                    Path.append(Path[-1])
                    foo = False
            else:
                next_s = (Path[-1][0] + self.actions[value][0] , Path[-1][1] + self.actions[value][1])
                if (next_s[0] == -1) or (next_s[0] == self.maze.shape[0]) or (next_s[1] == -1) or(next_s[1] == self.maze.shape[1]):
                    continue
                else:
                    Path.append(next_s)
                    foo = False
        return Path
        
    def simulate(self, start, policy, method):
        
        if method not in methods:
          error = 'ERROR: the argument method must be in {}'.format(method)
          raise NameError(error);
        path = []
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            t = 0
            s = start
            path.append(start)
            while t < horizon-1:
                next_s = self.states[self.__move(self.map[s],policy[self.map[s],t])]
                if next_s == path[-1] and next_s == self.monatour_Path[0]:
                    break
                else:
                    path.append(next_s)
                    # update time and state for next iteration
                    t = t+1
                    s = next_s
            return path
        else:
            error = 'no such method'
            raise NameError(error)
