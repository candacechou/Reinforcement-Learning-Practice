import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random


def QLearning(rob,limits,method):

    Q           = rob.Q 
    r           = rob.rewards
    n_states    = rob.n_states
    n_actions   = rob.n_actions
    epsilon     = rob.epsilon
    Lambda      = rob.Lambda
    Alpha       = rob.Alpha
    V           = np.zeros(n_states)
    Vt          = []
    Vt1         = []
    Vt2         = []
    t           = 1
    flag        = True
    s           = (0,0,3,3)
    markInts    = rob.map[s]
    markInt2    = random.randint(0,n_states-1)
    markInt3    = random.randint(0,n_states-1)
    while markInt3 == markInt2:
        markInt3    = random.randint(0,n_states-1)
    rate        = 1
    tol         = 1e-5 #(1 - Lambda)* epsilon/Lambda
    action_r    = generate_action(rob,s,Q,epsilon)
    err         = 0.0
  
   # Initialization of the VI
    while(t < limits and flag):
    
        if method == 'QLearning':
            epsilon = 1
            action_r = generate_action(rob,s,Q,epsilon)

    
        next_s = next_move(rob,s,action_r)
        rate,Alpha = update_alpha(rob.map[s],action_r,Alpha)
        if method == 'QLearning':
            Q[rob.map[s],action_r] = (1 - rate) * Q[rob.map[s],action_r] + rate * (r[rob.map[s],action_r] + Lambda * (np.max(Q[rob.map[next_s],:])))
        else :
          #rate  = 1/t
            next_action = generate_action(rob,next_s,Q,epsilon)
            Q[rob.map[s],action_r] = (1 - rate) * Q[rob.map[s],action_r] + rate * (r[rob.map[s],action_r] + Lambda * (Q[rob.map[next_s],next_action]))
            action_r = next_action

        V      = np.max(Q, 1)
        temp_v = V[markInts]
        Vt.append(temp_v)
        temp_v = V[markInt2]
        Vt1.append(temp_v)
        temp_v = V[markInt3]
        Vt2.append(temp_v)


        t      += 1
        s       = next_s
        if(t%100000 == 0):
            print('current iteration:',t)
    
    return Vt,Vt1,Vt2, Q,t



def update_alpha(s,a,Alpha):
  #print(Alpha[s][a])
  Alpha[s][a] += 1
  #print(Alpha[s][a] )
  num_update = Alpha[s][a]
  
  #rate = num_update**(-2/3)
  rate  = 1/num_update
  return rate, Alpha

def next_move(town,s,action_r):
  row  = s[0]
  col  = s[1]
  prow = s[2]
  pcol = s[3]
  
  ### robber moves
  new_row  = s[0] + town.actions[action_r][0]
  new_col  = s[1] + town.actions[action_r][1]
  hitting_rob = (new_row == -1) or (new_row == town.town.shape[0]) or (new_col == -1) or (new_col == town.town.shape[1])
  ### police moves
  Flag = True
  while(Flag):
    action_p = random.randint(1,4)
    new_prow = s[2] + town.actions[action_p][0]
    new_pcol = s[3] + town.actions[action_p][1]
    hitting_po = (new_prow == -1) or (new_prow == town.town.shape[0]) or(new_pcol == -1) or (new_pcol == town.town.shape[1])
    if hitting_po:
      Flag = True
    else:
      Flag = False
  next_s = (new_row,new_col,new_prow,new_pcol)
  if hitting_rob :
    ##change
    return (row,col,new_prow,new_pcol)
  else:
    return next_s

def generate_action(rob,s,Q,epsilon):
  if random.random() < epsilon:
    action_r = random.randint(0,4)
  else:
    action_r = np.argmax(Q[rob.map[s],:])
  return action_r


class Robber:
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

  # Reward Value
  IN_BANK     = 1
  GOT_CAUGHT  = -10
  ### not used
  NIN_BANK    = 0
  NEAR_POLICE = 0
  IMPOSSIBLE_MOVE = 0
  def __init__(self,town,Lambda,epsilon = 0.1):
    """  Constructor of the environmnet Maze"""
    self.town            = town
    self.Lambda          = Lambda
    self.epsilon         = epsilon
    self.actions         = self.__actions()
    self.states,self.map = self.__states()
    self.n_states        = len(self.states)
    self.n_actions       = len(self.actions)
    self.Q               = np.zeros((self.n_states,self.n_actions))
    self.rewards         = self.__rewards()
    self.Alpha           = np.zeros((self.n_states,self.n_actions)) 

  def __actions(self):
    actions                  = dict()
    actions[self.STAY]       = (0,0)
    actions[self.MOVE_LEFT]  = (0,-1)
    actions[self.MOVE_RIGHT] = (0,1)
    actions[self.MOVE_UP]    = (-1,0)
    actions[self.MOVE_DOWN]  = (1,0)
    return actions

  def __states(self):
    states  = dict()
    map      = dict()
    s        = 0
    for i in range(self.town.shape[0]):
      for j in range(self.town.shape[1]):
        for k in range(self.town.shape[0]):
          for l in range(self.town.shape[1]):
            states[s]      = (i,j,k,l)
            map[(i,j,k,l)] = s
            s              += 1 
    return states, map

  def __move(self,state,action):
    row = self.states[state][0] + self.actions[action][0]
    col = self.states[state][1] + self.actions[action][1]
    next_states = []
    hitting_robber =  (row == -1) or (row == self.town.shape[0]) or \
                              (col == -1) or (col == self.town.shape[1]) 
    for i in range(self.n_actions):
      if i == 0:
        continue
      else:
        mrow = self.states[state][2] + self.actions[i][0]
        mcol = self.states[state][3] + self.actions[i][1]
        hitting_police =  (mrow == -1) or (mrow == self.town.shape[0]) or \
                                    (mcol == -1) or (mcol == self.town.shape[1]) 
        if hitting_police:
          continue
        else:
          if hitting_robber:
            next_states.append(self.map[(self.states[state][0],self.states[state][1],mrow,mcol)])
          else:
            next_states.append(self.map[(row,col,mrow,mcol)])
    return next_states

  def __transitions(self):
    ## Initialize the transition probailities tensor (S,S,A)
    dimensions               = (self.n_states,self.n_states,self.n_actions)
    transition_probabilities = np.zeros(dimensions)
    # Compute the transition probabilities. 
    # Note that the transitions are deterministic.
    for s in range(self.n_states):
      for a in range(self.n_actions):
        next_s  = self.__move(s,a)
        for npos_s in next_s:                           
          transition_probabilities[npos_s,s,a] = 1/ len(next_s)
    return transition_probabilities

  def __rewards(self):
    rewards = np.zeros((self.n_states,self.n_actions))
    for s in self.states:
      for a in range(self.n_actions):
        next_s    = self.__move(s,a)
        dead_flag = False
        next_row  = self.states[next_s[0]][0]
        next_col  = self.states[next_s[0]][1]
        #print(s)
        hitting   = (next_row == self.states[s][0]) and (next_col == self.states[s][1]) and (a != 0)
        if hitting:
          rewards[s,a] = self.IMPOSSIBLE_MOVE
          dead_flag    = True
          
        else:
      # If Police and Robber might get same position : DEAD_REWARD
          #print('---------------')
          for pos_next_s in next_s:
            # print(self.states[pos_next_s])
            if (next_row == self.states[pos_next_s][2]) and (next_col == self.states[pos_next_s][3]) and dead_flag == False:
              rewards[s,a] = self.GOT_CAUGHT 
            if (self.town[next_row,next_col]== 2) and (dead_flag == False):
              rewards[s,a] = self.IN_BANK
            elif (self.town[next_row,next_col] == 0) and dead_flag == False:
              rewards[s,a] =self.NIN_BANK
            
              
    return rewards
            