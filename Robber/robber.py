import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random

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
  IN_BANK     = 10
  GOT_CAUGHT  = -50
  IMPOSSIBLE_MOVE = 0
  NIN_BANK        = 0
  def __init__(self,town,Lambda,epsilon = 0.1):
    """  Constructor of the environmnet Maze"""
    self.town                     = town
    self.Lambda                   = Lambda
    self.epsilon                  = epsilon
    self.actions                  = self.__actions()
    self.states,self.map          = self.__states()
    self.n_states                 = len(self.states)
    self.n_actions                = len(self.actions)
    self.Q                        = np.zeros((self.n_states,self.n_actions))
    self.rewards                  = self.__rewards()
    self.Alpha                    = np.zeros((self.n_states,self.n_actions))
    self.transition_probabilities = self.__transitions()
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
    rrow = self.states[state][0]
    rcol = self.states[state][1]
    prow = self.states[state][2]
    pcol = self.states[state][3]
    next_rrow = self.states[state][0] + self.actions[action][0]
    next_rcol = self.states[state][1] + self.actions[action][1]
    hit_rob   = (next_rrow == -1) or (next_rrow == self.town.shape[0]) or \
                                    (next_rcol == -1) or (next_rcol == self.town.shape[1]) 
    if hit_rob:
      next_rrow = self.states[state][0]
      next_rcol = self.states[state][1]
    next_s = []
    ## if both are in the same row but not same column
    if rrow == prow and rcol != pcol:
      ## check if the police is left or right to you
      if rcol > pcol : ## police is left you
        for i in range(self.n_actions):
          if i == 0 or i == 1:
            continue
          else:
            next_prow = prow + self.actions[i][0]
            next_pcol = pcol + self.actions[i][1]
            if hitwall(self.town,next_prow,next_pcol):
              continue
            else:
              next_s.append(self.map[(next_rrow,next_rcol,next_prow,next_pcol)])

      elif rcol < pcol : ## police is right to you
        for i in range(self.n_actions):
          if i == 0 or i == 2 :
            continue
          else :
            next_prow = prow + self.actions[i][0]
            next_pcol = pcol + self.actions[i][1]
            if hitwall(self.town,next_prow,next_pcol):
              continue
            else:
              next_s.append(self.map[(next_rrow,next_rcol,next_prow,next_pcol)])
      
    ## if both are in the same column but not same row
    elif rcol == pcol and rrow != prow:
      ## check if the police is up or down to you
      if rrow > prow : ## police is above you 
        for i in range(self.n_actions):
          if i == 0 or i == 3:
            continue
          else:
            next_prow = prow + self.actions[i][0]
            next_pcol = pcol + self.actions[i][1]
            if hitwall(self.town,next_prow,next_pcol):
              continue
            else:
              next_s.append(self.map[(next_rrow,next_rcol,next_prow,next_pcol)])

      elif rrow < prow : ## police is down to you 
        for i in range(self.n_actions):
          if i == 0 or i == 4:
            continue
          else:
            next_prow = prow + self.actions[i][0]
            next_pcol = pcol + self.actions[i][1]
            if hitwall(self.town,next_prow,next_pcol):
              continue
            else:
              next_s.append(self.map[(next_rrow,next_rcol,next_prow,next_pcol)])


    elif rcol != pcol and rrow != prow: ### we need to check which direction you in
      if rrow > prow : ### police is above you
        if rcol > pcol : ### police is left above you
          for i in range(self.n_actions):
            if i == 0 or i == 1 or i == 3:
              continue
            else:
              next_prow = prow + self.actions[i][0]
              next_pcol = pcol + self.actions[i][1]
              if hitwall(self.town,next_prow,next_pcol):
                continue
              else:
                next_s.append(self.map[(next_rrow,next_rcol,next_prow,next_pcol)])

        else : ## the police is right above you
          for i in range(self.n_actions):
            if i == 0 or i == 2 or i == 3:
              continue
            else:
              next_prow = prow + self.actions[i][0]
              next_pcol = pcol + self.actions[i][1]
              if hitwall(self.town,next_prow,next_pcol):
                continue
              else:
                next_s.append(self.map[(next_rrow,next_rcol,next_prow,next_pcol)])

      else: ## police is down to you
        if rcol > pcol : ### police is left down to you
          for i in range(self.n_actions):
            if i == 0 or i == 1 or i == 4:
              continue
            else:
              next_prow = prow + self.actions[i][0]
              next_pcol = pcol + self.actions[i][1]
              if hitwall(self.town,next_prow,next_pcol):
                continue
              else:
                next_s.append(self.map[(next_rrow,next_rcol,next_prow,next_pcol)])

        else: ## the police is right down to you
          for i in range(self.n_actions):
            if i == 0 or i == 2 or i == 4:
              continue
            else:
              next_prow = prow + self.actions[i][0]
              next_pcol = pcol + self.actions[i][1]
              if hitwall(self.town,next_prow,next_pcol):
                continue
              else:
                next_s.append(self.map[(next_rrow,next_rcol,next_prow,next_pcol)])
    else:
      next_s.append(self.map[(0,0,1,2)])
    
    return next_s  

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
        caught    = (self.states[s][0]==self.states[s][2]) and (self.states[s][1] == self.states[s][3])
        if caught:
          rewards[s,a] = -50
        else:
          hitting   = (next_row == self.states[s][0]) and (next_col == self.states[s][1]) and (a != 0) 

          if hitting:
            rewards[s,a] = self.IMPOSSIBLE_MOVE
            dead_flag    = True
            continue
          else:
            for pos_next_s in next_s:
              caught    = (self.states[pos_next_s][0]==self.states[pos_next_s][2]) and (self.states[pos_next_s][1] == self.states[pos_next_s][3])
              if caught:
                rewards[s,a] = -50
                dead_flag == False
          ## if the robber is in the bank 
              if (self.town[next_row,next_col]== 2) and (dead_flag == False):
                rewards[s,a] = self.IN_BANK
              elif (self.town[next_row,next_col] == 0) and dead_flag == False:
                rewards[s,a] =self.NIN_BANK         
    return rewards

  def simulate(self,start,Time,policy):
    rob_path = []
    police_path = []
    rob_path.append((start[0],start[1]))
    police_path.append((start[2],start[3]))
    s = start
    t = 0
    while t < Time - 1 :
      next_s = self.__move(self.map[s],policy[self.map[s]])
      i = random.randint(0,len(next_s)-1)
      s = next_s[i] 
      rob_path.append((self.states[s][0],self.states[s][1]))
      police_path.append((self.states[s][2],self.states[s][3]))
      s = self.states[s]
      t = t +1
    return rob_path,police_path  
def hitwall(town,row,col):
    hitting = (row == -1) or (row == town.shape[0]) or (col == -1) or (col == town.shape[1]) 
    return hitting
