import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# some colors
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

def reward_matrix(env,monatour_state,weights= None, random_rewards = None,monatour_stand = False):
  """ Calculate the reward_matrix at every time t
      return a reward matrix dimension s*a
  """
  rewards = np.zeros((env.n_states,env.n_actions))
  for s in range(env.n_states):
    for a in range(env.n_actions):
      rewards[s,a] = env.rewards(env.states[s], a , monatour_state , weights,random_rewards,monatour_stand)
  return rewards

def dynamic_programming(env,horizon,monatour_stand=False):
  """ Solves the shortest path problem with a monatour random walk using dynamic
      programming
      : input Maze env : The maze environment in which we seek to
      : input int horizon : The time T up to which we solve the problem.
      : return numpy.array V : Optimal values for every state at every time 
                               dimension S*T
      : return numpy.array policy : Optimal time-varying policy at every state.
                                    dimension S*T            

  """
  p = env.transition_probabilities
    ## we need to calculate r every time step
  n_states = env.n_states
  n_actions = env.n_actions
  T = horizon
  V = np.zeros((n_states,T+1))
  policy = np.zeros((n_states, T+1))
  Q = np.zeros((n_states,n_actions))
  # Initialization
  Q = reward_matrix(env,env.monatour_Path[-1],monatour_stand)
  V[:,T] = np.max(Q,1)
  policy[:,T] = np.argmax(Q,1)
  # The dynamic programming 
  for t in range(T-1 , -1, -1):
    # update the reward r
    r   = reward_matrix(env,env.monatour_Path[t],monatour_stand)
    # Update the value function acccording to the bellman equation
    for s in range(n_states):
      for a in range(n_actions):
          # Update of the temporary Q values
        Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
      # Update by taking the maximum Q value w.r.t the action a
    V[:,t] = np.max(Q,1)
    # The optimal action is the one that maximizes the Q function
    policy[:,t] = np.argmax(Q,1)
  return V,policy


## draw maze
def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};
    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];
    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);
    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];
    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))
    # Create a table to color
    grid = plt.table(cellText=None, cellColours=colored_maze, cellLoc='center',loc=(0,0),edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);
        
def animate_solution(maze,path,m_path):
    col_map ={0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}
    # size of the maze
    rows,cols = maze.shape
    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))
    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])
    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];
    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))
    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)
    flag = True
    # Update the color at each frame
    for i in range(len(path)):
        if flag == False:
            break
        grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i])].get_text().set_text('Player')
        grid.get_celld()[(m_path[i])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(m_path[i])].get_text().set_text('Minotaur')
        if i >= 0 :
            grid.get_celld()[(m_path[i-1])].set_facecolor(col_map[maze[m_path[i-1]]])
            grid.get_celld()[(m_path[i-1])].get_text().set_text('')
            grid.get_celld()[(path[i-1])].set_facecolor(col_map[maze[path[i-1]]])
            grid.get_celld()[(path[i-1])].get_text().set_text('')
            
            if  maze[path[i]] == 2 :
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i])].get_text().set_text('Player is out')
                grid.get_celld()[(m_path[i])].set_facecolor(LIGHT_PURPLE)
                grid.get_celld()[(m_path[i])].get_text().set_text('Minotaur')
                flag = False
            elif path[i] == path[i-1] and maze[path[i]] != 2 and path[i] != m_path[i]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(path[i])].get_text().set_text('Player')
                grid.get_celld()[(m_path[i])].set_facecolor(LIGHT_PURPLE)
                grid.get_celld()[(m_path[i])].get_text().set_text('Minotaur')
                flag = True
            elif path[i] == m_path[i]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(path[i])].get_text().set_text('got caught')
                flag = False
            else:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(path[i])].get_text().set_text('Player')
                grid.get_celld()[(m_path[i])].set_facecolor(LIGHT_PURPLE)
                grid.get_celld()[(m_path[i])].get_text().set_text('Minotaur')
                flag = True
            display.display(fig)
            display.clear_output(wait = True)
            time.sleep(1)
