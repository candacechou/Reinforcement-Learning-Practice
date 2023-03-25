import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random

# some colors
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

def value_iteration(town,gamma,limits):
    epsilon = town.epsilon
    p       = town.transition_probabilities
    r       = town.rewards
    n_states = town.n_states
    n_actions = town.n_actions

    V      = np.zeros(n_states)
    Q      = np.zeros((n_states,n_actions))
    BV     = np.zeros(n_states)

    n = 0
    tol = 0.001

    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
    BV = np.max(Q, 1)
    while np.linalg.norm(V-BV) >= tol and n < limits:
        n +=1
        V = np.copy(BV)
        for s in range(n_states):
            for a in range(n_actions):
                Q[s,a] = r[s,a] + gamma * np.dot(p[:,s,a],V)
        BV = np.max(Q,1)

    policy = np.argmax(Q,1)

    return V, policy

def animate_solution(maze,path,m_path):
  # Map a color to each cell in the maze
    col_map ={0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # size of the maze
    rows,cols = maze.shape
    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))
    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy Simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

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
        grid.get_celld()[(path[i])].get_text().set_text('Robber')
        grid.get_celld()[(m_path[i])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(m_path[i])].get_text().set_text('Police')

        if i > 0 :
            grid.get_celld()[(m_path[i-1])].set_facecolor(col_map[maze[m_path[i-1]]])
            grid.get_celld()[(m_path[i-1])].get_text().set_text('')
            grid.get_celld()[(path[i-1])].set_facecolor(col_map[maze[path[i-1]]])
            grid.get_celld()[(path[i-1])].get_text().set_text('')
            if  maze[path[i]] == 2 and path[i] != m_path[i]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i])].get_text().set_text('Robber')
                grid.get_celld()[(m_path[i])].set_facecolor(LIGHT_PURPLE)
                grid.get_celld()[(m_path[i])].get_text().set_text('police')
                flag = False
            elif path[i] == path[i-1] and maze[path[i]] != 2 and path[i] != m_path[i]:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(path[i])].get_text().set_text('Robber')
                grid.get_celld()[(m_path[i])].set_facecolor(LIGHT_PURPLE)
                grid.get_celld()[(m_path[i])].get_text().set_text('police')
                flag = True
            elif path[i] == m_path[i]:
                grid.get_celld()[(m_path[i])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(m_path[i])].get_text().set_text('got caught')
                flag = False
            else:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
                grid.get_celld()[(path[i])].get_text().set_text('Robber')
                grid.get_celld()[(m_path[i])].set_facecolor(LIGHT_PURPLE)
                grid.get_celld()[(m_path[i])].get_text().set_text('police')
                flag = True

            display.display(fig)
            display.clear_output(wait = True)
            time.sleep(1)


def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    gp = grid.properties()
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)