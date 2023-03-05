#Utility functions for the gridworld environment
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Visualization helper functions
def chars_to_num(char_grid):
    """
    Creates a copy of a 2D character gridworld converted into a 2D array of integers
    for easy visualization in matplotlib. The mapping is as follows:
    '-' --> 0
    'A' --> 1 
    'O' --> 2
    'D' --> 3
    """
    num_grid = char_grid.copy()
    num_grid[num_grid == '-'] = 0
    num_grid[num_grid == 'A'] = 1
    num_grid[num_grid == 'O'] = 2
    num_grid[num_grid == 'D'] = 3
    num_grid = num_grid.astype('int64')
    return num_grid

def chars_to_icons(char_grid):
    """
    Creates a copy of a 2D character gridworld converted into a 2D array of integers
    for easy visualization in matplotlib. The mapping is as follows:
    '-' --> '🍂'
    'A' --> '🤖' 
    'O' --> '🌲'
    'D' --> '🧗🏽'
    """
    icon_grid = char_grid.copy()
    icon_grid[icon_grid == '-'] = '🍂'
    icon_grid[icon_grid == 'A'] = '🤖'
    icon_grid[icon_grid == 'O'] = '🌲'
    icon_grid[icon_grid == 'D'] = '🧗'
    return icon_grid

def visualize_grid(char_grid):
    """
    Visualize a 2D grid of characters in matplotlib with emojis
    """
    colors = ['saddlebrown', 'red', 'green', 'yellow']
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots()
    num_grid = chars_to_num(char_grid)
    icon_grid = chars_to_icons(char_grid)

    for y in range(char_grid.shape[0]):
       for x in range(char_grid.shape[1]):
          plt.text(x , y, icon_grid[y, x],
             horizontalalignment='center',
             verticalalignment='center',
             fontname='Segoe UI Emoji'
          )
    ax.matshow(num_grid, cmap=cmap, vmin=0, vmax=len(colors))
    
    
#policy score calculator
def policy_score(rewards, discount_factor):
    """
    Calculates the score of a policy using a list of rewards and a discount factor
    """
    score = 0
    for i in range(len(rewards)):
        score += (discount_factor**i)*rewards[i]
    return score