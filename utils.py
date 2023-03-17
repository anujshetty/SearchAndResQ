#Utility functions for the gridworld environment
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from IPython import display
import random
import copy
import numpy as np

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
    '-' --> '-'
    'A' --> 'A' 
    'O' --> 'X'
    'D' --> 'D'
    """
    icon_grid = char_grid.copy()
    icon_grid[icon_grid == '-'] = '-'
    icon_grid[icon_grid == 'A'] = 'A'
    icon_grid[icon_grid == 'O'] = 'X'
    icon_grid[icon_grid == 'D'] = 'D'
    return icon_grid

def visualize_grid(g):
    """
    Visualize a 2D grid of characters in matplotlib with emojis
    """
    char_grid = g.gridworld_to_arr()
    colors = ['saddlebrown', 'red', 'green', 'yellow']
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots()
    num_grid = chars_to_num(char_grid)
    icon_grid = chars_to_icons(char_grid)
    orientations = [0, 90, 180, 270]   
    
    for y in range(char_grid.shape[0]):
        for x in range(char_grid.shape[1]):

            if icon_grid[y, x] == 'A':
                plt.text(x , y, f" {icon_grid[y, x]} ",
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=orientations[g.getOrientation()],
                    rotation_mode='anchor',
                    #fontname='Segoe UI Emoji'
                ) 
            else:
                plt.text(x , y, icon_grid[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    #fontname='Segoe UI Emoji'
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

def plot_scores(scores, batch_size=1000, xlabel=None, ylabel="Average score"):
    """
    Plots the scores of a policy over time
    """
    avg_scores = np.mean(np.array(scores).reshape(-1, batch_size), axis=1)
    plt.plot(range(len(avg_scores)), avg_scores)
    if xlabel is None:
        xlabel = f"Batch number ({batch_size} episodes per batch)"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def heatmap(array):
    # Array can be Q or U, we take max over all but the first 2 state variables
    axes = tuple(range(-1,-len(array.shape)+1,-1)) 
    array_max = np.max(array, axis=axes)
    plt.imshow(array_max, cmap='hot', interpolation='nearest')
    plt.show()

def simulate_policy(g, run_to_completion=True, num_steps=0, policy=None, model=None, visualize=True):
    """
    Simulates a run of specified policy in a given gridworld g for either num_steps steps or until target is reached.
    Returns discounted sum of rewards for the input policy.
    """
    rewards=[]
    def simulate_step(rewards):
        if visualize:
            visualize_grid(g)
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.2)
        orig_state = copy.deepcopy(g.state)
        action = policy.next_action(model, orig_state)
        # take the action and update the state
        reward = g.takeAction(action)
        rewards.append(reward)
        # update model if applicable
        if policy.isModelUpdate:
            model.update(orig_state, action, reward, g.state)

    while g.getCoords() != g.destination:
        simulate_step(rewards)
        if not run_to_completion and len(rewards) >= num_steps:
            break
    if visualize:
        plt.close("all")
    
    return policy_score(rewards, g.gamma)
 
