#Utility functions for the gridworld environment
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from IPython import display
import random

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

            if icon_grid[y, x] == '🤖':
                plt.text(x , y, f" {icon_grid[y, x]} ",
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation=orientations[g.getOrientation()],
                    rotation_mode='anchor',
                    fontname='Segoe UI Emoji'
                ) 
            else:
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

def simulate_policy(g, policy_type, run_to_completion=True, num_iters=0, policy=None, model=None, visualize=True):
    """
    Simulates a run of specified policy in a given gridworld g for either num_iters iterations or until target is reached.
    Returns discounted sum of rewards for the input policy.
    Policy type can either be "random" (which auto-generates actions) or "fixed" (which uses the provided policy dictionary)
    policy dictionary should contain mappings (s) -> (a) given states
    """
    rewards=[]
    def simulate_iteration(rewards):
        if visualize:
            visualize_grid(g)
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.2)
        orig_state = g.state[:]
        # choose a random action
        #print("Current State:", g.state)
        if policy_type=="random":
            action = random.choice(g.actions[0])
        elif policy_type=="epsilon-greedy":
            action = policy.next_action(model, orig_state)
        elif policy_type=="fixed":
            action = policy.next_action(orig_state)
        #print(f'Taking action: {action}')
        # take the action and update the state
        reward = g.takeAction(action)
        #print("Action: ", action, "Reward: ", reward)
        rewards.append(reward)
        
        # update model if applicable
        if model:
            model.update(orig_state, action, reward, g.state)

    if run_to_completion:
        while g.getCoords() != g.destination:
            simulate_iteration(rewards)
    else:
        for _ in range(num_iters):
            simulate_iteration(rewards)
    plt.close("all")
    return policy_score(rewards, g.gamma)
 