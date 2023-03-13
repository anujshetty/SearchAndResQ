# Define gridworld class

import numpy as np
import random
from operator import add

class Gridworld:
    def __init__(self, gridworld_length=10, gridworld_width=10, num_obstacles=10,
                 collisionReward= -1, destinationReward= 10, defaultReward= 0, outOfBoundsReward = -1, failChance= 0.1, gamma= 0.9):
        self.gridworld_length = gridworld_length
        self.gridworld_width = gridworld_width
        self.grid = np.zeros((gridworld_length,gridworld_width))
        self.ds_actions = {"u": [0,-1], "r": [1,0], "d": [0,1], "l": [-1,0], 
                           "tr": [0,0], "tl": [0,0]} # turn right/left
        self.actions= list(self.ds_actions.keys()),
        self.num_obstacles = num_obstacles
        self.source, self.destination, self.obstacle_positions = self.initiate_gridworld()
        self.num_orientations = 4
        # Initialize 1 of 4 orientations for agent to be facing
        orientation = random.randint(0,3)
        self.state = self.source + [orientation]
        self.state = self.state + self.getSurroundingMarkers()
        self.collisionReward = collisionReward
        self.destinationReward = destinationReward
        self.defaultReward = defaultReward
        self.failChance = failChance
        self.gamma = gamma
        self.outOfBoundsReward = outOfBoundsReward

    def getCoords(self):
        return self.state[:2]
    
    def getOrientation(self):
        return self.state[2]
    
    def randomCoords(self):
        return [random.randint(0, self.gridworld_length-1), random.randint(0, self.gridworld_width-1)]
    
    def getNumStates(self):
        return self.gridworld_length * self.gridworld_width * self.num_orientations * (3**3)
    
    def getNumActions(self):
        return len(self.actions[0])
    
    def reset_position(self):
        pos = self.randomCoords()
        while pos == self.destination or (pos in self.obstacle_positions):
            pos = self.randomCoords()
        orientation = random.randint(0,3)
        self.state = pos + [orientation]
        self.state = self.state + self.getSurroundingMarkers()

    def initiate_gridworld(self):
        # add a random source and destination to the gridworld
        source = self.randomCoords()
        destination = self.randomCoords()
        while destination == source:
            destination = self.randomCoords()

        # add some random obstacles to the gridworld, making sure that the source and destination are not obstacles
        obstacle_positions = []
        
        while len(obstacle_positions) < self.num_obstacles:
            position = self.randomCoords()
            if position != source and position != destination:
                obstacle_positions.append(position)
        return source, destination, obstacle_positions

    def getMarker(self, posn):
        if posn == self.destination:
            return 2
        if posn in self.obstacle_positions:
            return 1
        return 0
    
    # get the markers of the 3 cells
    def getSurroundingMarkers(self):
        markers = []
        x, y = self.ds_actions[self.actions[0][self.getOrientation()]]
        if x == 0:
            steps = [[-1,y], [0,y], [1,y]]
        if y == 0:
            steps = [[x,-1], [x,0], [x,1]]
        for step in steps:
            adj_posn = list(map(add, self.getCoords(), step))
            markers.append(self.getMarker(adj_posn))
        return markers
    
    def turn(self, a):
        if a == 'tr':
            self.state[2] = (self.state[2] + 1) % 4
        if a == 'tl':
            self.state[2] = (self.state[2] - 1) % 4

    def takeAction(self, a):
        # take action with probability 0.1, stay in same state with probability 0.9
        if random.random() < 1 - self.failChance:
            new_state = list(map(add, self.getCoords(), self.ds_actions[a]))
            # if turning
            if self.getCoords() == new_state:
                self.turn(a)
                return self.defaultReward
            if new_state[0] < 0 or new_state[0] >= self.gridworld_length or \
                new_state[1] < 0 or new_state[1] >= self.gridworld_width:
                return self.outOfBoundsReward
            # if collision
            if new_state in self.obstacle_positions:
                return self.collisionReward
            
            if new_state == self.destination:
                self.state[:2] = new_state
                return self.destinationReward
            else:
                temp = self.state[:2]
                self.state[:2] = new_state
                return self.defaultReward
        # if no action taken, or step taken without collision/reaching destination
        return self.defaultReward

    def print_gridworld(self):
        for row in range(self.gridworld_length):
            for col in range(self.gridworld_width):
                if [row,col] in self.obstacle_positions:
                    print('O', end=' ')
                elif [row,col] == self.destination:
                    print('D', end=' ')
                elif [row, col] == self.getCoords():
                    print('A', end=' ')
                else:
                    print('-', end=' ')
            print()
    
    def gridworld_to_arr(self):
        char_grid = np.zeros([self.gridworld_length, self.gridworld_width]).astype('<U1')
        for row in range(self.gridworld_length):
            for col in range(self.gridworld_width):
                if [row,col] in self.obstacle_positions:
                    char_grid[row, col] = 'O'
                elif [row,col] == self.destination:
                    char_grid[row, col] = 'D'
                elif [row, col] == self.getCoords():
                    char_grid[row, col] = 'A'
                else:
                    char_grid[row, col] = '-'
        return char_grid
    
    def state_to_ind(self, s):
        """
        Converts a state in list format to an index for indexing into a Q value matrix
        """
        return self.state[:]

    def offline(self, gamma = 0.7, error = 0.001):
        self.gamma = gamma
        self.num_states = self.gridworld_length*self.gridworld_width*self.num_orientations*20 #20 as only one destination possible (2^3 + 3*2^2)
        self.utility = np.zeros(self.num_states)
        self.error = error
        self.policy = np.zeros(self.num_states)

        return self.runOffline()
        
    def runOffline(self):
        while True:
            max_error = 0
            index = 0
            for x in range(self.gridworld_length):
                for y in range(self.gridworld_width):
                    for o in range(self.num_orientations):
                        for rest in range(20):
                            if rest < 8:
                                state = [x,y,o, rest//4, rest//2, rest%2]
                            elif rest < 12:
                                state = [x,y,o,2, (rest-8)//2, (rest-8)%2]
                            elif rest < 16:
                                state = [x,y,o, (rest-12)//2,2, (rest-12)%2]
                            else:
                                state = [x,y,o, (rest-16)//2, (rest-16)%2,2]

                            bellmanList = []
                            for a in range(3): #front turn left and turn right
                                bellman = 0
                                #R(s,a)
                                if state[:2] == self.destination:
                                    bellman += self.destinationReward
                                elif state[:2] in self.obstacle_positions:
                                    bellman += self.collisionReward
                                else:
                                    bellman += self.defaultReward

                                if a == 0:
                                    #new index
                                    newIndex = index
                                    if o == 0:
                                        newIndex += self.gridworld_length*self.num_orientations*20
                                    elif o == 1:
                                        newIndex += self.num_orientations*20
                                    elif o == 2:
                                        newIndex -= self.gridworld_length*self.num_orientations*20
                                    else:
                                        newIndex -= self.num_orientations*20

                                    xNew = (newIndex/(self.num_orientations*20))%self.gridworld_length
                                    yNew = (newIndex/(self.num_orientations*20))//self.gridworld_length

                                    if ([xNew, yNew] in self.obstacle_positions) or xNew < 0 or yNew < 0 or xNew >= self.gridworld_length or yNew >= self.gridworld_width:
                                        newIndex = index
                                    
                                    #gamma sum(T(s',s,a)U(s'))                                    
                                    bellman += self.gamma*self.failChance*self.utility[index]
                                    bellman += self.gamma*(1 - self.failChance)*self.utility[newIndex]

                                    bellmanList.append(bellman)

                                else:
                                    #gamma sum(T(s',s,a)U(s'))
                                    bellman += self.gamma*self.utility[index]
                                    bellmanList.append(bellman)

                            #search for maximum error
                            error = abs(self.utility[index] - max(bellmanList))
                            if error > max_error:
                                max_error = error

                            #Update
                            self.utility[index] = max(bellmanList)
                            self.policy[index] = bellmanList.index(max(bellmanList))
                            index += 1       

            #convergence check
            print(max_error,self.error)
            if self.error >= max_error:
                break               
                     
        return self.policy
            


    
    def action_to_ind(self, a):
        """
        Converts an action in string format to an index for indexing into a Q value matrix
        """
        return self.actions[0].index(a)
    
'''    def getGeneralReward(self, old, new):
        if old[0] == new[0]:
            if (self.destination[1] - old[1])*(new[1] - old[1]) >= 0:
                return self.generalReward
            else:
                return -self.generalReward
        else:
            if (self.destination[0] - old[0])*(new[0] - old[0]) >= 0:
                return self.generalReward
            else:
                return -self.generalReward
'''

'''
###########################################################
############# SOLVING OFFLINE VALUE ITERATION #############
###########################################################
'''

    
