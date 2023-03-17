import numpy as np
import random

class QLearningModel:
    """
    Defines online Q-learning setup and update rule for a given gridworld
    """
    
    def __init__(self, g, alpha):
        self.gamma = g.gamma
        Q_dims = [g.gridworld_length, g.gridworld_width, g.num_orientations,
                  3, 3, 3, 
                  g.getNumActions()]
        self.Q = np.zeros(Q_dims)
        self.alpha = alpha
        self.actions = g.actions[0]
        self.g = g
        
    def lookahead(self, s, a):
        g = self.g
        s = g.state_to_ind(s)
        a = g.action_to_ind(a)
        Q_index = s + [a]
        return self.Q[tuple(Q_index)]
    
    def update(self, s, a, r, s_prime):
        g = self.g
        s = g.state_to_ind(s)
        a = g.action_to_ind(a)
        s_prime = g.state_to_ind(s_prime)
        Q_index = s + [a]
        self.Q[tuple(Q_index)] += self.alpha * (r + self.gamma * max(self.Q[tuple(s_prime)]) - self.Q[tuple(Q_index)])


class ValueIterationModel:
    """
    Defines an offline policy based on value iteration
    """

    def __init__(self, g, residual = 0.001, maxIter = 1000):
        self.res = residual
        self.maxIter = maxIter
        self.converged = False
        self.g = g
        self.actions = g.actions[0]
        self.gamma = g.gamma
        self.U_dims = [g.gridworld_length, g.gridworld_width, g.num_orientations]
        self.U = np.zeros(self.U_dims)
        self.policy = np.zeros(self.U_dims)       

    def value_update(self):
        counter = 0
        while not self.converged:
            counter += 1
            maxRes = 0        
            printer = 0    
            A = self.g.actions[0]
            for i in range(self.U_dims[0]):
                for j in range(self.U_dims[1]):
                    for o in range(self.U_dims[2]):
                        self.g.state[:3] = [i,j,o]
                        self.g.state[3:] = self.g.getSurroundingMarkers()
                        old_U = self.U[i,j,o]
                        old_state = self.g.state
                        #Now you have self.g.state iterating through every x,y,o and the associated surrounding markers
                        #The U and policy would be a list of length 4 for each x,y

                        #I have replaced val with self.U[i,j,o], is that correct?
                        self.U[i,j,o] = max((self.g.takeAction(a, old_state) + self.gamma*(self.g.failChance*self.U[i,j,o] + (1-self.g.failChance)*self.U[tuple(self.g.state[:3])])) for a in A)
                        
                        if maxRes <= abs(self.U[i,j,o] - old_U):
                            maxRes = abs(self.U[i,j,o] - old_U)
                        

            #print(maxRes)

            """
            for s_ind, val in np.ndenumerate(self.U):
                self.g.state = list(s_ind)
                old_U = self.U[s_ind]
                self.U[s_ind] = max((self.g.takeAction(a, self.g.state) + 
                                     self.gamma*(self.g.failChance*val + (1-self.g.failChance)*self.U[tuple(self.g.state)])) for a in A)
                if maxRes <= abs(self.U[s_ind] - old_U):
                    maxRes = abs(self.U[s_ind] - old_U)
            print(maxRes)
            """
            if maxRes <= self.res:
                self.converged = True
            if counter == self.maxIter:
                break
        return self.converged

    def lookahead(self, s, a):
        self.g.state = s
        value = self.g.takeAction(a) + self.gamma*(self.g.failChance*self.U[tuple(s[:3])] + (1-self.g.failChance)*self.U[tuple(self.g.state[:3])])
        return value


class Policy:
    """
    Defines a policy for a given model
    """
    
    def __init__(self, g):
        self.g = g
        self.isModelUpdate = False
        
    def next_action(self, s):
        raise NotImplementedError("You must implement this method")
    
    def greedy_action(self, model, s):
        list_temp = []
        for a_i in range(len(self.g.actions[0])):
            a = self.g.actions[0][a_i]
            list_temp.append([model.lookahead(s, a) for a in self.g.actions[0]])
        max_val = max(list_temp)
        # choose randomly from the best possible actions in the event of a tie
        candidate_actions = [self.g.actions[0][a_i] for a_i in range(len(self.g.actions[0])) if list_temp[a_i] == max_val]
        next_a = random.choice(candidate_actions)
            
        return next_a


class RandomPolicy(Policy):
    """
    Defines a random policy
    """
    
    def __init__(self, g):
        Policy.__init__(self, g)
    
    def next_action(self, model, s):
        return random.choice(self.g.actions[0])

class GreedyPolicy(Policy):
    """
    Helper class to apply a learned policy
    """
    def __init__(self, g):
        Policy.__init__(self, g)
    
    def next_action(self, model, s):
        return self.greedy_action(model, s)
        
        
class EpsilonGreedyExploration(Policy):
    """
    Defines exploration policy with specified parameter epsilon and decay rate alpha
    """

    def __init__(self, g, epsilon, alpha=1):
        Policy.__init__(self, g)
        self.isModelUpdate = True
        self.epsilon = epsilon
        self.alpha = alpha
    
    def next_action(self, model, s):
        A = model.actions
        if np.random.uniform() < self.epsilon:
            next_a = random.choice(model.actions)
        else:
            next_a = self.greedy_action(model, s)
        self.epsilon *= self.alpha
        return next_a
    
