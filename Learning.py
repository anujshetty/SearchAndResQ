import numpy as np
import random

class QLearning:
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
        
    def extract_policy(self):
        policy_indices = np.argmax(self.Q, axis=len(self.Q.shape)-1) # take argmax along actions
        return policy_indices
        
class FixedPolicy:
    """
    Helper class to apply a learned policy
    """
    def __init__(self, policy, g, model):
        self.policy = policy
        self.g = g
        self.model = model
    
    def next_action(self, s):
        state_ind = tuple(self.g.state_to_ind(s))
        max_val = max(self.model.Q[state_ind])
        candidate_actions = [self.g.action_to_ind(a) for a in self.g.actions[0] if self.model.Q[state_ind][self.g.action_to_ind(a)] == max_val]
        next_a = random.choice(candidate_actions)
            
        return self.g.actions[0][next_a]
        
        
class EpsilonGreedyExploration:
    """
    Defines exploration policy with specified parameter epsilon and decay rate alpha
    """

    def __init__(self, epsilon, alpha=1):
        self.epsilon = epsilon
        self.alpha = alpha
    
    def next_action(self, model, s):
        A = model.actions
        if np.random.uniform() < self.epsilon:
            next_a = random.choice(model.actions)
        else:
            # choose randomly from the best possible actions in the event of a tie
            max_val = max([model.lookahead(s, a) for a in A])
            candidate_actions = [a for a in A if model.lookahead(s, a) == max_val]
            next_a = random.choice(candidate_actions)
        self.epsilon *= self.alpha
        return next_a
    
class ValueIteration:
    """
    Defines an offline policy based on value iteration
    """

    def __init__(self, g, gamma=0.9, residual = 0.0001, maxIter = 1000, search = 10):
        self.gamma = 0.9
        self.search = search
        self.res = residual
        self.maxIter = maxIter
        self.converged = False
        self.g = g
        self.U_dims = [g.gridworld_length, g.gridworld_width, g.num_orientations,3, 3, 3]
        self.U = np.zeros(U_dims)
        self.polciy = np.zeros(U_dims)
        if self.update():
            print("Converged")
        else:
            print("Not converged, max iteration reached")
        self.extract_policy()

    def update(self):
        counter = 0
        while not self.converged:
            counter += 1
            maxRes = 0
            A = self.g.actions
            for x in range[U_dims[0]]:
                for y in range[self.U_dims[1]]:
                    for o in range[self.U_dims[2]]:
                        for tl in range[self.U_dims[3]]:
                            for m in range[self.U_dims[3]]:
                                for tr in range[self.U_dims[3]]:
                                    self.g.state = [x,y,o,tl,m,tr]
                                    s = self.g.state
                                    old_U = self.U[ind]
                                    self.U[ind] = max((self.g.takeAction(s, a) + self.gamma * self.U[self.g.state]) for a in A)
                                    if maxRes <= abs(self.U[ind] - old_U):
                                        max_Res = abs(self.U[ind] - old_U)
            if max_Res <= self.res:
                self.converger = True
            if counter == self.maxIter:
                break
        return self.converged

    def extract_policy(self):
        A = self.g.actions
        for x in range[U_dims[0]]:
            for y in range[self.U_dims[1]]:
                for o in range[self.U_dims[2]]:
                    for tl in range[self.U_dims[3]]:
                        for m in range[self.U_dims[3]]:
                            for tr in range[self.U_dims[3]]:
                                self.g.state = [x,y,o,tl,m,tr]
                                s = self.g.state
                                max_U = max((self.g.takeAction(s, a) + self.gamma * self.U[self.g.state]) for a in A)
                                candiadate_actions = [a for a in A if (self.g.takeAction(s, a) + self.gamma * self.U[self.g.state]) == max_val]
                                self.policy[s] = random.choice(candidate_actions)
        return True

    def next_action(self, s):

        return self.policy[s]
