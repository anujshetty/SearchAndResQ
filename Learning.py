import numpy as np
import random

class QLearning:
    """
    Defines online Q-learning setup and update rule for a given gridworld
    """
    
    def __init__(self, g, alpha):
        self.gamma = g.gamma
        Q_dims = [g.gridworld_length, g.gridworld_width, g.num_orientations, 3, 3, 3, g.getNumActions()]
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
        #print("Q index:", Q_index)
        self.Q[tuple(Q_index)] += self.alpha * (r + self.gamma * max(self.Q[tuple(s_prime)]) - self.Q[tuple(Q_index)])
        
        #print(f"Updated Q value of state {s}, action {g.actions[0][a]} to {self.Q[tuple(Q_index)]} using reward {r}")
        
    def extract_policy(self):
        policy_indices = np.argmax(self.Q, axis=6) # take argmax along actions
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
        if np.all(self.model.Q[tuple(self.g.state_to_ind(s))] == 0):
            print("taking random action")
            return random.choice(self.g.actions[0])
        else:
            print(self.model.Q[tuple(self.g.state_to_ind(s))])
        return self.g.actions[0][self.policy[tuple(self.g.state_to_ind(s))]]
        
        
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
    
        
     
            
    