from __future__ import unicode_literals
from Gridworld import Gridworld
from utils import visualize_grid, policy_score, simulate_policy

from Learning import QLearning, EpsilonGreedyExploration, FixedPolicy

g = Gridworld(gridworld_length=10, gridworld_width = 10, num_obstacles=98)      
#policy_score = simulate_policy(g, policy_type="random", run_to_completion=True)
#eps_greedy_policy =  EpsilonGreedyExploration(0.8, alpha=0.99)
#qlearning_model = QLearning(g, 0.2)

#for i in range(1000):
#    g.reset_position()
 #   simulate_policy(g, policy_type="epsilon-greedy", model=qlearning_model, policy=eps_greedy_policy, run_to_completion=False, num_iters=1000, visualize=False)

#learned_policy = FixedPolicy(qlearning_model.extract_policy(), g, model=qlearning_model)


#policy_score_learned = simulate_policy(g, policy_type="fixed", model=None, policy=learned_policy, run_to_completion=True)
#print(policy_score_learned)

policy = g.offline(gamma = 0.7, error = 0.0000001)

count0 = 0
count1 = 0
count2 = 0

for i in policy:
    if i == 0:
        count0+=1
    elif i == 1:
        count1+=1
    elif i == 2:
        count2+= 1

print(len(policy), count0,count1,count2)
