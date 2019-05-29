import gym
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Environment and Q-table structure
env = gym.make('FrozenLake8x8-v0')
Q = np.zeros([env.observation_space.n,env.action_space.n])
# env.obeservation.n, env.action_space.n gives number of states and action in env loaded
# 2. Parameters of Q-leanring
eta = .628
gma = .9
epis = 5000
rev_list = [] # rewards per episode calculate
# 3. Q-learning Algorithm

for i in range(epis):
    # Reset environment
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    # The Q-Table learning algorithm
    while d != True:
        env.render()
        # Choose action from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # Get new state & reward from environment
        s1, r, d, _ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + eta * (r + gma * np.max(Q[s1, :]) - Q[s, a])
        s = s1
        rev_list.append(r)
# Code will stop at d == True, and render one state before it
print('rewards ',rev_list)

print ("Reward Sum on all episodes " + str(sum(rev_list)/epis))
print ("Final Values Q-Table")
print (Q)

plt.plot(range(len(rev_list)), rev_list,label = 'rewards')
plt.legend()
plt.title("Q-learning FL (showing rewards at each episode)")
plt.grid()
plt.show()