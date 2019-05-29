import numpy as np
import gym
import random
import matplotlib.pyplot as plt
import time
env_name  = 'FrozenLake8x8-v0'
env = gym.make(env_name)

action_size = env.action_space.n
print(action_size)
state_size = env.observation_space.n
print(state_size)
qtable = np.zeros((state_size, action_size))

total_episodes = 10000         # Total episodes
learning_rate = 0.90           # Learning rate
max_steps = 999                # Max steps per episode
gamma = 0.95                   # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005            # Exponential decay rate for exploration prob

# List of rewards
rewards = []
eps = []
total_step = []

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0


    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :] + np.random.randn(1, env.action_space.n) * (1. / (episode + 1)))
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()


        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)



        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] =  qtable[state, action] + learning_rate * (
                    reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])



        # Our new state is state
        state = new_state

        # If done (if we're dead) : finish episode
        if done == True:
            #env.render()
            total_step.append(step)
            break


    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    eps.append(epsilon)
    rewards.append(total_rewards)
    #print (epsilon)
    #print("--")

print(qtable)
print("-+-+-+-+-+-+-+-+-+-+-+-")
plt.plot(range(total_episodes), eps,label = 'epsilon')
plt.legend()
plt.title("Q-learning epsilon (exploration to exploitation) " +env_name)
plt.grid()
plt.show()


total_step = []
env.reset()
final_reward = []
total_rewards = []
tot_time = []
max_steps = 999
for episode in range(100):
    state = env.reset()
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)


    prize = 0
    start_time = time.time()

    for step in range(max_steps):

        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)

        prize = prize - 0.01


        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()

            prize = prize + 20
            # We print the number of step it took.
            print("Number of steps", step)
            break

        state = new_state

    end_time = time.time()
    tot_time.append(end_time-start_time)
    total_rewards.append(prize)
    total_step.append(step)
    temp_reward = []
    temp_step = []

env.close()

plt.plot(range(len(total_rewards)), total_rewards,label = 'rewards')
plt.legend()
plt.title("Q-learning FL (showing rewards at each episode) " +env_name)
plt.grid()
plt.show()

print (len(total_step))
plt.plot(range(len(total_step)), total_step, label = 'steps')
plt.legend()
plt.title("Q-learning FL (showing steps in each episode) " + env_name)
plt.grid()
plt.show()

print (len(tot_time))
plt.plot(range(len(tot_time)), tot_time, label = 'time')
plt.legend()
plt.title("Q-learning FL (time taken for each episode) " + env_name)
plt.grid()
plt.show()