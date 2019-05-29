import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import time

def get_unwrapped_env(env):
    return env.env

def run_episode(env, policy, gamma, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """

    obs = env.reset()
    total_reward = 0.0
    step_idx = 0
    total_steps = []
    while True:
        if render:
            env.render()

        obs, reward, done , _ = env.step(int(policy[obs]))
        print(obs)
        total_reward += (gamma ** step_idx * reward)
        print("rewards", reward)
        step_idx += 1

        if done:
            #env.render()
            break

    print("----")
    return total_reward


def evaluate_policy(env, policy, gamma,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [run_episode(env, policy, gamma = gamma, render = False)
            for episode in range(n)]
    return scores

def extract_policy(v, gamma ):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.unwrapped.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma):
    """ Value-iteration algorithm """
    v = np.zeros(env.observation_space.n)  # initialize value-function
    max_iterations = 1000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            q_sa = [sum([p*(r + gamma * prev_v[s_]) for p, s_, r, _ in env.unwrapped.P[s][a]]) for a in range(env.unwrapped.nA)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            cnv_val = i+1
            print ('Value-iteration converged at iteration# %d.' %cnv_val)
            break

    return v, cnv_val


if __name__ == '__main__':
    #env_name  = 'FrozenLake-v0'
    env_name  = 'FrozenLake8x8-v0'
    #env_name = 'Taxi-v2'
    gammas = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95]
    env = gym.make(env_name)

    value = []
    times = []
    cnv_value = []
    mean_policy = []
    for gamma in gammas:
        start_time = time.time()
        optimal_v,cnv_val_iter = value_iteration(env, gamma)
        policy = extract_policy(optimal_v, gamma)
        policy_score = evaluate_policy(env, policy, gamma, n=100)
        end_time = time.time()

        value.append(optimal_v)
        cnv_value.append(cnv_val_iter)
        times.append(end_time - start_time)
        mean_policy.append(np.mean(policy_score))


    plt.plot(gammas,times,label= "time")
    plt.plot(gammas, mean_policy, label = "mean score")
    plt.title("Mean score and time with diffrent Gamma "+env_name)
    plt.legend()
    plt.xlabel("Gamma")
    plt.ylabel("Mean Score and Time")
    plt.grid()
    plt.show()

    print(policy_score)
    mean = [np.mean(policy_score)]*len(policy_score)
    plt.plot(range(100), policy_score, label = 'scores')
    plt.plot(range(100), mean, label='Mean', linestyle='--')
    plt.legend()
    plt.title("Rewards for 100 episodes (gamma=0.95) " + env_name)
    plt.grid()
    plt.show()

    x = range(len(policy))
    y = policy
    plt.plot(x, y)
    plt.xlabel("State")
    plt.ylabel("")
    plt.title('Final Policy (action) States for Gamma 0.95')
    plt.grid()
    plt.show()

    plt.plot(gammas,cnv_value)
    plt.title("Value iteration convergence with diffrent Gamma "+env_name)
    plt.xlabel("Gamma")
    plt.ylabel("Value iteration convergence")
    plt.grid()
    plt.show()