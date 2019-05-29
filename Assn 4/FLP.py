import numpy as np
import gym
import matplotlib.pyplot as plt
import time

def run_episode(env, policy, gamma, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    start_time = time.time()
    while True:
        if render:
            env.render()
        obs, reward, done , prob = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1

        if done:
            # print(step_idx)
            # env.render()
            break

    return total_reward

def evaluate_policy(env, policy, gamma, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]

    return scores

def extract_policy(v, gamma):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.unwrapped.nS)
    for s in range(env.unwrapped.nS):
        q_sa = np.zeros(env.unwrapped.nA)
        for a in range(env.unwrapped.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.unwrapped.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s]
    and solve them to find the value function.
    """
    v = np.zeros(env.unwrapped.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.unwrapped.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.unwrapped.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.unwrapped.nA, size=(env.unwrapped.nS))  # initialize a random policy

    max_iterations = 10000
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy,gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        # print(old_policy_v)
        # print(new_policy)
        # print("-----")
        if (np.all(policy == new_policy)):
            cnv_val = i + 1
            print ('Policy-Iteration converged at step %d.' %cnv_val)
            break
        policy = new_policy
    return policy, cnv_val


if __name__ == '__main__':
    env_name  = 'FrozenLake-v0'

    env = gym.make(env_name)
    gammas = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95]

    value = []
    times = []
    cnv_value = []
    mean_policy = []

    for gamma in gammas:
        start_time = time.time()
        optimal_policy,cnv_val_iter = policy_iteration(env, gamma)

        print("optimal policy", optimal_policy)
        scores = evaluate_policy(env, optimal_policy, gamma)
        end_time = time.time()

        cnv_value.append(cnv_val_iter)
        mean_policy.append(np.mean(scores))
        times.append(end_time - start_time)

    print (scores)
    plt.plot(gammas,times,label= "time")
    plt.plot(gammas, mean_policy, label = "mean score")
    plt.title("'Policy Iter' Mean score and time with diffrent Gamma "+env_name)
    plt.legend()
    plt.xlabel("Gamma")
    plt.ylabel("Mean Score and Time")
    plt.grid()
    plt.show()

    plt.plot(gammas,cnv_value)
    plt.title("Policy iteration convergence with diffrent Gamma "+env_name)
    plt.xlabel("Gamma")
    plt.ylabel("Policy iteration convergence")
    plt.grid()
    plt.show()

    x = range(len(optimal_policy))
    y = optimal_policy
    plt.plot(x, y)
    plt.xlabel("State")
    plt.ylabel("")
    plt.title('Final Policy (action) States for Gamma 0.95 ' + env_name)
    plt.grid()
    plt.show()

    mean = [np.mean(scores)]*len(scores)
    plt.plot(range(100), scores, label = 'scores')
    plt.plot(range(100), mean, label='Mean', linestyle='--')
    plt.legend()
    plt.title("Rewards for 100 episodes (gamma=0.95) " + env_name)
    plt.grid()
    plt.show()



