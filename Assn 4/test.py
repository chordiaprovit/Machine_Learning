from mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning
import mdptoolbox.util as util
import mdptoolbox.example
import numpy as np
import matplotlib.pyplot as plt


class Stocks(object):

    """Defines the stock trading MDP."""

    def __init__(self, N=7):
        self.A = 3  # 3 actions
        self.N = N  # N stock states

    def transitions(self):
        """Create transition matrices."""
        shape = (self.N + 1, self.N + 1)
        transitions = []

        # Define for action "buy"
        matrix = np.identity(self.N + 1)  # buy is the same as "hold"
        matrix[0, 0] = 0
        matrix[0, (self.N + 1) // 2] = 1  # always buy "average" stock
        util.checkSquareStochastic(matrix)
        transitions.append(matrix)

        # Define for action "hold". Basically a random walk.
        matrix = np.zeros(shape)
        matrix[0, 0] = 1
        matrix[1, 0:2] = 1. / 2
        for i in range(2, self.N):
            for j in range(i - 1, i + 2, 1):
                matrix[i, j] = 1. / 3
        matrix[self.N, self.N - 1:self.N + 1] = 1. / 2
        util.checkSquareStochastic(matrix)
        transitions.append(matrix)

        # Define for action "sell"
        matrix = np.zeros(shape)
        matrix[:, 0] = 1  # always reset to initial state, i.e. no stock
        util.checkSquareStochastic(matrix)
        transitions.append(matrix)

        return transitions

    def rewards(self, rval=None):
        """Define rewards matrix."""

        # Define reward values
        if rval is None:
            rval = [-1, -0.05, 3]
        shape = (self.N + 1, self.A)
        rewards = np.zeros(shape)

        # Define for action "buy". Always incur some purchase cost
        # greater than the opportunity cost of holding.
        rewards[:, 0] = rval[0]

        # Define for action "hold". Always incur opportunity cost.
        rewards[:, 1] = rval[1]

        # Define rewards for selling.
        xmin = -(self.N - 1) // 2
        xmax = -xmin + 1
        rewards[1:, 2] = [rval[2] ** x for x in range(xmin, xmax, 1)]
        rewards[0, 2] = -999  # Never do this

        return rewards


class Maze(object):

    """The Maze MDP."""

    def __init__(self, maze=None, goal=None, theseus=None, minotaur=None):
        # The pre-defined maze
        if maze is None:
            self.maze = np.asarray(
                [[0, 1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 1, 0, 1, 0],
                 [1, 0, 0, 0, 0, 1, 0],
                 [0, 0, 1, 1, 1, 1, 0],
                 [0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 1, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0]])
        else:
            self.maze = maze
        self.X = self.maze.shape[0]
        self.Y = self.maze.shape[1]

        # Goal position
        if goal is None:
            self.goal = (0, 0)
        else:
            self.goal = goal

        # Theseus starting position
        if theseus is None:
            self.theseus = (5, 3)
        else:
            assert self.maze[theseus] == 0
            assert theseus != self.goal
            self.theseus = theseus

        # Minotaur starting position
        if minotaur is None:
            self.minotaur = (5, 1)
        else:
            assert self.maze[minotaur] == 0
            self.minotaur = minotaur

    def transitions(self):
        """Return transition matrices, one for each action."""

        # Initialize transition matrices
        shape = self.maze.shape
        num_states = (shape[0] * shape[1]) ** 2
        matrix_size = (4, num_states, num_states)
        T = np.zeros(matrix_size)

        # All possible positions on the map
        pos = [(i, j) for i in range(shape[0]) for j in range(shape[1])]

        # For every pair of positions, get transition probabilities
        pos2 = ((theseus, minotaur) for theseus in pos for minotaur in pos)
        for theseus, minotaur in pos2:
            # Get Theseus's new positions (deterministic)
            theseus_next = self.get_moves(theseus)

            # Get Minotaur's possible new positions (random)
            minotaur_next = self.get_moves(minotaur)

            # Update transition probabilities for each action matrix
            current_state = self.global_state(theseus, minotaur)
            for a in range(4):
                # Get current and next states
                next_states = [self.global_state(theseus_next[a], M)
                               for M in minotaur_next]
                # Update transition probabilities
                for ns in next_states:
                    T[a, current_state, ns] += 0.25

        # "Reset" to initial state when meeting minotaur.
        initial_state = self.global_state(self.theseus, self.minotaur)
        for p in pos:
            # All states where Theseus and minotaur are co-located
            current_state = self.global_state(p, p)

            # Reset to initial state after minotaur is guaranteed
            T[:, current_state, :] = 0
            T[:, current_state, initial_state] = 1

            # All states where Theseus is at the goal
            goal_state = self.global_state(self.goal, p)

            # Reset to initial state after goal is guaranteed
            T[:, goal_state, :] = 0
            T[:, goal_state, initial_state] = 1

        # Confirm stochastic matrices
        for a in range(4):
            util.checkSquareStochastic(T[a])
        return T

    def rewards(self, rval=None):
        """Returns reward matrix."""

        # Define reward values
        if rval is None:
            # Reward for goal, penalty for minotaur, step penalty
            self.rval = [1, -1, -0.01]
        else:
            self.rval = rval

        # Initialize rewards matrix with step penalty
        shape = self.maze.shape
        num_states = (shape[0] * shape[1]) ** 2
        R = np.ones((num_states, 4)) * self.rval[2]

        # All possible positions on the map
        pos = [(i, j) for i in range(shape[0]) for j in range(shape[1])]

        # Positions adjacent to goal and the goal-reaching action. Adjacent
        # positions found by moving from the goal without rebounds off walls.
        # Optimal action for an adjacent position is the opposite of the
        # action which led from the goal to the adjacent position. E.g, if
        # "South" led to position X, then optimal action for X is "North".
        # penultimate = [(s, (a + 2) % 4)
        #                for a, s in enumerate(self.get_moves(self.goal))
        #                if s != self.goal]

        # Reward for taking a goal-reaching action, no matter where the
        # minotaur is going to be. Do not consider case where minotaur
        # sits at goal, or collides with Theseus.
        # for adj, a_star in penultimate:
        #     for minotaur in pos:
        #         R[self.global_state(adj, minotaur), a_star] = self.rval[0]

        # Reward for being in the goal state, no matter where the minotaur is.
        for p in pos:
            R[self.global_state(self.goal, p), :] = self.rval[0]

        # Penalty for Theseus sharing the same state as the minotaur, no
        # matter what action is taken next :(
        for p in pos:
            R[self.global_state(p, p), :] = self.rval[1]

        return R

    def get_moves(self, pos):
        """Get result of N, E, S, W moves."""
        x = pos[0]
        y = pos[1]
        moves = []
        # Check North
        if (x > 0) and (self.maze[x - 1, y] == 0):
            moves.append((x - 1, y))
        else:
            moves.append((x, y))
        # Check East
        if (y < self.Y - 1) and (self.maze[x, y + 1] == 0):
            moves.append((x, y + 1))
        else:
            moves.append((x, y))
        # Check South
        if (x < self.X - 1) and (self.maze[x + 1, y] == 0):
            moves.append((x + 1, y))
        else:
            moves.append((x, y))
        # Check West
        if (y > 0) and (self.maze[x, y - 1] == 0):
            moves.append((x, y - 1))
        else:
            moves.append((x, y))

        return moves

    def local_state(self, pos):
        """Convert position on map to a local state."""
        state = (self.Y * pos[0] + pos[1])
        return state

    def global_state(self, pos1, pos2):
        """Convert a pair of positions on map to a unique state."""
        num_local_states = (self.X * self.Y)
        state1 = self.local_state(pos1)
        state2 = self.local_state(pos2)
        global_state = state1 * num_local_states + state2
        return global_state

    def get_pos(self, state):
        """Recover pair of positions from global state."""
        # Get local states
        num_local_states = (self.X * self.Y)
        state2 = state % num_local_states
        state1 = (state - state2) / num_local_states

        # Convert local states to positions
        pos1 = [0, 0]
        pos1[1] = state1 % self.Y
        pos1[0] = (state1 - pos1[1]) / self.Y

        pos2 = [0, 0]
        pos2[1] = state2 % self.Y
        pos2[0] = (state2 - pos2[1]) / self.Y

        return tuple(pos1), tuple(pos2)

    def visualize(self, state):
        print (1)

    def visualize_policy(self, state, policy):
        pass

    def unit_test_global_state(self):
        """Test proper functionality of global_state()."""
        height = self.X
        width = self.Y
        pos = [(i, j) for i in range(height) for j in range(width)]
        global_states = [self.global_state(pos1, pos2)
                         for pos1 in pos
                         for pos2 in pos]
        total = (height ** 2) * (width ** 2)  # Expected state size
        assert len(set(global_states)) == total
        assert set(global_states).difference(set(range(total))) == set()


def mini_maze():
    """Creates toy version of the Maze MDP."""
    maze = np.asarray([[0, 0],
                       [0, 1]])
    goal = (0, 0)
    theseus = (0, 1)
    minotaur = (1, 0)
    M = Maze(maze=maze, goal=goal, theseus=theseus, minotaur=minotaur)

    return M


def unit_test_Maze():
    """Test proper functionality of Maze()."""
    maze = np.asarray([[0, 0],
                       [0, 1]])
    goal = (0, 0)
    theseus = (0, 1)
    minotaur = (1, 0)

    M = Maze(maze=maze, goal=goal, theseus=theseus, minotaur=minotaur)
    print ( "Transitions")
    print ( M.transitions())
    print ( "\nRewards")
    print ( M.rewards())


def example():
    """Run the MDP Toolbox forest example."""
    transitions, rewards = mdptoolbox.example.forest()
    viter = ValueIteration(transitions, rewards, 0.9)
    viter.run()
    print ( viter.policy)


def solve_stocks(N=7):
    """Solve the Stocks MDP."""
    tmp = Stocks(N)
    discount = 0.9
    T = tmp.transitions()
    R = tmp.rewards()

    viter = ValueIteration(T, R, discount)
    viter.run()
    print ( "\nValue iteration: {}".format(viter.policy))
    print ( "# of iterations: {}".format(viter.iter))
    print ( "Execution time: {}".format(viter.time))

    piter = PolicyIteration(T, R, discount)
    piter.run()
    print ( "\nPolicy iteration: {}".format(piter.policy))
    print ( "# of iterations: {}".format(piter.iter))
    print ( "Execution time: {}".format(piter.time))

    qlearn = QLearning(T, R, discount, n_iter=200000)
    qlearn.run()
    print ( "\nQ-learning: {}".format(qlearn.policy))
    #print ( "\nQ: \n{}".format(qlearn.Q)
    print ( "# of iterations: {}".format(qlearn.max_iter))
    print ( "Execution time: {}".format(qlearn.time))

    return viter, piter, qlearn


def solve_maze():
    """Solves the Maze aka Theseus and the Minotaur MDP."""
    M = Maze()
    T = M.transitions()
    R = M.rewards()
    discount = 0.9

    viter = ValueIteration(T, R, discount)
    viter.run()
    print ( "\nValue iteration: {}".format(viter.policy))
    print ( "# of iterations: {}".format(viter.iter))
    print ( "Execution time: {}".format(viter.time))

    piter = PolicyIteration(T, R, discount, max_iter=2000)
    piter.run()
    print ( "\nPolicy iteration: {}".format(piter.policy))
    print ( "# of iterations: {}".format(piter.iter))
    print ( "Execution time: {}".format(piter.time))

    qlearn = QLearning(T, R, discount, n_iter=10000)
    qlearn.run()
    print ( "\nQ-learning: {}".format(qlearn.policy))
    print ( "# of iterations: {}".format(qlearn.max_iter))
    print ( "Execution time: {}".format(qlearn.time))

    return viter, piter, qlearn


def solve_mini_maze():
    """Solve miniature Maze MDP."""
    M = mini_maze()
    T = M.transitions()
    R = M.rewards()
    discount = 0.9

    viter = ValueIteration(T, R, discount)
    viter.run()
    print ( "\nValue iteration: {}".format(viter.policy))
    print ( "# of iterations: {}".format(viter.iter))
    print ( "Execution time: {}".format(viter.time))

    piter = PolicyIteration(T, R, discount, max_iter=2000)
    piter.run()
    print ( "\nPolicy iteration: {}".format(piter.policy))
    print ( "# of iterations: {}".format(piter.iter))
    print ( "Execution time: {}".format(piter.time))

    qlearn = QLearning(T, R, discount, n_iter=50000)
    qlearn.run()
    print ( "\nQ-learning: {}".format(qlearn.policy))
    print ( "# of iterations: {}".format(qlearn.max_iter))
    print ( "Execution time: {}".format(qlearn.time))

    return viter, piter, qlearn


def simulate_policy(alg, mdp, N=100000):
    """Simulate the results of following a policy for a MDP."""
    T = mdp.transitions()
    R = mdp.rewards()
    P = alg.policy
    # random initial state
    state_space = range(T[0].shape[0])
    state = np.random.randint(0, T[0].shape[0])

    state_sequence = [state]
    total_reward = []
    for i in range(N):
        # Take action according to policy
        action = P[state]
        # Get reward associated with state and action
        reward = R[state, action]
        total_reward.append(reward)
        # Next state
        state = np.random.choice(state_space, p=T[action][state, :])
        state_sequence.append(state)

    return total_reward, state_sequence


def stocks_vs_state(n_states=None):
    """Compare performance on the Stocks MDP as a function of state size."""
    if n_states is None:
        n_states = [7, 9, 13, 15, 17, 23, 29, 35, 41, 53, 65, 77, 89]

    for N in n_states:
        mdp = Stocks(N)
        discount = 0.9
        T = mdp.transitions()
        R = mdp.rewards()

        viter = ValueIteration(T, R, discount)
        viter.run()
        rewards, _ = simulate_policy(viter, mdp)
        print ( "\nValue iteration: {}".format(viter.policy))
        print ( "# of iterations: {}".format(viter.iter))
        print ( "Execution time: {}".format(viter.time))
        print ( "Average reward: {}".format(np.mean(rewards)))

        piter = PolicyIteration(T, R, discount)
        piter.run()
        rewards, _ = simulate_policy(piter, mdp)
        print ( "\nPolicy iteration: {}".format(piter.policy))
        print ( "# of iterations: {}".format(piter.iter))
        print ( "Execution time: {}".format(piter.time))
        print ( "Average reward: {}".format(np.mean(rewards)))

        qlearn = QLearning(T, R, discount, n_iter=10000)
        qlearn.run()
        rewards, _ = simulate_policy(piter, mdp)
        print ( "\nQ-learning: {}".format(qlearn.policy))
        print ( "# of iterations: {}".format(qlearn.max_iter))
        print ( "Execution time: {}".format(qlearn.time))
        print ( "Average reward: {}".format(np.mean(rewards)))


def qlearning_vs_iter(iters=None):
    """Compare Q-Learning performance on the Stocks MDP as a function of
    the number of iterations."""
    nit = []
    tyme = []
    rew = []
    if iters is None:
        iters = range(10000, 20001, 1000)
    mdp = Maze()
    discount = 0.99
    T = mdp.transitions()
    R = mdp.rewards()
    for num_iters in iters:
        qlearn = QLearning(T, R, discount=discount, n_iter=num_iters)
        qlearn.run()
        rewards, _ = simulate_policy(qlearn, mdp, N=10000)
        print ( "\nIterations: {}".format(num_iters))
        print ( "Execution time: {}.".format(qlearn.time))
        print ( "Average reward: {}".format(np.mean(rewards)))
        nit.append(num_iters)
        tyme.append(qlearn.time)
        rew.append(np.mean(rewards))

    plt.plot(nit, tyme, label = "Time")
    plt.plot(nit, rew, label = "Reward")
    plt.title("Qlearning")
    plt.xlabel("Number of  Iters")
    plt.ylabel("Time and Rewards")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.grid()
    plt.show()

def main():
    """Run everything."""
    # mdps = [Stocks(), mini_maze(), Maze()]
    # mdp_names = ["Stocks MDP", "Mini maze MDP", "Theseus and the Minotaur MDP"]
    # solvers = [solve_stocks, solve_mini_maze, solve_maze]

    mdps = [Stocks(), Maze()]
    mdp_names = ["Stocks MDP", "Theseus and the Minotaur MDP"]
    solvers = [solve_stocks, solve_maze]
    alg_names = ("Value iteration", "Policy iteration", "Q-Learning")

    # For all MDP problems
    for mdp, mdp_name, solver in zip(mdps, mdp_names, solvers):
        print("-----------------------------")
        print ("mdp_name:", mdp_name)
        # Solve using 3 algorithms
        algs = solver()
        for alg, alg_name in zip(algs, alg_names):
            rewards, states = simulate_policy(alg, mdp)
            print (alg_name)
            print ( "Average reward: {}".format(np.mean(rewards)))



if __name__ == "__main__":
    stocks_vs_state()
    #qlearning_vs_iter()
    #main()