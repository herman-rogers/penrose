import random as rand

import numpy as np

class QLearner(object):
    def __init__(
            self,
            num_states=100,
            num_actions=4,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
    ):
        """
        Constructor method
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.state = 0
        self.action = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q_table = np.zeros(shape=(num_states, num_actions))
        self.hallucinations = np.empty((0, 4))

    def querysetstate(self, s):
        """
        Update the state without updating the Q-table
        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """
        action = self.q_table[s].argmax()

        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.num_actions - 1)

        if self.verbose:
            print(f"s = {s}, a = {action}")

        self.set_current_state(s, action)
        return action

    """
    Update the Q table and return an action
    :param s_prime: The new state
    :type s_prime: int
    :param r: The immediate reward
    :type r: float
    :return: The selected action
    :rtype: int
    """

    # Update Algorithm:
    # Q'[s,a] = (1 - alpha)Q[s,a] + alpha(r + Gamma * later rewards)
    # later rewards = Q[s', argmaxa'(Q[s', a'])]
    def query(self, s_prime, r):

        self.q_table[self.state, self.action] = self.get_improved_q(s_prime, r)

        if self.dyna > 0:
            updated_reward = (1 - self.alpha) * r + (self.alpha * r)
            visited = np.array([self.state, self.action, s_prime, updated_reward])
            self.hallucinations = np.append(self.hallucinations, np.array([visited]), axis=0)

            for i in range(self.dyna):
                index = rand.randint(0, len(self.hallucinations) - 1)
                h_prime = self.hallucinations[index]
                prev_state = int(h_prime[0])
                prev_action = int(h_prime[1])
                state_prime = int(h_prime[2])
                reward = h_prime[3]

                a_prime = self.q_table[state_prime].argmax()

                prev_learned = (1 - self.alpha) * self.q_table[prev_state, prev_action] + self.alpha * (
                        reward + self.gamma * self.q_table[state_prime, a_prime])
                self.q_table[prev_state, prev_action] = prev_learned

        action = self.q_table[s_prime].argmax()
        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.num_actions - 1)

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")

        self.set_current_state(s_prime, action)
        self.rar = self.rar * self.radr

        return action

    def get_improved_q(self, state, reward):
        a_prime = self.q_table[state].argmax()
        return (1 - self.alpha) * self.q_table[self.state, self.action] + self.alpha * (
                reward + self.gamma * self.q_table[state, a_prime])

    def set_current_state(self, state, action):
        self.state = state
        self.action = action

    def author(self):
        return "hrogers34"


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")