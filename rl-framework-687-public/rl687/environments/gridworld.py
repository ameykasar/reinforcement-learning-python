import numpy as np
from .skeleton import Environment


class Gridworld(Environment):
    """
    The Gridworld as described in the lecture notes of the 687 course material.

    Actions: up (0), down (1), left (2), right (3)

    Environment Dynamics: With probability 0.8 the robot moves in the specified
        direction. With probability 0.05 it gets confused and veers to the
        right -- it moves +90 degrees from where it attempted to move, e.g.,
        with probability 0.05, moving up will result in the robot moving right.
        With probability 0.05 it gets confused and veers to the left -- moves
        -90 degrees from where it attempted to move, e.g., with probability
        0.05, moving right will result in the robot moving down. With
        probability 0.1 the robot temporarily breaks and does not move at all.
        If the movement defined by these dynamics would cause the agent to
        exit the grid (e.g., move up when next to the top wall), then the
        agent does not move. The robot starts in the top left corner, and the
        process ends in the bottom right corner.

    Rewards: -10 for entering the state with water
            +10 for entering the goal state
            0 everywhere else



    """

    def __init__(self, startState=0, endState=24, shape=(5, 5),
                 obstacles=[12, 17], waterStates=[6, 18, 22]):
        """
        inittialize
        """
        self.startState = startState
        self.state = startState
        self.endState = endState
        self.shape = shape
        self.obstacles = obstacles
        self.waterStates = waterStates
        self.timeStep = 0
        self.reward = 0

    @property
    def name(self):
        """
        returns name of the world
        """
        return("GRIDWORLD")

    @property
    def action(self):
        """
        generates and returns an action
        """
        # random.random or random.uniform or np.random.random or random.choice
        gen_action = np.random.choice(4, 1)[0]
        return self.stoch_action(gen_action)

    def stoch_action(self, gen_action):
        """
        for non deterministic/stochastic actions
        """
        stoch_choice = np.random.choice(4, 1, p=[0.8, 0.05, 0.05, 0.1])[0]
        if stoch_choice == 0:           # intended action 0.8
            return gen_action
        elif stoch_choice == 1:         # veers right 0.05
            if gen_action == 0:
                return 3
            elif gen_action == 1:
                return 2
            elif gen_action == 2:
                return 0
            elif gen_action == 3:
                return 1
        elif stoch_choice == 2:         # veers left 0.05
            if gen_action == 0:
                return 2
            elif gen_action == 1:
                return 3
            elif gen_action == 2:
                return 1
            elif gen_action == 3:
                return 0
        elif stoch_choice == 3:     # breaks down
            return 4

    def step(self, a):
        s = self.state
        if a == 0:   # AU
            if (s - 5) < 0:  # Hit upper wall
                pass
            elif (s - 5) in self.obstacles:  # obstacle
                pass
            else:
                self.state = s - 5  # next state
        elif a == 1:  # AD
            if (s + 5) > 24:
                pass
            elif (s + 5) in self.obstacles:
                pass
            else:
                self.state = s + 5
        elif a == 2:   # AL
            if s in [0, 5, 10, 15, 20]:
                pass
            elif (s - 1) in self.obstacles:
                pass
            else:
                self.state = s - 1
        elif a == 3:   # AR
            if s in [4, 9, 14, 19, 25]:
                pass
            elif (s + 1) in self.obstacles:
                pass
            else:
                self.state = s + 1
        elif a == 4:  # STAY
            pass
        self.reward = self.reward + ((self.gamma ** self.timeStep) * self.R(self.state))
        self.timeStep = self.timeStep + 1

    @property
    def isEnd(self):
        if self.state == self.endState:
            return True

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, s):
        self.__state = s

    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, g):
        self.__gamma = g

    def reset(self):
        self.__init__(self.startState)

    @property
    def reward(self):
        return self.__reward

    @reward.setter
    def reward(self, r):
        self.__reward = r

    def R(self, s):
        """
        reward function

        output:
         reward resulting in the agent being in a particular state
        """
        if s in self.waterStates:
            return -10
        elif s == self.endState:
            return 10
        else:
            return 0
