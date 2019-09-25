import numpy as np
import matplotlib.pyplot as plt
from rl687.environments.gridworld import Gridworld


def problemA():
    print("PROBLEM A...")
    episodes = 10000
    arr = np.zeros(episodes)
    G = Gridworld()
    G.gamma = 0.9
    for e in range(episodes):  # number of episodes loop
        G.timeStep = 0
        # print("episode %d" % (e+1))
        while(not G.isEnd):
            # print(G.currentState)
            G.step(G.action)
        arr[e] = G.reward
        G.reset()

    opt_disc_returns = np.amax(arr)
    opt_episode = np.argmax(arr) + 1
    mean = np.mean(arr)
    variance = np.var(arr)
    std_dev = np.std(arr)
    min = np.amin(arr)
    print("Highest observed discounted returns is %f achieved in"
          " episode number %d" % (opt_disc_returns, opt_episode))
    print("The mean of discounted returns is %f, variance is %f"
          " and standard deviation is %f" % (mean, variance, std_dev))
    print("Max is %f and min is %f" % (opt_disc_returns, min))
    return arr


def problemB():
    print("PROBLEM B...")
    print("OPTIMAL POLICY IS")
    print("|AR | AR | AR | AR | AD|")
    print("|AU | AR | AR | AR | AD|")
    print("|AU | AL | XX | AR | AD|")
    print("|AU | AL | XX | AR | AD|")
    print("|AU | AL | AR | AR | GG|")
    print("The optimal policy is not unique. Any action taken in the Goal state"
            " (terminal state) does not affect the discounted  returns. "
                    "So  multiple  optimal  policies  exist."
                            "   (Read  this  onPiazza.  Sorry)")


def problemC():
    print("PROBLEM C...")
    policy = np.array([3, 3, 3, 3, 1,
                       0, 3, 3, 3, 1,
                       0, 2, 4, 3, 1,
                       0, 2, 4, 3, 1,
                       0, 2, 3, 3, 4])
    episodes = 10000
    arr = np.zeros(episodes)
    G = Gridworld()
    G.gamma = 0.9
    for e in range(episodes):
        G.timestep = 0
#        print("episode %d" % (e+1))
        while(not G.isEnd):
            #  print(G.currentState)
            G.step(G.stoch_action(policy[G.state]))
        arr[e] = G.reward
        G.reset()
#        arr[e] = disc_returns
    opt_disc_returns = np.amax(arr)
    opt_episode = np.argmax(arr) + 1
    mean = np.mean(arr)
    variance = np.var(arr)
    std_dev = np.std(arr)
    min = np.amin(arr)
    print("Highest observed discounted returns is %f achieved in"
          " episode number %d" % (opt_disc_returns, opt_episode))
    print("The mean of discounted returns is %f, variance is %f"
          " and standard deviation is %f" % (mean, variance, std_dev))
    print("Max is %f and min is %f" % (opt_disc_returns, min))
#    print(np.argmin(arr) + 1)
    return arr


def problemD(x1, x2):
    print("PROBLEM D...")
#   PLOTS FOR RANDOM POLICY
    y1 = np.arange(1, len(x1)+1)/len(x1)
    plt.subplot(2, 2, 1)
    plt.plot(x1, y1 * 100, marker='.', linestyle='none')
    plt.xlabel('Discounted Returns')
    plt.ylabel('Percentile of episodes')
    plt.title('Empirical CDF of random policy')
    plt.margins(0.02)
    plt.subplot(2, 2, 2)
    plt.plot(y1 * 100, x1, marker='.', linestyle='none')
    plt.xlabel('Percentile of episodes')
    plt.ylabel('Discounted Returns')
    plt.title('Empirical Quantile of random policy')
    plt.margins(0.02)

#   PLOTS FOR OPTIMAL POLICY
    y2 = np.arange(1, len(x2)+1)/len(x2)
    plt.subplot(2, 2, 3)
    plt.plot(x2, y2 * 100, marker='.', linestyle='none')
    plt.xlabel('Discounted Returns')
    plt.ylabel('Percentile of episodes')
    plt.title('Empirical CDF of optimal policy')
    plt.margins(0.02)
    plt.subplot(2, 2, 4)
    plt.plot(y2 * 100, x2, marker='.', linestyle='none')
    plt.xlabel('Percentile of episodes')
    plt.ylabel('Discounted Returns')
    plt.title('Empirical Quantile of optimal policy')
    plt.margins(0.02)
    plt.show()


def problemE():
    print("PROBLEM E...")
    episodes = 10000
    count = 0
    G = Gridworld(startState=19)
    G.gamma = 0.9
    for e in range(episodes):
        G.timeStep = 0
        while((G.timeStep < 11)and (not G.isEnd)):
            G.step(G.action)
        if G.state == 22:
            count = count + 1
        G.reset()
    print("The empirical probability of S19 = 21 given S8 = 18 is %f" % (count / episodes))


def main():
    print("AMEY KASAR. SEED USED IS 0")
    np.random.seed(0)
    x1 = problemA()
    problemB()
    x2 = problemC()
    problemE()
    problemD(np.sort(x1), np.sort(x2))


main()
