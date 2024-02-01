import numpy as np
import matplotlib.pyplot as plt
# import random

def play_random(arms):
    idx = np.random.randint(len(arms.mean))
    reward = np.random.uniform(arms.mean[idx]-arms.half[idx], arms.mean[idx]+arms.half[idx])
    arms.count_pulls[idx] += 1
    arms.mean_est[idx] += (1/arms.count_pulls[idx]) * (reward - arms.mean_est[idx])
    return reward

def play_greedy(arms):
    idx = np.argmax(arms.mean_est)
    reward = np.random.uniform(arms.mean[idx]-arms.half[idx], arms.mean[idx]+arms.half[idx])
    arms.count_pulls[idx] += 1
    arms.mean_est[idx] += (1/arms.count_pulls[idx]) * (reward - arms.mean_est[idx])
    return reward

class arms_unif:
    def __init__(self, mean, half):
        self.mean = mean
        self.half = half
        self.mean_est = [0]*len(mean)
        self.count_pulls = [0]*len(mean)

arms_mean = [-0.3, 0, 0.2, 0.3]
arms_unif_half = [1, 1, 1, 1]
num_arms = len(arms_mean)
epsilon = 0.1

horizon_length = 1000
avg_reward = np.zeros(horizon_length+1)

arms = arms_unif(mean=arms_mean, half=arms_unif_half)

mean_est = [0]*num_arms
count_pulls = [0]*num_arms



for i in range(horizon_length):
    chance = np.random.uniform(0, 1)
    if chance < epsilon:
        reward = play_random(arms)
    else:
        reward = play_greedy(arms)

    avg_reward[i+1] = avg_reward[i] + (1/(i+1)) * (reward - avg_reward[i])

plt.plot(avg_reward)
plt.title('Epsilon-Greedy Bandit Algorithm')
plt.xlabel('Steps')
plt.ylabel('Average Cumulative Reward')
plt.legend()
plt.show()