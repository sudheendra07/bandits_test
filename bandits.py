import numpy as np
import matplotlib.pyplot as plt
from operator import add
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

def play_ucb(arms, t):
    c = 2
    if t < arms.no_arms:
        idx = t
    else:
        conf_int = [(c * np.sqrt(np.log(t)/n_t)) for n_t in arms.count_pulls]
        arms.ucb_est = list(map(add, arms.mean_est, conf_int))
        idx = np.argmax(arms.ucb_est)
    
    reward = np.random.uniform(arms.mean[idx]-arms.half[idx], arms.mean[idx]+arms.half[idx])
    try:
        arms.count_pulls[idx] += 1
        arms.mean_est[idx] += (1/arms.count_pulls[idx]) * (reward - arms.mean_est[idx])
    except:
        print(idx)
    
    return reward

class arms_unif:
    def __init__(self, mean, half):
        self.mean = mean
        self.half = half
        self.no_arms = len(mean)
        self.mean_est = [0]*self.no_arms
        self.count_pulls = [0]*self.no_arms
        self.ucb_est = [0]*self.no_arms

    def reset(self):
        self.count_pulls = [0]*self.no_arms
        self.mean_est = [0]*self.no_arms

arms_mean = [-0.3, 0, 0.2, 0.3]
arms_unif_half = [1, 1, 1, 1]
num_arms = len(arms_mean)
epsilon = 0.1

horizon_length = 1000
avg_reward = np.zeros(horizon_length+1)
avg_reward_ucb = np.zeros(horizon_length+1)

arms = arms_unif(mean=arms_mean, half=arms_unif_half)

for i in range(horizon_length):
    chance = np.random.uniform(0, 1)
    if chance < epsilon:
        reward = play_random(arms)
    else:
        reward = play_greedy(arms)

    avg_reward[i+1] = avg_reward[i] + (1/(i+1)) * (reward - avg_reward[i])

arms.reset()

for i in range(horizon_length):
    reward = play_ucb(arms, i+1)
    avg_reward_ucb[i+1] = avg_reward_ucb[i] + (1/(i+1)) * (reward - avg_reward_ucb[i])

plt.plot(avg_reward, 'g', label = f'E-G (eps = {epsilon})')
plt.plot(avg_reward_ucb, 'r', label = f'UCB (c = 2)')
plt.title('Epsilon-Greedy v/s UCB Bandit Algorithm')
plt.xlabel('Steps')
plt.ylabel('Average Cumulative Reward')
plt.legend()
plt.show()