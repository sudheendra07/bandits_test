import numpy as np
import matplotlib.pyplot as plt
from operator import add

def play_random(arms):
    # randomly sample arm with epsilon probability
    idx = np.random.randint(len(arms.mean))
    reward = np.random.uniform(arms.mean[idx]-arms.half[idx], arms.mean[idx]+arms.half[idx])
    arms.count_pulls[idx] += 1
    arms.mean_est[idx] += (1/arms.count_pulls[idx]) * (reward - arms.mean_est[idx])
    return reward, idx

def play_greedy(arms):
    # choose greedy arm with (1-epsilon) probability
    idx = np.argmax(arms.mean_est)
    reward = np.random.uniform(arms.mean[idx]-arms.half[idx], arms.mean[idx]+arms.half[idx])
    arms.count_pulls[idx] += 1
    arms.mean_est[idx] += (1/arms.count_pulls[idx]) * (reward - arms.mean_est[idx])
    return reward, idx

def play_ucb(arms, t, c):
    # play arms in round robin initially, then apply UCB to choose arm
    if t < arms.no_arms:
        idx = t
    else:
        conf_int = [(c * np.sqrt(np.log(t+1)/n_t)) for n_t in arms.count_pulls]
        arms.ucb_est = list(map(add, arms.mean_est, conf_int))
        idx = np.argmax(arms.ucb_est)
    
    reward = np.random.uniform(arms.mean[idx]-arms.half[idx], arms.mean[idx]+arms.half[idx])
    
    arms.count_pulls[idx] += 1
    arms.mean_est[idx] += (1/arms.count_pulls[idx]) * (reward - arms.mean_est[idx])
    
    return reward, idx

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

# modelling the arms as uniform dist with given mean and half-lengths
num_arms = 10
arms_mean = [np.random.uniform(-0.5, 0.5) for _ in range(num_arms)]
arms_unif_half = [0.5]*num_arms
best_arm = np.argmax(arms_mean)

# epsilon-greedy parameter
epsilon = 0.1

horizon_length = 10000
avg_reward = np.zeros(horizon_length+1)
avg_reward_optimistic = np.zeros(horizon_length+1)
avg_reward_ucb = np.zeros(horizon_length+1)

perc_best_arm = np.zeros(horizon_length+1)
perc_best_arm_optimistic = np.zeros(horizon_length+1)
perc_best_arm_ucb = np.zeros(horizon_length+1)

# initialize arms class
arms = arms_unif(mean=arms_mean, half=arms_unif_half)

# Epsilon greedy

best_arm_count = 0
for i in range(horizon_length):
    chance = np.random.uniform(0, 1)
    if chance < epsilon:
        reward, arm = play_random(arms)
    else:
        reward, arm = play_greedy(arms)

    if arm == best_arm:
        best_arm_count += 1

    avg_reward[i+1] = avg_reward[i] + (1/(i+1)) * (reward - avg_reward[i])
    perc_best_arm[i+1] = (best_arm_count*100/(i+1))

# reset arms mean estimation and pull count from E-G
arms.reset()

# Optimistic initial values
init_opt_est = 1
arms.mean_est = [init_opt_est]*arms.no_arms
best_arm_count = 0

for i in range(horizon_length):
    chance = np.random.uniform(0, 1)
    if chance < epsilon:
        reward, arm = play_random(arms)
    else:
        reward, arm = play_greedy(arms)

    if arm == best_arm:
        best_arm_count += 1

    avg_reward_optimistic[i+1] = avg_reward_optimistic[i] + (1/(i+1)) * (reward - avg_reward_optimistic[i])
    perc_best_arm_optimistic[i+1] = (best_arm_count*100/(i+1))

# reset arms mean estimation and pull count from optimistic E-G
arms.reset()

# UCB parameter
c = 1
best_arm_count = 0

for i in range(horizon_length):
    reward, arm = play_ucb(arms, i, c)

    if arm == best_arm:
        best_arm_count += 1

    avg_reward_ucb[i+1] = avg_reward_ucb[i] + (1/(i+1)) * (reward - avg_reward_ucb[i])
    perc_best_arm_ucb[i+1] = (best_arm_count*100/(i+1))

plt.subplot(1, 2, 1)
plt.plot(avg_reward, 'g', label = f'E-G (eps = {epsilon})')
plt.plot(avg_reward_optimistic, 'b', label = f'E-G Opt (M0 = {init_opt_est})')
plt.plot(avg_reward_ucb, 'r', label = f'UCB (c = {c})')
plt.title(f'E-G v/s UCB, Best reward = {max(arms_mean)}')
plt.xlabel('Steps')
plt.ylabel('Average Cumulative Reward')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(perc_best_arm, 'g', label = f'E-G (eps = {epsilon})')
plt.plot(perc_best_arm_optimistic, 'b', label = f'E-G Opt (M0 = {init_opt_est})')
plt.plot(perc_best_arm_ucb, 'r', label = f'UCB (c = {c})')
plt.title(f'Arms means = {[ round(elem, 2) for elem in arms_mean ]}')
plt.xlabel('Steps')
plt.ylabel('Percentage best arm picked')
plt.legend()

plt.show()