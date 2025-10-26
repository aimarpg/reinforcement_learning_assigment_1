import os
import json

from deustorl.common import EpsilonGreedyPolicy, evaluate_policy, max_policy
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.helpers import DiscretizedObservationWrapper
from deustorl.qlearning import QLearning
from deustorl.sarsa import Sarsa

import gymnasium as gym
import random
import multiprocessing

N_TRAINING_STEPS = 1_000_000

# In the form {"algo_name": [list of best hyperparameters dicts]}
best_trials = {}

# Load best trials from JSON files
for folder in os.listdir('./optuna/'):
    best_trial_path = os.path.join('./optuna/', folder, 'best_trial.json')
    if not os.path.exists(best_trial_path):
        continue

    algo_best_trials = []
    algo_name = folder.split('_')[-1]

    with open(best_trial_path, 'r') as f:
        data = f.read()
        trials = json.loads(data)

        for trial in trials:
            algo_best_trials.append(trial['parameters'])

    best_trials[algo_name] = algo_best_trials


def train_with_hyperparameters(algo_name, hyperparameters):
    # Placeholder for the actual training logic
    print(f"Training {algo_name} with hyperparameters: {hyperparameters}")

    env = DiscretizedObservationWrapper(gym.make("LunarLander-v3"), n_bins=10)
    seed = 47
    random.seed(seed)
    env.reset(seed=seed)

    match algo_name:
        case "sarsa":
            algo = Sarsa(env)
        case "esarsa":
            algo = ExpectedSarsa(env)
        case "qlearning":
            algo = QLearning(env)
        case _:
            raise ValueError(f"Unknown algorithm name: {algo_name}")
        
    mapped_hyperparameters = {
        "discount_rate": round(hyperparameters["discount_rate"], 8),
        "lr": round(hyperparameters["learning_rate"], 8),
        "lrdecay": round(hyperparameters["learning_rate_decay"], 8),
        "n_episodes_decay": int(hyperparameters["lr_episodes_decay"])
    }

    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=hyperparameters["epsilon"])
    algo.learn(epsilon_greedy_policy, n_steps=N_TRAINING_STEPS, **mapped_hyperparameters, tb_epsode_period=1000)

    avg_reward, avg_steps = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=20)
    print(f"Evaluation results for {algo_name} with hyperparameters {hyperparameters}:")
    print(f"  Average Reward: {avg_reward}")
    print(f"  Average Steps: {avg_steps}")
    print()

os.system("rm -rf ./logs/")
def train_wrapper(args):
    algo_name, params, rank = args
    print(f"Training {algo_name} (Rank {rank+1}) with hyperparameters: {params}")
    train_with_hyperparameters(algo_name, params)


""" 
for algo_name, trials in best_trials.items():
    for i, params in enumerate(trials):
        train_with_hyperparameters(algo_name, params)
    print() """

# Train each algorithm with its best hyperparameters for longer
if __name__ == "__main__":
    tasks = []
    for algo_name, trials in best_trials.items():
        for i, params in enumerate(trials):
            tasks.append((algo_name, params, i))
        print()
    
    with multiprocessing.Pool(3) as pool:
        pool.map(train_wrapper, tasks)

