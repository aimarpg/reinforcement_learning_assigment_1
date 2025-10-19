import gymnasium as gym
from deustorl.common import EpsilonGreedyPolicy, evaluate_policy
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.helpers import DiscretizedObservationWrapper
import os
import json

from deustorl.qlearning import QLearning
from deustorl.sarsa import Sarsa
from deustorl.common import *

import optuna
from optuna.samplers import TPESampler


# ============ COPIADO ============

def objective(trial):

    # Normally, algorithm selection is not part of the hyperparameter optimization
    # but this is just to show the possibility and becaouse these 3 algorithms are very similar
    algo_name = trial.suggest_categorical("algo_name",["sarsa", "esarsa", "qlearning"])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    lr_decay = trial.suggest_float("learning_rate_decay", 0.9 , 1.0,  step=0.01)
    lr_episodes_decay = trial.suggest_categorical("lr_episodes_decay",[100, 1_000, 10_000])
    discount_rate = trial.suggest_float("discount_rate", 0.8, 1.0, step=0.05)
    epsilon= trial.suggest_float("epsilon", 0.0, 0.4, step=0.05)
        	  	
    n_steps = 200_000

    if algo_name == "sarsa":
        algo = Sarsa(training_env)
    elif algo_name == "esarsa": 
        algo = ExpectedSarsa(training_env)
    elif algo_name == "qlearning": 
        algo = QLearning(training_env)

    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
    algo.learn(epsilon_greedy_policy,n_steps=n_steps, discount_rate=discount_rate, lr=lr, lrdecay=lr_decay, n_episodes_decay=lr_episodes_decay)

    avg_reward, avg_steps = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100)
    
    return avg_reward

# ==================================

ENV_NAME = 'LunarLander-v3'
N_STEPS = 3_000
CHECKPOINT_DIR = './checkpoints/'
N_EVAL_EPISODES = 10
N_BINS = 10
seed = 47

os.system("rm -rf ./logs/")
os.system("mkdir -p ./optuna/")

storage_file = f"sqlite:///optuna/optuna.db"
study_name = "lunarlander_test_1"
full_study_dir_path = f"optuna/{study_name}"
tpe_sampler = TPESampler(seed=seed) # For reproducibility
study = optuna.create_study(sampler=tpe_sampler, direction='maximize', study_name=study_name, storage=storage_file, load_if_exists=True)
n_trials = 10 # Normally 50 or 100 at least

training_env = DiscretizedObservationWrapper(gym.make(ENV_NAME), n_bins=N_BINS, tensoorboard_logdir=f"./logs/{study_name}/")
visual_env = DiscretizedObservationWrapper(gym.make(ENV_NAME, render_mode="human"), n_bins=N_BINS)


print(f"Searching for the best hyperparameters in {n_trials} trials...")
study.optimize(objective, n_trials=n_trials)

training_env.close()
visual_env.close()

# ======================= Otro copy paste abajo =======================

best_trial = study.best_trial

# Generate the policy_kwargs key before writing to file
best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)

# Save the data in a JSON file
os.system(f"mkdir -p {full_study_dir_path}")
best_trial_file = open(f"{full_study_dir_path}/best_trial.json", "w")
best_trial_file.write(best_trial_params)
best_trial_file.close()

# Generate the improtant figures of the results
fig = optuna.visualization.plot_optimization_history(study)
fig.write_html(f"{full_study_dir_path}/optimization_history.html")
fig = optuna.visualization.plot_contour(study)
fig.write_html(f"{full_study_dir_path}/contour.html")
fig = optuna.visualization.plot_slice(study)
fig.write_html(f"{full_study_dir_path}/slice.html")
fig = optuna.visualization.plot_param_importances(study)
fig.write_html(f"{full_study_dir_path}/param_importances.html")
