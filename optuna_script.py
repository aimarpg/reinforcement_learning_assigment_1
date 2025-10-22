import os
import gymnasium as gym
import random
import optuna
from optuna.samplers import TPESampler
import json

# from gymnasium.envs.toy_text.frozen_lake import generate_random_map  # unused; you can remove
from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.qlearning import QLearning
from deustorl.helpers import DiscretizedObservationWrapper
from optuna.trial import TrialState

def make_objective(env, fixed_algo_name):
    def objective(trial):
        # Hyperparameter configurations
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        lr_decay = trial.suggest_float("learning_rate_decay", 0.9 , 1.0, step=0.01)
        lr_episodes_decay = trial.suggest_categorical("lr_episodes_decay", [50, 100, 500, 1_000, 10_000])
        discount_rate = trial.suggest_float("discount_rate", 0.8, 1.0, step=0.05)
        epsilon = trial.suggest_float("epsilon", 0.0, 0.4, step=0.05)
        n_steps = 200_000

        if fixed_algo_name == "sarsa":
            algo = Sarsa(env)
        elif fixed_algo_name == "esarsa":
            algo = ExpectedSarsa(env)
        elif fixed_algo_name == "qlearning":
            algo = QLearning(env)
        else:
            raise ValueError(f"Unknown algorithm: {fixed_algo_name}")

        epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
        algo.learn(
            epsilon_greedy_policy,
            n_steps=n_steps,
            discount_rate=discount_rate,
            lr=lr,
            lrdecay=lr_decay,
            n_episodes_decay=lr_episodes_decay,
        )

        avg_reward, avg_steps = evaluate_policy(algo.env, algo.q_table, max_policy, n_episodes=100)

        trial.set_user_attr("algo_name", fixed_algo_name)
        return avg_reward
    return objective

if __name__ == "__main__":
    env_name = "LunarLander-v3"
    env = DiscretizedObservationWrapper(gym.make(env_name), n_bins=10)
    seed = 47
    random.seed(seed)
    env.reset(seed=seed)

    # Prepare output directories
    os.system("rm -rf ./logs/")
    os.system("mkdir -p ./optuna/")

    storage_file = "sqlite:///optuna/optuna.db"
    n_trials_per_algo = 100

    for algo_name in ["sarsa", "esarsa", "qlearning"]:
        study_name = f"{env_name}_{algo_name}"
        full_study_dir_path = f"optuna/{study_name}"
        os.system(f"mkdir -p {full_study_dir_path}")

        tpe_sampler = TPESampler(seed=seed)
        study = optuna.create_study(
            sampler=tpe_sampler,
            direction="maximize",
            study_name=study_name,
            storage=storage_file,
            load_if_exists=True,
        )

        print(f"Searching for the best hyperparameters for {algo_name} in {n_trials_per_algo} trials...")
        study.optimize(make_objective(env, algo_name), n_trials=n_trials_per_algo)

        completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:3]
        best_trials = []

        for i, trial in enumerate(sorted_trials):
            trial_info = {
                "rank": i+1, 
                "number": trial.number,
                "value": trial.value, 
                "parameters": trial.params
            }
            best_trials.append(trial_info)
            with open(f"{full_study_dir_path}/best_trial.json", "w") as f:
                f.write(json.dumps(trial_info, sort_keys=True, indent=4))

        best_trial = study.best_trial
        best_params = dict(best_trial.params)
        best_params["algo_name"] = algo_name
        with open(f"{full_study_dir_path}/best_trial.json", "w") as f:
            f.write(json.dumps(best_trials, sort_keys=True, indent=4))

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"{full_study_dir_path}/optimization_history.html")
        fig = optuna.visualization.plot_contour(study)
        fig.write_html(f"{full_study_dir_path}/contour.html")
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(f"{full_study_dir_path}/slice.html")
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"{full_study_dir_path}/param_importances.html")

    env.close()