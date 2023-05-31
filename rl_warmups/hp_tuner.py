import optuna

from rl_warmups.sarsa import Params, main


def create_study():
    return optuna.create_study(
        study_name="test-study-1",
        storage="sqlite:///hpopt.db",
        direction="maximize",
        load_if_exists=True,
    )


def objective(trial: optuna.Trial):
    params = Params(
        global_num_steps=75000,
        # seed=trial.suggest_int("seed", 0, 3),
        # learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        # batch_size=trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
        # num_replays=trial.suggest_categorical("num_replays", [1, 2, 4, 8]),
        # gamma=trial.suggest_float("gamma", 0.1, 1., step=0.01),
        # tau=trial.suggest_categorical("tau", [1., 0.1, 0.01, 0.001, 0.0001]),
        layer_init=trial.suggest_categorical("init_method", ["orthogonal_", "xavier_normal_"]),
    )
    return main(params, trial=trial)


if __name__ == "__main__":
    study = create_study()
    study.optimize(objective, n_trials=1)
