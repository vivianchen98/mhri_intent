import csv
import multiprocessing
import hydra
import omegaconf
import numpy as np
import tqdm
from mhri.envs.gnomes_at_night import GnomesAtNightEnv9
from mhri.algos.transition_belief import TransitionBelief
from mhri.algos.shortest_path import shortest_path_min_violation_with_belief


def run_shortest_path(env: GnomesAtNightEnv9, start_pos: tuple, treasure_pos: tuple, render=False):
    env.reset(token_pos=start_pos, treasure_pos=treasure_pos)

    state_space = [(i, j) for i in range(9) for j in range(9)]
    action_space = [0, 1, 2, 3]
    player_transition_beliefs = {
        0: TransitionBelief(state_space, action_space, env.maze_shape),
        1: TransitionBelief(state_space, action_space, env.maze_shape),
    }

    def get_shortest_path(player_id):
        path, actions_on_path, violations = shortest_path_min_violation_with_belief(
            actions={
                1: np.array([0, 1]),  # right
                2: np.array([-1, 0]),  # up
                3: np.array([0, -1]),  # left
                4: np.array([1, 0]),  # down
            },
            start_state=tuple(env.token_pos),
            final_state=tuple(env.treasure_pos),
            transitions=env.transitions[player_id],
            teammate_transition_beliefs=player_transition_beliefs[1 - player_id],
            violation_cost=10,
        )
        return path, actions_on_path, violations

    step = 0
    n_switch_turns = 0
    success = False

    terminated = False
    truncated = False

    while not (terminated or truncated) and step < 1000:
        path, actions_on_path, violations = get_shortest_path(env.current_player)
        action = actions_on_path[0]
        # With 20% chance, we take a random action
        if np.random.random() < 0.2:
            action = np.random.choice([1, 2, 3, 4])
        if action - 1 in env.transitions[env.current_player][(env.token_pos[0], env.token_pos[1])]:
            gym_action = {
                "control": action - 1,
                "comm": "",
            }
            player_transition_beliefs[env.current_player].update_with_trajectory(
                [((env.token_pos[0], env.token_pos[1]), action - 1)]
            )
        else:
            gym_action = {"control": 4, "comm": ""}
            n_switch_turns += 1

        obs, rew, terminated, truncated, info = env.step(gym_action)
        if rew > 0:
            success = True
        if render:
            env.render(mode="human")
        step += 1

    return step, success, n_switch_turns


def run_experiment(parameters):
    maze_id, start_pos, treasure_pos, trial_id = parameters
    env = GnomesAtNightEnv9(
        render_mode="rgb_array",
        trajectory_dump_path=None,
        train_mazes=[maze_id],
        communication_type="text",
    )
    step, success, n_switch_turns = run_shortest_path(env, start_pos, treasure_pos)
    result = {
        "start_pos": start_pos,
        "treasure_pos": treasure_pos,
        "trial_id": trial_id,
        "step": step,
        "success": success,
        "n_switch_turns": n_switch_turns,
    }
    return result


@hydra.main(version_base=None, config_path="config", config_name="mcts")
def main(cfg: omegaconf.DictConfig) -> None:
    experiment_results = []

    starting_positions = []
    treasure_positions = []
    trial_ids = list(range(cfg.n_trials))
    for i in range(9):
        for j in range(9):
            starting_positions.append((i, j))
            treasure_positions.append((i, j))

    experiment_parameters = []
    for start_pos in starting_positions:
        for treasure_pos in treasure_positions:
            for trial_id in trial_ids:
                if start_pos != treasure_pos:
                    experiment_parameters.append((cfg.maze_id, start_pos, treasure_pos, trial_id))

    print(f"Running {len(experiment_parameters)} experiments...")
    experiments = []
    for i in range(cfg.n_trials):
        experiments.append(experiment_parameters[i :: cfg.n_trials])

    for i in range(cfg.n_trials):
        print(f"Running trial {i + 1} of {cfg.n_trials}...")
        with multiprocessing.Pool(processes=cfg.n_parallel) as pool:
            experiment_results = list(
                tqdm.tqdm(
                    pool.imap(run_experiment, experiments[i]),
                    total=len(experiments[i]),
                )
            )

        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        hydra_output_path = hydra_cfg["runtime"]["output_dir"]
        with open(f"{hydra_output_path}/results_{i}.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=experiment_results[0].keys())
            writer.writeheader()
            writer.writerows(experiment_results)


if __name__ == "__main__":
    main()
