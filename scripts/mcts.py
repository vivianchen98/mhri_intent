import csv
import multiprocessing
import hydra
import omegaconf
import numpy as np
import tqdm
from mhri.envs.gnomes_at_night import GnomesAtNightEnv9
from mhri.algos.transition_belief import TransitionBelief
from mhri.algos.mcts_intent import GameState, GameDynamics, MCTS
from mhri.algos.shortest_path import shortest_path_min_violation_with_belief


def run_mcts(env: GnomesAtNightEnv9, intent_mode: str, start_pos: tuple, treasure_pos: tuple, render=False):
    env.reset(token_pos=start_pos, treasure_pos=treasure_pos)

    state_space = [(i, j) for i in range(9) for j in range(9)]
    action_space = [0, 1, 2, 3]
    player_transition_beliefs = {  # Player (1 - i)'s belief over player i's transitions
        0: TransitionBelief(state_space, action_space, env.maze_shape),
        1: TransitionBelief(state_space, action_space, env.maze_shape),
    }

    player_dynamics = {
        0: GameDynamics(
            player_id=0,
            player_transitions=env.transitions[0],
            teammate_transitions_belief=player_transition_beliefs[1],
            teammate_intent=[],  # The intent received from the teammate
            intent_mode=intent_mode,
        ),
        1: GameDynamics(
            player_id=1,
            player_transitions=env.transitions[1],
            teammate_transitions_belief=player_transition_beliefs[0],
            teammate_intent=[],  # The intent received from the teammate
            intent_mode=intent_mode,
        ),
    }

    def get_shortest_path(player_id):
        path, actions_on_path, violations = shortest_path_min_violation_with_belief(
            actions={
                1: np.array([0, 1]),  # right
                2: np.array([-1, 0]),  # up
                3: np.array([0, -1]),  # left
                4: np.array([1, 0]),  # down
            },
            start_state=tuple(env.token_pos.tolist()),
            final_state=tuple(env.treasure_pos.tolist()),
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
        current_state = GameState(tuple(env.token_pos.tolist()), tuple(env.treasure_pos.tolist()), env.current_player)
        mcts = MCTS(
            initial_state=current_state,
            game_dynamics=player_dynamics[env.current_player],
            n_iterations=100,
            exploration_constant=np.sqrt(2),
            gamma=0.99,
        )
        action = mcts.search()
        gym_action = {
            "control": action,
            "comm": "",
        }

        if action != 4:  # Not SWITCH_TURN, update belief
            player_transition_beliefs[env.current_player].update_with_trajectory([(current_state.token_pos, action)])
            if intent_mode == "single_step":  # Single step intent is only effective for one step
                # clear intent when the other player has acted
                player_dynamics[1 - env.current_player].teammate_intent = []
        else:  # SWITCH_TURN, generate intent
            if intent_mode != "none":
                intent, _, _ = get_shortest_path(env.current_player)
                player_dynamics[1 - env.current_player].teammate_intent = intent
            n_switch_turns += 1

        # step simulator
        obs, reward, terminated, truncated, info = env.step(gym_action)
        if reward > 0:
            success = True
        if render:
            env.render(mode="human")
        step += 1

    return step, success, n_switch_turns


# player_transition_beliefs[0].render_belief_heatmap()


def run_experiment(parameters):
    maze_id, intent_mode, start_pos, treasure_pos, trial_id = parameters
    env = GnomesAtNightEnv9(
        render_mode="rgb_array",
        trajectory_dump_path=None,
        train_mazes=[maze_id],
        communication_type="text",
    )
    step, success, n_switch_turns = run_mcts(env, intent_mode, start_pos, treasure_pos)
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
    assert cfg.intent_mode in ["none", "single_step", "fixed", "length_inverse", "discounted"]

    experiment_results = []

    starting_positions = []
    treasure_positions = []
    trial_ids = list(range(cfg.n_trials))
    for i in range(9):
        for j in range(9):
            starting_positions.append((i, j))
            treasure_positions.append((i, j))

    # Create all combinations of experiment parameters consisting of a starting position, a treasure position, and a trial id
    experiment_parameters = []
    for start_pos in starting_positions:
        for treasure_pos in treasure_positions:
            for trial_id in trial_ids:
                if start_pos != treasure_pos:
                    experiment_parameters.append((cfg.maze_id, cfg.intent_mode, start_pos, treasure_pos, trial_id))

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

        # Save the results to a CSV file
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        hydra_output_path = hydra_cfg["runtime"]["output_dir"]
        with open(f"{hydra_output_path}/results_{i}.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=experiment_results[0].keys())
            writer.writeheader()
            writer.writerows(experiment_results)


if __name__ == "__main__":
    main()

    # Render one experiment
    # intent_mode = "discounted"
    # env = GnomesAtNightEnv9(
    #     render_mode="rgb_array",
    #     trajectory_dump_path=None,
    #     train_mazes=["0000"],
    #     communication_type="text",
    # )
    # step, success, n_switch_turns = run_mcts(env, intent_mode, start_pos=None, treasure_pos=None, render=True)
