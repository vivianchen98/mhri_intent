import numpy as np
import gymnasium as gym
from gymnasium import spaces

# import pygame
import os
import json
import pickle
import copy
from mhri.envs.comm_signal import (
    CommunicationSignal,
    DiscreteSignal,
    ContinuousSignal,
    TextSignal,
    ImageSignal,
)


class GnomesAtNightEnv9(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "text"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        train_mazes=["0000", "0001", "0002", "0003", "0004", "0005"],
        trajectory_dump_path=None,
        communication_type="continuous",
    ):
        self.n_agents = 2
        self.current_player = 0

        self.renderer = None

        # declare None for global variables
        self.maze_id = None
        self.round = None
        self.size = None
        self.maze_shape = None
        self.wall_trim = None
        self.n_additional_walls = None
        self.one_hot_walls = None
        self.treasure_pos = None

        # Load the training mazes
        self.train_mazes = train_mazes
        self.randomize()  # Randomly select a maze and round

        # Define position of token
        self.token_pos = np.array((0, 0), dtype=int)

        # Define the communication signal
        self.communication_type = communication_type
        if communication_type == "discrete":
            # discrete communication signals = {0: null, 1: "stay put", 2: "move right", 3: "move up", 4: "move left", 5: "move down"}
            communication_signals: CommunicationSignal = DiscreteSignal(6)
        elif communication_type == "continuous":
            communication_signals: CommunicationSignal = ContinuousSignal(8)
        elif communication_type == "text":
            communication_signals: CommunicationSignal = TextSignal(1024)
        else:
            raise ValueError(f"Invalid communication type: {communication_type}")

        # Define the observation space: (149, 2) array: |current_player|+|token_pos|+|walls|+|treasure_pos|+|comm signal| = 1+2+8*9*2+2 =149 for each player
        # obs_low = np.array([0] + [0, 0] + [0] * 144 + [-1, -1] + [0], dtype=np.int32)
        # obs_high = np.array([1] + [8, 8] + [1] * 144 + [8, 8] + [5], dtype=np.int32)
        obs_low = np.array([0] + [0, 0] + [0] * 144 + [-1, -1], dtype=np.int32)
        obs_high = np.array([1] + [8, 8] + [1] * 144 + [8, 8], dtype=np.int32)

        observation_space = {}
        for i in range(self.n_agents):
            observation_space[f"agent_{i}"] = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.int32)
        observation_space["comm"] = communication_signals.space

        self.observation_space = spaces.Dict(observation_space)

        self.actions_map = {0: "right", 1: "up", 2: "left", 3: "down", 4: "switch"}
        control_action_space = spaces.Discrete(5)

        self.action_space = spaces.Dict(
            {
                "control": control_action_space,
                "comm": communication_signals.space,
            }
        )

        # flag to indicate if just switched turn
        self.just_switched_turn = False

        if self.communication_type == "discrete":
            self.comm_signals = 0
        elif self.communication_type == "continuous":
            self.comm_signals = np.zeros(8)
        elif self.communication_type == "text":
            self.comm_signals = ""

        # Define the mapping from actions to directions
        self._action_to_direction = {
            0: np.array([0, 1]),  # right
            1: np.array([-1, 0]),  # up
            2: np.array([0, -1]),  # left
            3: np.array([1, 0]),  # down
        }

        # translate the walls in environment as transitions
        self.transitions = [self.walls_to_transitions(self.one_hot_walls[i]) for i in range(self.n_agents)]

        # set the render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.trajectory_dump_path = trajectory_dump_path
        self.trajectory = None
        self.traj_index = None

    """ Game dynamics: turn-based, two-player, cooperative control """

    def step(self, action):
        control_action, communication_signal = action["control"], action["comm"]

        assert control_action in range(5), f"action={control_action}, should be in [0, 1, 2, 3, 4]"
        if self.communication_type == "discrete":
            assert communication_signal in range(
                6
            ), f"communication_signal={communication_signal}, should be in [0, 1, 2, 3, 4, 5]"
        elif self.communication_type == "continuous":
            assert communication_signal.shape == (8,), f"communication_signal={communication_signal} has wrong shape"
        elif self.communication_type == "text":
            assert (
                len(communication_signal) <= 1024
            ), f"communication_signal={communication_signal} execeds max length of 1024"

        if control_action == 4:  # switch turn between players 0 and 1
            self.current_player = 1 - self.current_player

            # set the flag to indicate that just switched turn
            self.just_switched_turn = True

            # set the communication signal
            self.comm_signals = communication_signal

            # render the frame if in human mode
            if self.render_mode == "human":
                self.render(mode="human")

            self.trajectory.add_state(
                GameTransition(
                    current_player=self.current_player,
                    token_pos=self.token_pos,
                    treasure_pos=self.treasure_pos,
                    action=copy.deepcopy(action),
                    reward=0,
                )
            )

            return self.get_observations(), -1, self.is_final(), False, self.get_infos()
        else:
            # Update the token position based on the action if valid
            if self.is_action_valid(self.current_player, control_action):
                self.token_pos = self.token_pos + self._action_to_direction[control_action]

            # get updated observations, rewards, terminations, and infos
            terminated = self.is_final()
            reward = self._get_rewards(control_action)
            obs_n = self.get_observations()
            info = self.get_infos()
            truncated = False

            # render the frame if in human mode
            if self.render_mode == "human":
                self.render(mode="human")

            self.trajectory.add_state(
                GameTransition(
                    current_player=self.current_player,
                    token_pos=self.token_pos,
                    treasure_pos=self.treasure_pos,
                    action=copy.deepcopy(action),
                    reward=reward,
                )
            )

            # Return the new obs, reward, done, and any additional info
            return obs_n, reward, terminated, truncated, info

    def reset(self, seed=None, options=None, token_pos=None, treasure_pos=None):
        # Reset the token position to (1, 1)
        self.current_player = 0
        if token_pos is not None:
            self.token_pos = np.array(token_pos, dtype=int)
        else:
            self.token_pos = np.array((0, 0), dtype=int)
        if treasure_pos is not None:
            self.treasure_pos = np.array(treasure_pos, dtype=int)

        # # Set the random seed if one is provided
        # if seed is not None:
        #     np.random.seed(seed)

        # # randomize the token and treasure positions
        # self.current_player = np.random.randint(0, 2)
        # self.treasure['pos'] = np.random.randint(0, 9, size=2, dtype=int)
        # while np.array_equal(self.token_pos, self.treasure['pos']):
        #     self.token_pos = np.random.randint(0, 9, size=2, dtype=int)

        # # reset the random seed to None to avoid affecting other parts of the program
        # np.random.seed(None)

        # render the initial frame if in human mode
        self.frames = []
        if self.render_mode == "human":
            self.render(mode="human")

        obs_n = self.get_observations()
        info = self.get_infos()

        if self.trajectory is not None:
            # Dump the previous trajectory with pickle
            if self.trajectory_dump_path is not None:
                if not os.path.exists(self.trajectory_dump_path):
                    os.makedirs(self.trajectory_dump_path)
                with open(f"{self.trajectory_dump_path}/traj_{self.traj_index}.pkl", "wb") as f:
                    self.trajectory.metadata["maze_id"] = self.maze_id
                    self.trajectory.metadata["round"] = self.round
                    self.trajectory.metadata["size"] = self.size
                    self.trajectory.metadata["wall_trim"] = self.wall_trim
                    self.trajectory.metadata["n_additional_walls"] = self.n_additional_walls
                    self.trajectory.metadata["treasure_pos"] = copy.deepcopy(self.treasure_pos)
                    pickle.dump(self.trajectory, f)
            self.traj_index += 1
        else:
            self.traj_index = 0

        self.trajectory = GANTrajectory()
        self.trajectory.add_state(
            GameTransition(
                current_player=self.current_player,
                token_pos=self.token_pos,
                treasure_pos=self.treasure_pos,
                action=None,
                reward=0,
            )
        )

        return obs_n, info

    def set_maze(self, maze_id="0000", round=1):
        self.maze_id = maze_id
        self.round = round
        walls, treasures, (height, width, wall_trim, n_additional_walls) = self.generate_maze(self.maze_id)
        assert height == width
        self.size = height
        self.maze_shape = (self.size, self.size)
        self.wall_trim = wall_trim
        self.n_additional_walls = n_additional_walls
        self.one_hot_walls = self.encode_walls_as_one_hot(walls)
        self.treasure_pos = treasures[self.round - 1]

    def randomize(self):
        self.set_maze(
            maze_id=np.random.choice(self.train_mazes),
            round=np.random.randint(1, 6),
        )

    """ Helper functions """

    def is_final(self):
        return np.array_equal(self.token_pos, self.treasure_pos)

    def is_action_inbound(self, action):
        # Assuming self.size is the size of the grid
        max_index = self.size - 1

        # Moving right from the rightmost column
        if action == 1 and self.token_pos[1] == max_index:
            return False

        # Moving up from the topmost row
        if action == 2 and self.token_pos[0] == 0:
            return False

        # Moving left from the leftmost column
        if action == 3 and self.token_pos[1] == 0:
            return False

        # Moving down from the bottom row
        if action == 4 and self.token_pos[0] == max_index:
            return False

        return True

    def is_action_valid(self, current_player, action):
        # Assuming self.size is the size of the grid
        max_index = self.size - 1

        x, y = self.token_pos
        # Moving right
        if action == 0:
            if y == max_index:
                # print("invalid: out of bounds; Retry!")
                return False
            # Check for a vertical wall to the right of the token
            elif self.one_hot_walls[current_player]["vertical"][x, y] == 1:
                # print("wall on the right")
                return False

        # Moving up
        if action == 1:
            if x == 0:
                # print("invalid: out of bounds; Retry!")
                return False
            # Check for a horizontal wall above the token
            if self.one_hot_walls[current_player]["horizontal"][x - 1, y] == 1:
                # print("wall above")
                return False

        # Moving left
        if action == 2:
            if y == 0:
                # print("invalid: out of bounds; Retry!")
                return False
            # Check for a vertical wall to the left of the token
            if self.one_hot_walls[current_player]["vertical"][x, y - 1] == 1:
                # print("wall on the left")
                return False

        # Moving down
        if action == 3:
            if x == max_index:
                # print("invalid: out of bounds; Retry!")
                return False
            # Check for a horizontal wall below the token
            if self.one_hot_walls[current_player]["horizontal"][x, y] == 1:
                # print("wall below")
                return False

        return True

    def get_valid_actions(self, current_player):
        return [a for a in range(5) if self.is_action_valid(current_player, a)]

    def get_inbound_actions(self):
        return [a for a in range(5) if self.is_action_inbound(a)]

    def _get_rewards(self, action):
        reward = 0

        # if reach the treasure, reward = 10 (to encourage reaching the treasure)
        if np.array_equal(self.token_pos, self.treasure_pos):
            reward += 100

        # Scaled reward based on Manhattan distance to treasure (wrong! because of the walls)
        # reward -= 0.1*self.get_infos()['distance']

        # if hit a wall, reward = -0.5 (to discourage hitting walls)
        if not self.is_action_valid(self.current_player, action):
            reward -= 5

        # each step, reward = -0.1 (to encourage shorter paths)
        reward -= 1

        return reward

    def get_observations(self):
        obs = {}
        for player in range(self.n_agents):
            obs[f"agent_{player}"] = self.get_player_observation(player)
        obs["comm"] = self.comm_signals
        return obs

    def get_player_observation(self, player):
        treasure_pos = self.treasure_pos

        # flatten the one_hot_walls matrices
        walls = self.one_hot_walls[player]
        flattened_walls = np.concatenate((walls["vertical"].flatten(), walls["horizontal"].flatten()))

        _player_i_obs = [self.current_player]  # current_player
        _player_i_obs += self.token_pos.tolist()  # token_pos
        _player_i_obs += flattened_walls.tolist()  # walls
        _player_i_obs += treasure_pos.tolist()  # treasure_pos

        # add communication signal if just switched turn
        # if self.just_switched_turn:
        #     _player_i_obs.append(self.comm_signals)
        #     self.just_switched_turn = False
        # else:
        #     _player_i_obs.append(-1)

        return np.array(_player_i_obs, dtype=np.int32)

    def get_infos(self):
        return {
            "current_player": self.current_player,
            "token": self.token_pos,
            "treasure_pos": self.treasure_pos,
            "distance": np.linalg.norm(
                self.token_pos - self.treasure_pos,
                ord=1,  # Manhattan distance from token to treasure
            ),
        }

    """ Rendering """

    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode
        if mode == "rgb_array" or mode == "human":
            if self.renderer is None:
                from mhri.utils.renderer import MatplotlibRenderer

                self.renderer = MatplotlibRenderer(self)
            frame = self.renderer.render(mode=mode)
            if mode == "rgb_array":
                return frame
        else:
            raise ValueError("Unsupported render mode: {}".format(mode))

    """ Generate maze (wall, treasures) from seed """

    def generate_maze(self, maze_id="0000"):
        # Load the maze layout from the JSON file
        maze_layout_path = os.path.join(os.path.dirname(__file__), "maze_layouts", f"{maze_id}.json")
        with open(maze_layout_path, "r") as f:
            layout = json.load(f)

        treasures = []
        if "treasures" in layout:
            # Extract the treasure positions from the layout if available
            for t in layout["treasures"]:
                treasures.append(np.array([t["x"], t["y"]], dtype=int))
        else:
            # randomly generate 5 treasure positions if not available
            for _ in range(5):
                treasures.append(
                    np.array(
                        [
                            np.random.randint(0, layout["height"]),
                            np.random.randint(0, layout["width"]),
                        ],
                        dtype=int,
                    )
                )

        # Extract the walls from the layout
        walls = [layout["p1-walls"], layout["p2-walls"]]

        return (
            walls,
            treasures,
            (
                layout["height"],
                layout["width"],
                layout["wall_trim"] if "wall_trim" in layout else None,
                layout["n_additional_walls"] if "n_additional_walls" in layout else None,
            ),
        )

    def encode_walls_as_one_hot(self, walls):
        one_hot_walls = []

        for wall_group in walls:
            # Initialize the numpy arrays for vertical and horizontal walls
            vertical_walls = np.zeros((9, 8), dtype=int)
            horizontal_walls = np.zeros((8, 9), dtype=int)

            for wall in wall_group:
                # Calculate the wall indices, adjusting for 0-indexing
                wall_index_x = wall["from"]["x"]
                wall_index_y = wall["from"]["y"]

                # Check if the wall is vertical
                if wall["from"]["y"] == wall["to"]["y"]:
                    # Wall is vertical
                    vertical_walls[wall_index_y, wall_index_x] = 1
                else:
                    # Wall is horizontal
                    horizontal_walls[wall_index_y, wall_index_x] = 1

            # Append the wall matrices for the current player to the one_hot_walls list
            one_hot_walls.append({"vertical": vertical_walls, "horizontal": horizontal_walls})

        return one_hot_walls

    def decode_player_wall_obs(self, player_obs_walls):
        # player_obs_walls is a flattened array of shape (2 * 9 * 8,)
        # The first 9 * 8 elements correspond to the vertical walls
        # The next 9 * 8 elements correspond to the horizontal walls
        # The following code reshapes the array into two matrices of shape (9, 8)
        # The first matrix corresponds to the vertical walls
        #
        # The second matrix corresponds to the horizontal walls
        vertical_walls = player_obs_walls[: 9 * 8].reshape((9, 8))
        horizontal_walls = player_obs_walls[9 * 8 :].reshape((8, 9))
        one_hot_walls = {"vertical": vertical_walls, "horizontal": horizontal_walls}
        return one_hot_walls

    """ Parse transitions from maze walls """

    def walls_to_transitions(self, one_hot_walls):
        vertical_walls = one_hot_walls["vertical"]
        horizontal_walls = one_hot_walls["horizontal"]

        rows, cols = self.maze_shape
        states = [(x, y) for x in range(rows) for y in range(cols)]

        transitions = {}  # transitions = {state: {action: next_state}}

        for state in states:
            for action, action_label in self.actions_map.items():
                if action_label == "stay_put":
                    transitions[state] = {action: state}  # stay_put allowed in all states
                else:
                    x, y = state
                    if action_label == "right" and y < cols - 1 and vertical_walls[x, y] == 0:
                        if state not in transitions:
                            transitions[state] = {action: (x, y + 1)}
                        else:
                            transitions[state][action] = (x, y + 1)
                    elif action_label == "left" and y > 0 and vertical_walls[x, y - 1] == 0:
                        if state not in transitions:
                            transitions[state] = {action: (x, y - 1)}
                        else:
                            transitions[state][action] = (x, y - 1)
                    elif action_label == "down" and x < rows - 1 and horizontal_walls[x, y] == 0:
                        if state not in transitions:
                            transitions[state] = {action: (x + 1, y)}
                        else:
                            transitions[state][action] = (x + 1, y)
                    elif action_label == "up" and x > 0 and horizontal_walls[x - 1, y] == 0:
                        if state not in transitions:
                            transitions[state] = {action: (x - 1, y)}
                        else:
                            transitions[state][action] = (x - 1, y)

        return transitions

    """ Closing the environment """

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


class GameTransition:
    def __init__(self, current_player, token_pos, treasure_pos, action, reward):
        self.current_player = current_player
        self.token_pos = token_pos
        self.treasure_pos = treasure_pos
        self.action = action
        self.reward = reward


class GANTrajectory:
    def __init__(self):
        self.metadata = {}
        self.states = []

    def add_state(self, state):
        self.states.append(state)
