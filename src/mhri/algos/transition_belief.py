import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

MOVE_RIGHT = 0
MOVE_UP = 1
MOVE_LEFT = 2
MOVE_DOWN = 3
SWITCH_TURN = 4


def next_state_without_walls(current_state, action, maze_size=(9, 9)):
    action_to_direction = {
        MOVE_RIGHT: np.array([0, 1]),  # right
        MOVE_UP: np.array([-1, 0]),  # up
        MOVE_LEFT: np.array([0, -1]),  # left
        MOVE_DOWN: np.array([1, 0]),  # down
    }
    x, y = tuple(np.array(current_state) + action_to_direction[action])

    # Check if the next state is out of bounds
    row, col = maze_size
    if x < 0 or x > row - 1 or y < 0 or y > col - 1:
        # raise ValueError(f"Next state ({x}, {y}) is out of bounds.")
        x = max(0, min(x, row - 1))
        y = max(0, min(y, col - 1))
        return (x, y)
    else:
        return (x, y)


def compute_ber_weight_success(ber_param, weight_failure):
    # ber_param:: parameter of this Bernoulli distribution, i.e., \theta in Ber(\theta)
    # weight_failure:: c^-
    # output: weight_success c^+, which is computed by \frac{\log(1-(1-\theta)^{c^-})}{\log \theta}
    return np.log(1 - (1 - ber_param) ** weight_failure) / np.log(ber_param)


class DummyTransitionBelief:
    def __init__(self, state_space, action_space, maze_shape):
        self.state_space = state_space
        self.action_space = action_space
        self.maze_shape = maze_shape

    def get_belief(self, state, action):
        next_state, is_out_of_bounds = next_state_without_walls(state, action, self.maze_shape)
        if is_out_of_bounds:
            return next_state, 1.0
        else:
            return next_state, 0.5

    def __call__(self, state, action):
        return self.get_belief(state, action)


class TransitionBelief:
    def __init__(self, state_space, action_space, maze_shape):
        self.state_space = state_space
        self.action_space = action_space
        self.maze_shape = maze_shape

        row, col = maze_shape
        assert row == col, "Only square maze is supported."

        # Modeling belief for walls by initializing as Beta(1, 1)
        # uniform numpy array [.,.,0] for vertical and [.,.,1] for horizontal
        self.belief_alpha = np.ones((row, col - 1, 2))
        self.belief_beta = np.ones((row, col - 1, 2))

        # element-wise: belief = alpha / (alpha + beta)
        self.belief = self.belief_alpha / (self.belief_alpha + self.belief_beta)

    def find_belief_coord(self, state, action):
        x, y = state
        # vertical walls
        if action == MOVE_RIGHT:
            return x, y, 0
        elif action == MOVE_LEFT:
            return x, y - 1, 0
        # horizontal walls (transposed)
        elif action == MOVE_UP:
            return y, x - 1, 1
        elif action == MOVE_DOWN:
            return y, x, 1
        else:
            raise ValueError(f"Invalid action: {action}")

    def get_belief(self, state, action):
        next_state = next_state_without_walls(state, action, self.maze_shape)
        if next_state == state:
            return next_state, 1.0
        else:
            coord = self.find_belief_coord(state, action)
            return next_state, self.belief[coord]

    def __call__(self, state, action):
        return self.get_belief(state, action)

    def update_with_trajectory(self, trajectory, weight_success=3):
        for state, action in trajectory:
            assert action in self.action_space, f"Invalid action: {action}"
            for other_action in self.action_space:
                next_state = next_state_without_walls(state, other_action, self.maze_shape)
                if next_state != state:
                    coord = self.find_belief_coord(state, other_action)
                    if other_action == action:
                        self.belief_alpha[coord] += weight_success
                    else:
                        self.belief_beta[coord] += compute_ber_weight_success(self.belief[coord], weight_success)
                    self.belief[coord] = self.belief_alpha[coord] / (self.belief_alpha[coord] + self.belief_beta[coord])

    def update_with_communication(self, message):
        pass

    def print_belief_matrix(self):
        print("vertical walls:")
        print(self.belief[:, :, 0])
        print("horizontal walls:")
        print(self.belief[:, :, 1].T)

    def render_belief_heatmap(self, show_colorbar=False, save_path=None):
        import matplotlib.pyplot as plt

        tick_positions = range(1, self.maze_shape[0] + 1)  # Positions from 1 to size

        if show_colorbar:
            fig, ax = plt.subplots(1, 1, figsize=(4.42, 3.6), tight_layout=True)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.6), tight_layout=True)

        ax.set_facecolor("white")
        # ax.grid(color="lightgray", linewidth=1)
        ax.set_xlim(0, self.maze_shape[0])
        ax.set_ylim(0, self.maze_shape[1])
        ax.set_xticks(range(self.maze_shape[0] + 1))
        ax.set_yticks(range(self.maze_shape[1] + 1))
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        # ax.set_title("Transition Belief")

        colors = ["white", "black"]
        cmap = LinearSegmentedColormap.from_list("linear_color_gradient", colors)

        # add grid axis labels
        ax.text(
            -0.4,
            self.maze_shape[0] + 0.1,
            "X",
            ha="center",
            va="center",
            color="darkgray",
            weight="bold",
        )
        ax.text(
            -0.1,
            -0.4,
            "Y",
            ha="center",
            va="center",
            color="darkgray",
            weight="bold",
        )
        for pos in tick_positions:
            ax.text(
                pos - 0.5,
                -0.4,
                str(pos - 1),
                ha="center",
                va="center",
                color="darkgray",
            )
            ax.text(
                -0.4,
                pos - 0.5,
                str(self.maze_shape[0] - pos),
                ha="center",
                va="center",
                color="darkgray",
            )

        # Plot vertical walls
        for x in range(self.maze_shape[0]):
            for y in range(self.maze_shape[1] - 1):  # Adjust for the column count in vertical walls
                ax.plot(
                    [y + 1, y + 1],
                    [self.maze_shape[0] - x - 1, self.maze_shape[0] - x],
                    color=cmap(1 - self.belief[x, y, 0]),
                    linewidth=2,
                    alpha=1 - self.belief[x, y, 0],
                )

        # Plot horizontal walls
        for x in range(self.maze_shape[0] - 1):  # Adjust for the row count in horizontal walls
            for y in range(self.maze_shape[1]):
                ax.plot(
                    [y, y + 1],
                    [self.maze_shape[1] - x - 1, self.maze_shape[1] - x - 1],
                    color=cmap(1 - self.belief[y, x, 1]),
                    linewidth=2,
                    alpha=1 - self.belief[y, x, 1],
                )

        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        cmappable = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
        if show_colorbar:
            plt.colorbar(cmappable, ax=ax)
            # Write text to the right of the colorbar
            ax.text(
                11.5,
                4.5,
                "$\mathbf{P}$(wall exists)",
                ha="right",
                va="center",
                color="black",
                rotation=-90,
            )

        if save_path is not None:
            fig.savefig(save_path, dpi=300)

        # Return a PIL image
        fig.canvas.draw()
        rgba = np.array(fig.canvas.renderer.buffer_rgba())
        # close the figure
        plt.close(fig)
        return Image.fromarray(rgba)
