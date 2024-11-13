import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MatplotlibRenderer:
    def __init__(self, env):
        self.env = env

        # Define the rendering variables
        self.CELL_SIZE = 40
        self.WALL_WIDTH = 3
        self.OUT_WALL_WIDTH = 5
        self.OFFSET = 20  # Space between the two mazes
        self.MARGIN = 20  # Margin around the mazes
        self.SCREEN_WIDTH = (self.env.size * self.CELL_SIZE) * 2 + self.OFFSET + (self.MARGIN * 2)
        self.SCREEN_HEIGHT = self.env.size * self.CELL_SIZE + (self.MARGIN * 2)
        self.WALL_COLOR = (0, 0, 0)  # black
        self.BG_COLOR = (255, 255, 255)  # white

        # Matplotlib rendering variables
        self.figure = plt.figure(figsize=(8, 4), dpi=100)
        self.ax0 = self.figure.add_subplot(121)
        self.ax1 = self.figure.add_subplot(122)

    def render_client(self, mode=None, client_id=0):
        if mode is None:
            mode = self.render_mode
        if mode == "rgb_array" or mode == "human":
            frame = self._render_frame_client(mode=mode, client_id=client_id)
            if mode == "rgb_array":
                H = 800
                W = 1600
                TOP_CROP = 44
                BOTTOM_CROP = 36
                if client_id == 0:
                    LEFT_CROP = 122
                    RIGHT_CROP = 758
                else:
                    LEFT_CROP = 798
                    RIGHT_CROP = 82
                assert frame.shape[:2] == (H, W), frame.shape
                frame = frame[TOP_CROP:-BOTTOM_CROP, LEFT_CROP:-RIGHT_CROP]
                return frame
        else:
            raise ValueError("Unsupported render mode: {}".format(mode))

    def _render_frame_client(self, mode=None, client_id=0):
        tick_positions = range(1, self.env.size + 1)  # Positions from 1 to size

        # Clear the previous visualization
        for idx, ax in enumerate([self.ax0, self.ax1]):
            if idx == client_id:
                ax.clear()
                ax.set_visible(True)
            else:
                ax.set_visible(False)

        # Set the axis to plot on
        ax = [self.ax0, self.ax1][client_id]

        # Set visualization properties
        ax.set_facecolor("white")
        ax.grid(color="lightgray", linewidth=1)
        ax.set_xlim(0, self.env.maze_shape[0])
        ax.set_ylim(0, self.env.maze_shape[1])
        ax.set_xticks(range(self.env.maze_shape[0] + 1))
        ax.set_yticks(range(self.env.maze_shape[1] + 1))
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        # ax.set_title("Player {}".format(client_id))

        # Set circumference line widths
        if self.env.current_player == client_id:
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_edgecolor("blue")
        else:
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_edgecolor("black")

        # Add token on both plots
        if self.env.current_player == client_id:
            self._addTokenSolid(ax, self.env.token_pos)
        else:
            self._addTokenDashed(ax, self.env.token_pos)

        # Add treaure on the side with the treasure
        # if self.treasure["onWhichSide"] == client_id:
        self._addTreasure(ax, self.env.treasure_pos)  # Include treasure on both sides

        # add walls on corresponding maze
        self._addWalls(ax, self.env.one_hot_walls[client_id])

        # add grid axis labels
        ax.text(
            -0.4,
            self.env.size + 0.1,
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
                str(self.env.size - pos),
                ha="center",
                va="center",
                color="darkgray",
            )

        if mode == "human":
            plt.draw()
            plt.pause(0.001)

        elif mode == "rgb_array":
            self.figure.canvas.draw()
            return np.array(self.figure.canvas.renderer.buffer_rgba())

    def render(self, mode=None):
        tick_positions = range(1, self.env.size + 1)  # Positions from 1 to size

        for idx, ax in enumerate([self.ax0, self.ax1]):
            # Clear the previous visualization
            ax.clear()

            # Set visualization properties
            ax.set_facecolor("white")
            ax.grid(color="lightgray", linewidth=1)
            ax.set_xlim(0, self.env.maze_shape[0])
            ax.set_ylim(0, self.env.maze_shape[1])
            ax.set_xticks(range(self.env.maze_shape[0] + 1))
            ax.set_yticks(range(self.env.maze_shape[1] + 1))
            ax.tick_params(axis="both", which="both", length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal")
            ax.set_title("Player {}".format(idx))

            # Set circumference line widths
            if self.env.current_player == idx:
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    spine.set_edgecolor("blue")
            else:
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    spine.set_edgecolor("black")

            # Add token on both plots
            if self.env.current_player == idx:
                self._addTokenSolid(ax, self.env.token_pos)
            else:
                self._addTokenDashed(ax, self.env.token_pos)

            # Always show the treasure
            self._addTreasure(ax, self.env.treasure_pos)

            # add walls on corresponding maze
            self._addWalls(ax, self.env.one_hot_walls[idx])

            # add grid axis labels
            ax.text(
                -0.4,
                self.env.size + 0.1,
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
                    str(self.env.size - pos),
                    ha="center",
                    va="center",
                    color="darkgray",
                )

        if mode == "human":
            plt.draw()
            plt.pause(0.001)

        elif mode == "rgb_array":
            self.figure.canvas.draw()
            return np.array(self.figure.canvas.renderer.buffer_rgba())

    def _posToText(self, pos):
        x, y = pos
        return y + 0.5, self.env.maze_shape[1] - x - 0.5

    def _posToCircle(self, pos):
        x, y = pos
        return y + 0.5, self.env.maze_shape[1] - x - 0.5

    def _addTokenSolid(self, ax, token_pos):
        ax.add_patch(patches.Circle(self._posToCircle(token_pos), 0.3, fill=False, color="blue", linewidth=2))
        ax.text(
            self._posToText(token_pos)[0],
            self._posToText(token_pos)[1],
            "S",
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
        )

    def _addTokenDashed(self, ax, token_pos):
        ax.add_patch(
            patches.Circle(
                self._posToCircle(token_pos),
                0.3,
                fill=False,
                color="blue",
                linewidth=2,
                linestyle="--",
            )
        )
        ax.text(
            self._posToText(token_pos)[0],
            self._posToText(token_pos)[1],
            "S",
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
        )

    def _addTreasure(self, ax, treasure_pos):
        ax.add_patch(
            patches.Circle(
                self._posToCircle(treasure_pos),
                0.3,
                fill=False,
                color="orange",
                linewidth=2,
            )
        )
        ax.text(
            self._posToText(treasure_pos)[0],
            self._posToText(treasure_pos)[1],
            "T",
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
        )

    def _addWalls(self, ax, one_hot_wall):
        vertical_walls = one_hot_wall["vertical"]
        horizontal_walls = one_hot_wall["horizontal"]

        # Plot vertical walls
        for x in range(self.env.size):
            for y in range(self.env.size - 1):  # Adjust for the column count in vertical walls
                if vertical_walls[x, y] == 1:
                    ax.plot(
                        [y + 1, y + 1],
                        [self.env.size - x - 1, self.env.size - x],
                        color="black",
                        linewidth=2,
                    )

        # Plot horizontal walls
        for x in range(self.env.size - 1):  # Adjust for the row count in horizontal walls
            for y in range(self.env.size):
                if horizontal_walls[x, y] == 1:
                    ax.plot(
                        [y, y + 1],
                        [self.env.size - x - 1, self.env.size - x - 1],
                        color="black",
                        linewidth=2,
                    )

    def close(self):
        plt.close()


class TextRenderer:
    def __init__(self, env):
        self.env = env

    def _render_as_text(self, player_idx=0):
        maze_rows = []

        horizontal_border = "+" + "-+" * self.env.size
        maze_rows.append(horizontal_border)

        def get_cell_char(y, x):
            if np.array_equal(self.env.token_pos, np.array([x, y])):
                return "S"
            elif np.array_equal(self.env.treasure["pos"], np.array([x, y])):
                return "T"
            else:
                return "o"

        for y in range(self.env.size):
            # Vertical walls
            row_str = "|"
            for x in range(self.env.size - 1):
                row_str += f"{get_cell_char(x, y)}"
                if self.env.one_hot_walls[player_idx]["vertical"][y, x] == 1:
                    row_str += "|"
                else:
                    row_str += "."
            row_str += f"{get_cell_char(self.env.size - 1, y)}|"
            maze_rows.append(row_str)

            # Horizontal walls
            if y != self.env.size - 1:
                row_str = "+"
                for x in range(self.env.size):
                    if self.env.one_hot_walls[player_idx]["horizontal"][y, x] == 1:
                        row_str += "-+"
                    else:
                        row_str += ".+"
                maze_rows.append(row_str)
            else:
                maze_rows.append(horizontal_border)

        return "\n".join(maze_rows)
