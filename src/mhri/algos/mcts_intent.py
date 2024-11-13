import math
import random
from typing import Callable, Dict, List, Tuple
import numpy as np

MOVE_RIGHT = 0
MOVE_UP = 1
MOVE_LEFT = 2
MOVE_DOWN = 3
SWITCH_TURN = 4


def next_state_without_walls(current_state, action, maze_size=(9, 9)):
    action_to_direction = {
        0: np.array([0, 1]),  # right
        1: np.array([-1, 0]),  # up
        2: np.array([0, -1]),  # left
        3: np.array([1, 0]),  # down
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


class GameState:
    def __init__(self, token_pos: Tuple[int], treasure_pos: Tuple[int], who_has_control: int):
        self.token_pos = token_pos
        self.treasure_pos = treasure_pos
        self.who_has_control = who_has_control

    def is_terminal(self):
        return self.token_pos == self.treasure_pos


class GameDynamics:
    def __init__(
        self,
        player_id: int,
        player_transitions: Dict[Tuple[int], Dict[int, Tuple[int]]],
        teammate_transitions_belief: Callable[[Tuple[int], int], Tuple[Tuple[int], float]],
        teammate_intent: List[Tuple[int]],
        intent_mode: str = None,
    ):
        self.player_id = player_id
        self.player_transitions = player_transitions
        self.teammate_transitions_belief = teammate_transitions_belief
        self.teammate_intent = teammate_intent
        self.intent_mode = intent_mode

    def get_reward(self, who_has_control, next_pos, treasure_pos):
        step_reward = 100 if next_pos == treasure_pos else -1
        if who_has_control == self.player_id:
            if next_pos in self.teammate_intent and self.intent_mode is not None and self.intent_mode != "none":
                if self.intent_mode == "fixed":
                    step_reward += 0.5
                elif self.intent_mode == "length_inverse":
                    if self.teammate_intent.index(next_pos) == len(self.teammate_intent) - 1:  # Reaches the end
                        step_reward += 1
                    else:
                        step_reward += 1 / len(self.teammate_intent)
                elif self.intent_mode == "discounted":  # discounted based on the steps until the end
                    found_intent_index = self.teammate_intent.index(next_pos)
                    step_reward += 0.9 ** (len(self.teammate_intent) - found_intent_index - 1)
                elif self.intent_mode == "single_step_reward":
                    if self.teammate_intent.index(next_pos) == 1:  # Only reward for the first step
                        step_reward += 0.5
        return step_reward

    def step(self, state: GameState, action: int) -> tuple[GameState, float, float]:
        if action == SWITCH_TURN:
            next_state = GameState(
                token_pos=state.token_pos,
                treasure_pos=state.treasure_pos,
                who_has_control=1 - state.who_has_control,
            )
            step_reward = -1
            transition_prob = 1
            return next_state, step_reward, transition_prob

        if state.who_has_control == self.player_id:  # We know the true transition function
            transition_prob = 1
            assert action in self.player_transitions[state.token_pos]
            next_pos = self.player_transitions[state.token_pos][action]
            next_state = GameState(
                token_pos=next_pos,
                treasure_pos=state.treasure_pos,
                who_has_control=state.who_has_control,
            )
        else:  # Estimate with transition belief
            next_pos, transition_prob = self.teammate_transitions_belief(state.token_pos, action)
            next_state = GameState(
                token_pos=next_pos,
                treasure_pos=state.treasure_pos,
                who_has_control=state.who_has_control,
            )

        step_reward = self.get_reward(state.who_has_control, next_pos, state.treasure_pos)
        return next_state, step_reward, transition_prob

    def get_possible_actions(self, state: GameState):
        if state.who_has_control == self.player_id:
            move_actions = list(self.player_transitions[state.token_pos].keys())
            return move_actions + [SWITCH_TURN]
        else:
            # Assume the teammate can move freely
            move_actions = []
            for a in [MOVE_RIGHT, MOVE_UP, MOVE_LEFT, MOVE_DOWN]:
                if next_state_without_walls(state.token_pos, a) != state.token_pos:  # ignore out of bounds
                    move_actions.append(a)
            return move_actions + [SWITCH_TURN]

    def rollout_from_state(self, state: GameState, gamma: float, horizon: int = 100):
        current_state = state

        q = 0

        for d in range(horizon):
            action = random.choice(self.get_possible_actions(current_state))
            next_state, _, transition_prob = self.step(current_state, action)
            if np.random.random() < transition_prob:  # transition succeeds
                current_state = next_state
            reward = self.get_reward(current_state.who_has_control, current_state.token_pos, current_state.treasure_pos)

            q += gamma**d * reward
            if current_state.is_terminal():
                break

        return q


# NOTES
# - never step the simulator
# - stopping criterion
# - when to accumulate reward (expand and/or simulate)
# Apply_action should have different behavior when called from simulate and expand.
# When expanding, we only consider the transition that makes progress, and use trans_prob to discount the reward
# When simulating, we sample if the transition succeeds or not with trans_prob.
# (draw random uniform 0, 1 if trans_prob < random_uniform, don't move)


class TreeNode:
    def __init__(self, state, parent, parent_action, step_reward, transition_prob):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.parent = parent
        self.parent_action = parent_action  # action that leads to this node from parent
        self.step_reward = step_reward  # reward for reaching this node from parent
        self.transition_prob = transition_prob  # P(current node | parent node, action)
        self.n_visits = 0
        self.total_reward = 0
        self.children: Dict[int, TreeNode] = {}

    def __str__(self):
        s = [
            "totalReward: %s" % self.totalReward,
            "numVisits: %d" % self.numVisits,
            "isTerminal: %s" % self.is_terminal,
            "possibleActions: %s" % (self.children.keys()),
        ]
        return "%s: {%s}" % (self.__class__.__name__, ", ".join(s))


class MCTS:
    def __init__(
        self,
        initial_state: GameState,
        game_dynamics: GameDynamics,
        n_iterations: int = 100,
        exploration_constant: float = np.sqrt(2),
        gamma: float = 0.99,  # discount factor
    ):
        self.root = TreeNode(initial_state, None, None, 0, 1)
        self.game_dynamics = game_dynamics

        self.n_iterations = n_iterations
        self.exploration_constant = exploration_constant
        self.gamma = gamma

    def search(
        self,
        verbose: bool = None,
    ):
        for itr in range(self.n_iterations):
            # print("iteration", itr)
            self.execute_round()

        # Return the most visited child
        if self.game_dynamics.intent_mode == "single_step" and len(self.game_dynamics.teammate_intent) > 1:
            children_n_visits = [child.n_visits for child in self.root.children.values()]
            max_visits = max(children_n_visits)

            best_child = None
            for child in self.root.children.values():
                if child.n_visits == max_visits and child.state.token_pos == self.game_dynamics.teammate_intent[1]:
                    best_child = child
                    break
            if best_child is None:
                best_child = max(self.root.children.values(), key=lambda x: x.n_visits)
        else:
            best_child = max(self.root.children.values(), key=lambda x: x.n_visits)
        action = best_child.parent_action
        if verbose:
            return action, best_child.total_reward / best_child.n_visits
        else:
            return action

    def execute_round(self):
        """
        execute a selection-expansion-simulation-backpropagation round
        """
        node = self.select_node(self.root)
        q = self.game_dynamics.rollout_from_state(node.state, self.gamma)
        self.backpropogate(node, q)

    def select_node(self, node: TreeNode) -> TreeNode:
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = self.get_best_child(node)
            else:
                return self.expand(node)
        return node

    def expand(self, node: TreeNode) -> TreeNode:
        # print("expanding node", node.state.token_pos)
        actions = self.game_dynamics.get_possible_actions(node.state)
        # print("possible actions", actions)
        for action in actions:
            if action not in node.children:
                next_state, reward, transition_prob = self.game_dynamics.step(node.state, action)
                new_child = TreeNode(next_state, node, action, reward, transition_prob)
                node.children[action] = new_child
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_child

        raise Exception("Trying to expand node that is already fully expanded.")

    def backpropogate(self, node: TreeNode, q: float):
        q_sample = q
        transition_prob = 1
        while node is not None:
            current_value = node.total_reward / node.n_visits if node.n_visits > 0 else 0
            expected_q_sample = (
                q_sample * transition_prob  # transition succeeds
                + current_value * (1 - transition_prob)  # transition fails
            )
            q_sample = node.step_reward + self.gamma * expected_q_sample
            node.total_reward += q_sample
            node.n_visits += 1
            transition_prob = node.transition_prob
            node = node.parent

    def get_best_child(self, node: TreeNode) -> TreeNode:
        best_value = -np.inf
        best_children = []
        for child in node.children.values():
            child_value = child.total_reward / child.n_visits + self.exploration_constant * math.sqrt(
                math.log(node.n_visits) / child.n_visits
            )
            if child_value > best_value:
                best_value = child_value
                best_children = [child]
            elif child_value == best_value:
                best_children.append(child)
        return random.choice(best_children)
