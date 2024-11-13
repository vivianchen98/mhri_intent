import numpy as np
import heapq
from collections import deque, defaultdict

""" Build Graph """


def build_graph(states, actions):
    graph = defaultdict(list)  # Use a set to store neighbors to avoid duplicates
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        action = actions[i]
        if (next_state, action) not in graph[current_state]:  # Avoid duplicates
            graph[current_state].append((next_state, action))
    return graph


def build_graph_only_actions(actions):
    # Define movements
    movements = {
        0: (0, 0),  # Stay put
        1: (0, 1),  # Right
        2: (-1, 0),  # Up
        3: (0, -1),  # Left
        4: (1, 0),  # Down
    }

    # Create the graph
    graph = defaultdict(list)
    current_position = (0, 0)
    positions_visited = [current_position]  # Track positions for path

    for action in actions:
        move = movements[action]
        new_position = (current_position[0] + move[0], current_position[1] + move[1])
        graph[current_position].append((new_position, action))
        current_position = new_position
        positions_visited.append(new_position)

    # Find shortest path from the first to the last position
    start_pos = positions_visited[0]
    end_pos = positions_visited[-1]

    return graph, start_pos, end_pos


""" Capping shortest path from action sequence """


def find_sublist(action_sequence, shortest_path_actions):
    # Convert lists to strings
    action_str = "".join(map(str, action_sequence))
    pattern_str = "".join(map(str, shortest_path_actions))
    pattern_length = len(pattern_str)
    segments = []
    start_idx = 0

    match_idx = action_str.find(pattern_str, start_idx)
    if match_idx != -1:
        # Match found
        match_found = True
        # Add the segment before the match if it exists
        if match_idx > 0:
            segments.append(list(map(int, action_str[start_idx:match_idx])))

        # Add the segment after the matched pattern if it exists
        end_idx = match_idx + pattern_length
        if end_idx < len(action_str):
            segments.append(list(map(int, action_str[end_idx:])))

        return match_found, segments

    # If no match is found, return None
    return False, None


def capping(action_sequence, shortest_path):
    # first check if shortest_path is emptry list, if so, return action_sequence
    if not shortest_path:
        return action_sequence

    # Cap the path_segment with the shortest_path, and return the capped action sequence
    segments = []
    pattern = []
    sp_index = 0  # Index to track elements in shortest_path

    # first check if shortest_path is a substring of action_sequence
    # if so, return the leftover actions in action_sequence by removing shortest_path from action_sequence
    match_found, sublist_segments = find_sublist(action_sequence, shortest_path)
    if match_found:
        return sublist_segments

    # Otherwise, cap the action_sequence with the shortest_path
    for a in action_sequence:
        # Ensure that we do not go out of range for shortest_path
        if sp_index < len(shortest_path) and a == shortest_path[sp_index]:
            sp_index += 1
            if pattern:
                segments.append(pattern)
                pattern = []
        else:
            pattern.append(a)

    # Add any remaining pattern to segments
    if pattern:
        segments.append(pattern)

    return segments


""" Shortest Path: BFS Algorithm """


def bfs_shortest_path(graph, start, goal):
    # first check if start and goal are the same
    if start == goal:
        return []

    queue = deque([(start, [])])
    visited = set()

    while queue:
        current, path = queue.popleft()

        if current == goal:
            return path

        if current not in visited:
            visited.add(current)
            for neighbor, action in graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [action]))
    return None


""" Shortest Path: A* Algorithm """


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(current_state, final_state, transitions):
    """
    Computes the shortest path in a grid from start_state to final_state using the A* algorithm.

    Parameters:
        current_state (tuple): Starting coordinates in the grid (x, y).
        final_state (tuple): Target coordinates in the grid (x, y).
        transitions (dict): Describes allowed transitions for each state.   Format: {current_state: {action: next_state}}.

    Returns:
        path (list): List of states from start to final state.
    """

    # The priority queue
    frontier = []
    heapq.heappush(frontier, (0, current_state, []))  # (cost + heuristic, state, path)

    # Costs to reach each state
    cost_so_far = {current_state: 0}
    # Keep track of the path
    came_from = {current_state: None}

    while frontier:
        _, current, path = heapq.heappop(frontier)

        # Goal test
        if current == final_state:
            return [current_state] + path

        # Explore neighbors
        for action, next_state in transitions[current].items():
            # Cost from start to the next state
            new_cost = cost_so_far[current] + 1  # Assuming uniform cost for simplicity
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                # Record the cost to reach the state and the state came from
                cost_so_far[next_state] = new_cost
                came_from[next_state] = current
                priority = new_cost + manhattan_distance(next_state, final_state)
                heapq.heappush(frontier, (priority, next_state, path + [next_state]))

    return []  # If no path is found


""" Shortest Path with Minimum Violations """


def next_state_without_walls(current_state, action, maze_size=(9, 9)):
    action_to_direction = {
        0: np.array([0, 0]),  # stay_put
        1: np.array([0, 1]),  # right
        2: np.array([-1, 0]),  # up
        3: np.array([0, -1]),  # left
        4: np.array([1, 0]),  # down
    }
    x, y = tuple((np.array(current_state) + action_to_direction[action]).tolist())

    # Check if the next state is out of bounds
    row, col = maze_size
    if x < 0 or x > row - 1 or y < 0 or y > col - 1:
        raise ValueError(f"Next state ({x}, {y}) is out of bounds.")
    else:
        return (x, y)


def shortest_path_min_violation(
    actions, start_state, final_state, transitions, hard_constraints, violation_cost=10, maze_size=(9, 9)
):
    """
    Computes the shortest path in a grid from start_state to final_state,
    minimizing the number of violations (e.g., wall crossings).

    Parameters:
        actions (dict): Maps action IDs to their respective directions.     Format: {action: 'action_label'}.
        start_state (tuple): Starting coordinates in the grid (x, y).
        final_state (tuple): Target coordinates in the grid (x, y).
        transitions (dict): Describes allowed transitions for each state.   Format: {current_state: {action: next_state}}.
        violation_cost (int): Cost added per violation. Default is 10.

    Returns:
        path (list): List of states from start to final state.
        actions_on_path (list): List of actions taken on the path.
        violations (list): List of state tuples in path that violate the transitions.
    """

    # The priority queue
    frontier = []
    heapq.heappush(
        frontier, (0, start_state, [], 0, [], [])
    )  # (priority, current_state, path, cost, actions_on_path, violations)

    # Stores the minimum cost to reach each state
    cost_so_far = {start_state: 0}
    # Stores the minimum violations to reach each state
    violations_so_far = {start_state: 0}

    while frontier:
        _, current, path, current_cost, actions_on_path, violations = heapq.heappop(frontier)

        if current == final_state:
            return [start_state] + path, actions_on_path, violations

        for action in actions.keys():
            # Skip action if next_state is out of bounds
            try:
                next_state = next_state_without_walls(current, action, maze_size=maze_size)
            except ValueError:
                continue

            new_cost = current_cost + 1  # Assuming each move has a cost of 1
            new_violations = violations_so_far[current] + (1 if next_state not in transitions[current].values() else 0)
            total_cost = new_cost + new_violations * violation_cost

            if (next_state not in cost_so_far or new_cost < cost_so_far[next_state]) or (
                next_state in violations_so_far and new_violations < violations_so_far[next_state]
            ):
                cost_so_far[next_state] = new_cost
                violations_so_far[next_state] = new_violations
                priority = total_cost + manhattan_distance(next_state, final_state)
                updated_violations = (
                    violations + [(current, next_state)]
                    if next_state not in transitions[current].values()
                    else violations
                )
                heapq.heappush(
                    frontier,
                    (
                        priority,
                        next_state,
                        path + [next_state],
                        total_cost,
                        actions_on_path + [action],
                        updated_violations,
                    ),
                )

    return [], [], []  # No path found


def shortest_path_min_violation_hard_constraints(
    actions, start_state, final_state, transitions, hard_constraints, violation_cost=10, maze_size=(9, 9)
):
    """
    Computes the shortest path in a grid from start_state to final_state,
    minimizing the number of violations (e.g., wall crossings).

    Parameters:
        actions (dict): Maps action IDs to their respective directions.     Format: {action: 'action_label'}.
        start_state (tuple): Starting coordinates in the grid (x, y).
        final_state (tuple): Target coordinates in the grid (x, y).
        transitions (dict): Describes allowed transitions for each state.   Format: {current_state: {action: next_state}}.
        hard_constraints (list): List of tuples representing hard constraints. Format: [(state1, state2)].
        violation_cost (int): Cost added per violation. Default is 10.

    Returns:
        path (list): List of states from start to final state.
        actions_on_path (list): List of actions taken on the path.
        violations (list): List of state tuples in path that violate the transitions.
    """

    # The priority queue
    frontier = []
    heapq.heappush(
        frontier, (0, start_state, [], 0, [], [])
    )  # (priority, current_state, path, cost, actions_on_path, violations)

    # Stores the minimum cost to reach each state
    cost_so_far = {start_state: 0}
    # Stores the minimum violations to reach each state
    violations_so_far = {start_state: 0}

    while frontier:
        _, current, path, current_cost, actions_on_path, violations = heapq.heappop(frontier)

        if current == final_state:
            return [start_state] + path, actions_on_path, violations

        for action in actions.keys():
            # Skip action if next_state is out of bounds
            try:
                next_state = next_state_without_walls(current, action, maze_size=maze_size)
            except ValueError:
                # print(f"Next state is out of bounds: {next_state}")
                continue

            # Check if the next state violates any hard constraints
            if (current, next_state) in hard_constraints or (
                next_state,
                current,
            ) in hard_constraints:
                # print(f"Hard constraint violated: {current} -> {next_state}")
                continue

            new_cost = current_cost + 1  # Assuming each move has a cost of 1
            new_violations = violations_so_far[current] + (1 if next_state not in transitions[current].values() else 0)
            total_cost = new_cost + new_violations * violation_cost

            if (next_state not in cost_so_far or new_cost < cost_so_far[next_state]) or (
                next_state in violations_so_far and new_violations < violations_so_far[next_state]
            ):
                cost_so_far[next_state] = new_cost
                violations_so_far[next_state] = new_violations
                priority = total_cost + manhattan_distance(next_state, final_state)
                updated_violations = (
                    violations + [(current, next_state)]
                    if next_state not in transitions[current].values()
                    else violations
                )
                heapq.heappush(
                    frontier,
                    (
                        priority,
                        next_state,
                        path + [next_state],
                        total_cost,
                        actions_on_path + [action],
                        updated_violations,
                    ),
                )

    return [], [], []  # No path found


def shortest_path_min_violation_with_belief(
    actions, start_state, final_state, transitions, teammate_transition_beliefs, violation_cost=10, maze_size=(9, 9)
):
    """
    Computes the shortest path in a grid from start_state to final_state,
    minimizing the number of violations (e.g., wall crossings).
    """
    # The priority queue
    frontier = []
    heapq.heappush(
        frontier, (0, start_state, [], 0, [], [])
    )  # (priority, current_state, path, cost, actions_on_path, violations)

    # Stores the minimum cost to reach each state
    cost_so_far = {start_state: 0}
    # Stores the minimum violations to reach each state
    violations_so_far = {start_state: 0}

    while frontier:
        _, current, path, current_cost, actions_on_path, violations = heapq.heappop(frontier)

        if current == final_state:
            return [start_state] + path, actions_on_path, violations

        all_actions = list(actions.keys())
        np.random.shuffle(all_actions)
        for action in all_actions:
            # Skip action if next_state is out of bounds
            try:
                next_state = next_state_without_walls(current, action, maze_size=maze_size)
            except ValueError:
                # print(f"Next state is out of bounds: {next_state}")
                continue

            new_cost = current_cost + 1  # Assuming each move has a cost of 1

            wall_crossed = next_state not in transitions[current].values()
            if wall_crossed:
                # Find the teammate's belief for successful transition
                _, teammate_belief = teammate_transition_beliefs(current, action - 1)
                new_violations = violations_so_far[current] + (1 - teammate_belief)
            else:
                new_violations = violations_so_far[current]

            total_cost = new_cost + new_violations * violation_cost

            if (next_state not in cost_so_far or new_cost < cost_so_far[next_state]) or (
                next_state in violations_so_far and new_violations < violations_so_far[next_state]
            ):
                cost_so_far[next_state] = new_cost
                violations_so_far[next_state] = new_violations
                priority = total_cost + manhattan_distance(next_state, final_state)
                updated_violations = (
                    violations + [(current, next_state)]
                    if next_state not in transitions[current].values()
                    else violations
                )
                heapq.heappush(
                    frontier,
                    (
                        priority,
                        next_state,
                        path + [next_state],
                        total_cost,
                        actions_on_path + [action],
                        updated_violations,
                    ),
                )

    return [], [], []  # No path found


def shortest_path_with_turns(
    actions, start_state, final_state, p0_transitions, p1_transitions, switch_cost=1, maze_size=(9, 9)
):
    """
    Computes the shortest path in a grid from start_state to final_state.

    Parameters:
        actions (dict): Maps action IDs to their respective directions.     Format: {action: 'action_label'}.
        start_state (tuple): Starting coordinates in the grid (x, y).
        final_state (tuple): Target coordinates in the grid (x, y).
        transitions (dict): Describes allowed transitions for each state.   Format: {current_state: {action: next_state}}.
        violation_cost (int): Cost added per violation. Default is 10.

    Returns:
        path (list): List of states from start to final state.
        actions_on_path (list): List of actions taken on the path.
        violations (list): List of state tuples in path that violate the transitions.
    """

    # Include the player ID as part of the state
    SWITCH_ACTION = -1
    start_state = (start_state, 0)
    # The priority queue
    frontier = []
    heapq.heappush(
        frontier, (0, start_state, [], 0, [], [])
    )  # (priority, current_state, path, cost, actions_on_path, violations)

    # Stores the minimum cost to reach each state
    cost_so_far = {start_state: 0}
    # Stores the minimum violations to reach each state
    violations_so_far = {start_state: 0}

    while frontier:
        _, (current, player_id), path, current_cost, actions_on_path, violations = heapq.heappop(frontier)

        transitions = p0_transitions if player_id == 0 else p1_transitions

        if current == final_state:
            return [start_state] + path, actions_on_path, violations

        for action in list(actions.keys()) + [SWITCH_ACTION]:
            # Skip action if next_state is out of bounds

            if action == SWITCH_ACTION:
                next_state = (current, 1 - player_id)
                new_cost = current_cost + switch_cost
                new_violations = violations_so_far[(current, player_id)]
            else:
                try:
                    next_state = next_state_without_walls(current, action, maze_size=maze_size)
                except ValueError:
                    continue

                if next_state not in transitions[current].values():
                    continue

                new_cost = current_cost + 1  # Assuming each move has a cost of 1
                next_state = (next_state, player_id)
                new_violations = violations_so_far[(current, player_id)]

            if (next_state not in cost_so_far or new_cost < cost_so_far[next_state]) or (
                next_state in violations_so_far and new_violations < violations_so_far[next_state]
            ):
                cost_so_far[next_state] = new_cost
                violations_so_far[next_state] = new_violations
                priority = new_cost + manhattan_distance(next_state[0], final_state)
                updated_violations = violations
                heapq.heappush(
                    frontier,
                    (
                        priority,
                        next_state,
                        path + [next_state],
                        new_cost,
                        actions_on_path + [action],
                        updated_violations,
                    ),
                )

    return [], [], []  # No path found
