import base64
import asyncio
import random
import time
import numpy as np
import websockets
from mhri.algos.mcts_intent import MCTS, GameDynamics, GameState
from mhri.algos.shortest_path import shortest_path_min_violation_with_belief
from mhri.envs.gnomes_at_night import GnomesAtNightEnv9
from mhri.utils.canvas_parser import CanvasParser
from mhri.algos.transition_belief import TransitionBelief
from PIL import Image
import io
import json
import os
import hydra
from omegaconf import DictConfig
from datetime import datetime

MOVEMENT_KEYS_TO_ACTION = {
    "ArrowUp": 1,
    "ArrowDown": 3,
    "ArrowLeft": 2,
    "ArrowRight": 0,
}

time_limit = 180


state_space = [(i, j) for i in range(9) for j in range(9)]
action_space = [0, 1, 2, 3]

form_methods = ["heuristic", "single_step", "multi_step"]
form_methods_index = 0
random.shuffle(form_methods)

method = None

previous_grid_message = None
player_transition_belief = None
agent_dynamics = None

episode_start_time = None
timer = None
timer_enabled = False


@hydra.main(version_base=None, config_path="config", config_name="server")
def main(cfg: DictConfig):
    print(f"Configuration: \n{cfg}")

    global server_state
    server_state = ServerState(cfg)

    global canvas_parser
    canvas_parser = CanvasParser()

    global mcts_env
    mcts_env = GnomesAtNightEnv9(
        render_mode="rgb_array",
        trajectory_dump_path="None",
        train_mazes=cfg.env.train_mazes,
        communication_type="text",
    )

    # Start both servers
    asyncio.run(run_servers(cfg))


def log_message(message):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_output_path = hydra_cfg["runtime"]["output_dir"]

    # If this file doesn't exist, set the current time as the starting time
    if not os.path.exists(f"{hydra_output_path}/server_logs.txt"):
        global start_time
        start_time = time.time()

    with open(f"{hydra_output_path}/server_logs.txt", "a") as f:
        # Write time in minutes and seconds, message
        f.write(f"{int(time.time() - start_time) // 60}:{int(time.time() - start_time) % 60}: {message}\n")


async def run_servers(cfg: DictConfig):
    game_server = websockets.serve(connection_handler, cfg.server.host, cfg.server.port)
    forms_server = websockets.serve(forms_handler, cfg.server.host, cfg.server.forms_port)

    await asyncio.gather(game_server, forms_server)
    while True:
        await asyncio.sleep(3600)


class ServerState:
    def __init__(self, cfg: DictConfig):
        # Get hydra output path
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        hydra_output_path = hydra_cfg["runtime"]["output_dir"]
        self.env = GnomesAtNightEnv9(
            render_mode="rgb_array",
            trajectory_dump_path=f"{hydra_output_path}/trajectories",
            train_mazes=cfg.env.train_mazes,
            communication_type="text",
        )
        self.env.render(mode="rgb_array")
        os.makedirs(self.env.trajectory_dump_path, exist_ok=True)

        self.token_pos_list = []
        for token_pos in cfg.env.token_pos:
            token_pos = tuple(map(int, token_pos[1:-1].split(",")))
            self.token_pos_list.append(token_pos)

        self.treasure_pos_list = []
        for treasure_pos in cfg.env.treasure_pos:
            treasure_pos = tuple(map(int, treasure_pos[1:-1].split(",")))
            self.treasure_pos_list.append(treasure_pos)

        self.trials = np.arange(len(self.token_pos_list))
        np.random.shuffle(self.trials)
        self.current_trial = 0

        self.players = {}
        self.previous_canvas = {0: None, 1: None}


def render_frame(
    player_id: int,
):
    frame = server_state.env.renderer.render_client(mode="rgb_array", client_id=player_id)
    frame = Image.fromarray(frame)
    return frame


async def send_game_update(
    client_id,
    frame: Image.Image = None,
    canvas: Image.Image = None,
    completed: bool = False,
    timeout: bool = False,
    episode_complete: bool = False,
):
    frame_data = None
    if frame is not None:
        frame_buffer = io.BytesIO()
        frame.save(frame_buffer, format="PNG")
        frame_data = base64.b64encode(frame_buffer.getvalue()).decode("utf-8")

    canvas_data = None
    if canvas is not None:
        canvas_buffer = io.BytesIO()
        canvas.save(canvas_buffer, format="PNG")
        canvas_data = base64.b64encode(canvas_buffer.getvalue()).decode("utf-8")

    global timer
    if timer:
        timer.cancel()
    timer = asyncio.create_task(update_timer(client_id))

    message = json.dumps(
        {
            "type": "game_update",
            "content": {
                "rendered_frame": frame_data,
                "teammate_image": canvas_data,
                "completed": completed,
                "timeout": timeout,
                "episode_complete": episode_complete,
            },
        }
    )
    await server_state.players[client_id].send(message)


def get_shortest_path():
    path, actions_on_path, violations = shortest_path_min_violation_with_belief(
        actions={
            1: np.array([0, 1]),  # right
            2: np.array([-1, 0]),  # up
            3: np.array([0, -1]),  # left
            4: np.array([1, 0]),  # down
        },
        start_state=tuple(server_state.env.token_pos.tolist()),
        final_state=tuple(server_state.env.treasure_pos.tolist()),
        transitions=server_state.env.transitions[1],
        teammate_transition_beliefs=player_transition_belief,
        violation_cost=10,
    )
    return path, actions_on_path, violations


async def send_intent_update(intent_trajectory):
    frame = render_frame(player_id=0)
    transparent_frame = frame.convert("RGBA")
    transparent_frame.putalpha(75)

    if agent_dynamics.intent_mode == "none":
        intent = []
    if "single_step" in agent_dynamics.intent_mode:
        intent = intent_trajectory[1:2]
    else:
        intent = intent_trajectory[1:]

    red_image = Image.new("RGBA", transparent_frame.size, (255, 0, 0, 100))
    for pos in intent:
        x, y = pos
        mask = canvas_parser.get_mask_from_grid(x, y)
        mask = Image.fromarray(mask)
        transparent_frame.paste(red_image, mask=mask)

    await send_game_update(client_id=0, canvas=transparent_frame)


async def handle_client(websocket, client_id):
    global episode_start_time, timer_enabled
    try:
        async for message in websocket:
            # Process the message
            try:
                data = json.loads(message)
                message_type = data["type"]
                message_content = data["content"]
            except json.JSONDecodeError:
                print(f"(Error) Invalid JSON from Client {client_id}: {message}")
                continue

            if message_type == "start":
                global method
                method = message_content["method"]

                log_message(f"Session ID: {message_content['session']}")

                global player_transition_belief
                player_transition_belief = TransitionBelief(state_space, action_space, mcts_env.maze_shape)

                global agent_dynamics
                agent_dynamics = GameDynamics(
                    player_id=1,
                    player_transitions=mcts_env.transitions[1],
                    teammate_transitions_belief=player_transition_belief,
                    teammate_intent=[],  # The intent received from the teammate
                    intent_mode=method,
                )

                log_message(f"Starting game with method ({method}).")

                server_state.current_trial = 0
                np.random.shuffle(server_state.trials)
                server_state.env.reset(
                    token_pos=server_state.token_pos_list[server_state.trials[server_state.current_trial]],
                    treasure_pos=server_state.treasure_pos_list[server_state.trials[server_state.current_trial]],
                )

                episode_start_time = time.time()
                timer_enabled = True

                frame = render_frame(client_id)
                await send_game_update(client_id, frame=frame)
                await send_intent_update(intent_trajectory=[])
                log_message(
                    f"Starting game with token_pos: {server_state.env.token_pos}, treasure_pos: {server_state.env.treasure_pos}."
                )

                global timer
                if timer:
                    timer.cancel()
                timer = asyncio.create_task(update_timer(client_id))

            elif message_type == "move":
                # Check if it's the player's turn
                if server_state.env.current_player != client_id:
                    print(f"(Ignored) Client {client_id}: {message}")
                    continue

                elapsed_time = min(int(time.time() - episode_start_time), time_limit)
                if elapsed_time >= time_limit:
                    server_state.env.current_player = 1  # Blocks the human from taking any more actions
                    continue

                print(f"Client {client_id}: Move {message_content}")
                log_message(f"Human: Move {MOVEMENT_KEYS_TO_ACTION[message_content]}")
                action = {
                    "control": MOVEMENT_KEYS_TO_ACTION[message_content],
                    "comm": "",
                }
                previous_token_pos = tuple(server_state.env.token_pos.tolist())
                obs, reward, terminated, truncated, info = server_state.env.step(action)
                log_message(f"Token Pos: {server_state.env.token_pos}")

                new_token_pos = tuple(server_state.env.token_pos.tolist())
                frame = render_frame(client_id)
                await send_game_update(client_id, frame=frame)

                if previous_token_pos != new_token_pos:
                    player_transition_belief.update_with_trajectory(
                        [(previous_token_pos, MOVEMENT_KEYS_TO_ACTION[message_content])]
                    )
                    log_message(
                        f"Transition Belief: {np.array2string(player_transition_belief.belief, separator=',', formatter={'float_kind': lambda x: f'{x}'})}"
                    )

            elif message_type == "chat":
                elapsed_time = min(int(time.time() - episode_start_time), time_limit)
                if elapsed_time >= time_limit:
                    server_state.env.current_player = 1  # Blocks the human from taking any more actions
                    continue

                # Check if it's the player's turn
                if server_state.env.current_player != client_id:
                    print(f"(Ignored) Client {client_id}: {message}")
                    continue

                print(f"Client {client_id}: Chat {message_content}")
                log_message("Human: Switch to AI.")
                action = {
                    "control": 4,
                    "comm": message_content,
                }
                obs, reward, terminated, truncated, info = server_state.env.step(action)

                frame = render_frame(client_id)
                await send_game_update(client_id, frame=frame)

                # AI has control now. Do MCTS.
                while not terminated and not truncated:
                    elapsed_time = min(int(time.time() - episode_start_time), time_limit)
                    if elapsed_time >= time_limit:
                        break

                    if method == "heuristic":
                        path, actions_on_path, violations = get_shortest_path()
                        if (
                            actions_on_path[0] - 1
                            in server_state.env.transitions[1][tuple(server_state.env.token_pos.tolist())]
                        ):
                            action = actions_on_path[0] - 1
                        else:
                            action = 4
                    else:
                        current_state = GameState(
                            tuple(server_state.env.token_pos.tolist()),
                            tuple(server_state.env.treasure_pos.tolist()),
                            server_state.env.current_player,
                        )
                        mcts = MCTS(
                            initial_state=current_state,
                            game_dynamics=agent_dynamics,
                            n_iterations=100,
                            exploration_constant=np.sqrt(2),
                            gamma=0.99,
                        )
                        action = mcts.search()
                    gym_action = {
                        "control": action,
                        "comm": "",
                    }
                    obs, reward, terminated, truncated, info = server_state.env.step(gym_action)

                    if action != 4:  # AI took a move action
                        log_message(f"AI: Move {action}")
                        log_message(f"Token Pos: {server_state.env.token_pos}")

                        # In single_step mode, intent is good for one move
                        if method == "single_step":
                            agent_dynamics.teammate_intent = []

                    time.sleep(0.3)
                    frame = render_frame(client_id)
                    await send_game_update(client_id, frame=frame)

                    if action == 4:
                        # Generate AI intent
                        if agent_dynamics.intent_mode != "none" and method != "heuristic":
                            intent_trajectory = get_shortest_path()[0]
                            log_message(f"AI: Intent {intent_trajectory}")
                            await send_intent_update(intent_trajectory)
                        break

            elif message_type == "grid":
                # Check if it's the player's turn
                if server_state.env.current_player != client_id:
                    print(f"(Ignored) Client {client_id}: {message}")
                    continue

                global previous_grid_message
                if message_content != previous_grid_message:
                    previous_grid_message = message_content
                    intent_states = []
                    for i, cell in enumerate(message_content):
                        if cell == 1:
                            intent_states.append((i // 9, i % 9))

                    if len(intent_states) > 0:
                        # First find the state closest to the current token position
                        closest_state = None
                        min_distance = float("inf")
                        for state in intent_states:
                            distance = np.linalg.norm(np.array(state) - np.array(server_state.env.token_pos), ord=1)
                            if distance < min_distance:
                                min_distance = distance
                                closest_state = state

                        intent = [closest_state]
                        intent_states.remove(closest_state)
                        while len(intent_states) > 0:
                            next_state = None
                            min_distance = float("inf")
                            for state in intent_states:
                                distance = np.linalg.norm(np.array(state) - np.array(intent[-1]), ord=1)
                                if distance < min_distance:
                                    min_distance = distance
                                    next_state = state
                            intent.append(next_state)
                            intent_states.remove(next_state)
                    else:
                        intent = []

                    # Always include the current token position at the beginning of the intent
                    if tuple(server_state.env.token_pos.tolist()) not in intent:
                        intent.insert(0, tuple(server_state.env.token_pos.tolist()))

                    log_message(f"Client {client_id}: Human intent: {intent}")

                    agent_dynamics.teammate_intent = intent

            elif message_type == "start_next_episode":
                log_message(f"Client {client_id}: Starting next episode.")
                await start_next_episode(client_id)
                continue

            else:
                print(f"(Error) Unknown message type from Client {client_id}: {message_type}")

            if message_type in ["move", "chat"] and (terminated or truncated):
                if timer:
                    timer.cancel()
                timer_enabled = False

                server_state.env.current_player = 1  # Blocks the human from taking any more actions
                log_message(f"Client {client_id}: Success!")

                completed = server_state.current_trial == len(server_state.token_pos_list) - 1
                if completed:
                    log_message("All trials completed!")
                await send_game_update(client_id, episode_complete=True, completed=completed)

    finally:
        del server_state.players[client_id]
        if timer:
            timer.cancel()


async def connection_handler(websocket, path):
    client_id = len(server_state.players)
    server_state.players[client_id] = websocket
    print(f"Client {client_id} connected")
    if client_id < 1:
        await handle_client(websocket, client_id)
    else:
        await websocket.close(1008, "Game is full.")


async def forms_handler(websocket, path):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    hydra_output_path = hydra_cfg["runtime"]["output_dir"]
    log_file = f"{hydra_output_path}/forms_log.json"

    print(f"Forms server started on ws://{websocket.host}:{websocket.port}")

    async for message in websocket:
        try:
            data = json.loads(message)
            timestamp = datetime.now().isoformat()
            data["timestamp"] = timestamp

            with open(log_file, "a") as f:
                json.dump(data, f)
                f.write("\n")

            print(f"Logged message: {data}")

            if data["type"] == "getNextMethod":
                global form_methods_index
                await websocket.send(
                    json.dumps(
                        {
                            "type": "nextMethod",
                            "value": form_methods[form_methods_index],
                            "session": form_methods_index + 1,
                        }
                    )
                )
                form_methods_index += 1
                if form_methods_index == len(form_methods):
                    form_methods_index = -1
        except json.JSONDecodeError:
            print(f"Invalid JSON received: {message}")


async def update_timer(client_id):
    global episode_start_time
    while True and timer_enabled:
        elapsed_time = min(int(time.time() - episode_start_time), time_limit)
        remaining_time = time_limit - elapsed_time
        minutes, seconds = divmod(remaining_time, 60)

        if elapsed_time <= time_limit:
            message = json.dumps(
                {
                    "type": "game_update",
                    "content": {
                        "timer": f"{minutes:02d}:{seconds:02d}",
                    },
                }
            )
            await server_state.players[client_id].send(message)

        if elapsed_time >= time_limit and timer_enabled:
            log_message(f"Client {client_id}: Timeout!")
            completed = server_state.current_trial == len(server_state.token_pos_list) - 1
            if completed:
                log_message("All trials completed!")
            await send_game_update(client_id, timeout=True, completed=completed)
            timer.cancel()

        await asyncio.sleep(1)


async def start_next_episode(client_id):
    global episode_start_time
    server_state.current_trial += 1
    if server_state.current_trial < len(server_state.token_pos_list):
        server_state.env.reset(
            token_pos=server_state.token_pos_list[server_state.trials[server_state.current_trial]],
            treasure_pos=server_state.treasure_pos_list[server_state.trials[server_state.current_trial]],
        )
        episode_start_time = time.time()
        global timer_enabled
        timer_enabled = True

        frame = render_frame(client_id)
        await send_game_update(client_id, frame=frame)
        await send_intent_update(intent_trajectory=[])
        global previous_grid_message
        previous_grid_message = None
        log_message(
            f"Starting game with token_pos: {server_state.env.token_pos}, treasure_pos: {server_state.env.treasure_pos}."
        )
    else:
        log_message("All trials completed!")
        await send_game_update(client_id, completed=True)


if __name__ == "__main__":
    main()
