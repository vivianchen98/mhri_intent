import os
import json
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import argparse
from mhri.algos.transition_belief import TransitionBelief
from mhri.envs.gnomes_at_night import GnomesAtNightEnv9
from mhri.utils.canvas_parser import CanvasParser

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--logs_dir", type=str, default="user_logs/2024-10-11/13-03-08")
args = parser.parse_args()

# global variable to store logs
main_logs = {}


# function to parse logs
def parse_logs(logs_dir):
    current_method = None
    ai_in_control = False
    numpy_array_buffer = ""

    server_log = os.path.join(logs_dir, "server_logs.txt")
    forms_log = os.path.join(logs_dir, "forms_log.json")

    with open(forms_log, "r") as f:
        for line in f:
            forms_data = json.loads(line)
            if "data" in forms_data:
                if "alias" in forms_data["data"]:
                    alias = forms_data["data"]["alias"]
                    print(f"Loading user: {alias}")
                    break

    with open(server_log, "r") as f:
        for line in f:
            if not line.strip():  # Ignore empty lines
                continue

            if line.startswith(" "):  # Continuation of a numpy array
                line = line.strip()
                numpy_array_buffer += line
                if line.endswith("]]"):  # End of a numpy array
                    belief_array = np.array(eval(numpy_array_buffer))
                    main_logs[current_method][-1]["belief_array"] = belief_array
                continue

            # Regular lines
            time_str, message = line.split(": ", 1)
            message = message.strip()

            if message.startswith("Starting game with method"):  # Start of a session
                current_method = message.split(" ")[-1][1:-2]
                main_logs[current_method] = []

            elif message.startswith("Starting game with token_pos"):  # Start of a trial
                # Example message:
                # Starting game with token_pos: [4 0], treasure_pos: [4 8].
                token_pos = message.split("token_pos: ")[-1].split(",")[0].strip()[1:-1]
                treasure_pos = message.split("treasure_pos: ")[-1].strip()[1:-2]
                token_pos = tuple(map(int, token_pos.split()))
                treasure_pos = tuple(map(int, treasure_pos.split()))
                main_logs[current_method].append(
                    {
                        "token_pos": token_pos,
                        "treasure_pos": treasure_pos,
                        "human_in_control": True,
                        "human_intent": None,
                        "ai_intent": None,
                    }
                )

            elif "Move" in message:
                if ai_in_control and "Human" in message:
                    ai_in_control = False
                main_logs[current_method].append({"human_in_control": not ai_in_control})

            elif "Token Pos" in message:
                token_pos = message.split("Token Pos: ")[-1].strip()[1:-1]
                token_pos = tuple(map(int, token_pos.split()))
                main_logs[current_method][-1]["token_pos"] = token_pos

            elif "Switch" in message:
                ai_in_control = True
                main_logs[current_method].append({"human_in_control": False})

            elif "Human intent" in message:
                human_intent = message.split("Human intent: ")[-1].strip()
                human_intent = np.array(eval(human_intent))
                main_logs[current_method][-1]["human_intent"] = human_intent

            elif "AI: Intent" in message:
                ai_intent = message.split("AI: Intent ")[-1].strip()
                ai_intent = np.array(eval(ai_intent))
                main_logs[current_method][-1]["ai_intent"] = ai_intent

            elif "Transition Belief" in message:
                transition_belief = message.split("Transition Belief: ")[-1].strip()
                numpy_array_buffer = transition_belief


if __name__ == "__main__":
    # create directory if not exists, clear this directory if has files, ask for permission first
    if not os.path.exists("visualization"):
        os.makedirs("visualization")
    else:
        if len(os.listdir("visualization")) > 0:
            print("Directory 'visualization' is not empty. Do you want to clear it? (y/n)")
            response = input()
            if response.lower() == "y":
                for file in os.listdir("visualization"):
                    os.remove(os.path.join("visualization", file))
            else:
                print("Exiting...")
                exit()

    parse_logs(args.logs_dir)

    # Initialize the environment
    env = GnomesAtNightEnv9(
        render_mode="rgb_array",
        trajectory_dump_path=None,
        train_mazes=["human"],
        communication_type="text",
    )
    env.render(mode="rgb_array")
    state_space = [(i, j) for i in range(9) for j in range(9)]
    action_space = [0, 1, 2, 3]
    transition_belief = TransitionBelief(state_space, action_space, env.maze_shape)

    red_canvas_parser = CanvasParser()
    green_canvas_parser = CanvasParser(detection_size=48)

    token_pos = None
    treasure_pos = None
    human_in_control = None
    human_intent = []
    ai_intent = []
    total_frames = len(main_logs["multi_step"])
    for i in tqdm(range(total_frames)):
        step_log = main_logs["multi_step"][i]
        if "token_pos" in step_log and step_log["token_pos"] is not None:
            token_pos = step_log["token_pos"]
        if "treasure_pos" in step_log and step_log["treasure_pos"] is not None:
            treasure_pos = step_log["treasure_pos"]
        if "belief_array" in step_log:
            transition_belief.belief = step_log["belief_array"]
        if "human_intent" in step_log:
            human_intent = step_log["human_intent"]
        if "ai_intent" in step_log:
            ai_intent = step_log["ai_intent"]
        if "human_in_control" in step_log:
            human_in_control = step_log["human_in_control"]

        env.reset(token_pos=token_pos, treasure_pos=treasure_pos)
        env.current_player = 1 - human_in_control

        game_frame = env.renderer.render_client(mode="rgb_array", client_id=1)
        game_frame = Image.fromarray(game_frame).convert("RGBA")

        if ai_intent is not None:
            overlay = Image.new("RGBA", game_frame.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for pos in ai_intent:
                x, y = pos
                if (x, y) != token_pos:
                    mask = Image.fromarray(red_canvas_parser.get_mask_from_grid(x, y))
                    draw.bitmap((0, 0), mask, fill=(255, 0, 0, 100))
            game_frame = Image.alpha_composite(game_frame, overlay)

        if human_intent is not None:
            overlay = Image.new("RGBA", game_frame.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            for pos in human_intent:
                x, y = pos
                if (x, y) != token_pos:
                    mask = Image.fromarray(green_canvas_parser.get_mask_from_grid(x, y))
                    draw.bitmap((0, 0), mask, fill=(0, 255, 0, 64))

            game_frame = Image.alpha_composite(game_frame, overlay)

        belief_frame = transition_belief.render_belief_heatmap(
            show_colorbar=True,
            save_path=None,
        )

        # Stitch the game frame and the belief frame
        stitched_frame = Image.new("RGB", (game_frame.width + belief_frame.width, game_frame.height))
        stitched_frame.paste(game_frame, (0, 0))
        stitched_frame.paste(belief_frame, (game_frame.width, 0))
        stitched_frame.save(f"visualization/stitched_{i:03d}.png")
