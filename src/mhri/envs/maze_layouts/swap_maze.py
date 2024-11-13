import json
import os


def read_json_file(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data


def swap_xy(data):
    # swap x y coordinates in treasures
    for d in data["treasures"]:
        d["x"], d["y"] = d["y"], d["x"]

    for d in data["p1-walls"]:
        d["from"]["x"], d["from"]["y"] = d["from"]["y"], d["from"]["x"]
        d["to"]["x"], d["to"]["y"] = d["to"]["y"], d["to"]["x"]

    for d in data["p2-walls"]:
        d["from"]["x"], d["from"]["y"] = d["from"]["y"], d["from"]["x"]
        d["to"]["x"], d["to"]["y"] = d["to"]["y"], d["to"]["x"]
    return data


# iterate over all json files in the directory
for file_name in os.listdir("src/mhri/envs/maze_layouts"):
    if file_name.endswith(".json"):
        file_path = os.path.join("src/mhri/envs/maze_layouts", file_name)
        data = read_json_file(file_path)  # read json file
        data = swap_xy(data)  # swap x y coordinates
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)  # write to the same file
