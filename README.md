# mhri_intent

## Setup

First, create and activate a Conda environment with Python 3.9:

```bash
conda create -n intentmcts python=3.9
conda activate intentmcts
```

Then, install the required dependencies from the repository:

```bash
pip install -e .
```

## Running Simulations

To run the agent-to-agent simulation with the `IntentMCTS` algorithm, run the following command:

```bash
python scripts/mcts.py
```

To run baseline methods and ablation studies, override the `intent_mode` parameter in `scripts/config/mcts.yaml`.



To run the shortest path heuristic baseline, use command:
```bash
python scripts/shortest_path_heuristic.py
```


<!-- ## Render Gameplay

To render the gameplay using the user logs from an experiment, use the visualizer script. 
For example, to visualize logs from `user_logs/2024-10-11/13-03-08`, run:

```bash
python scripts/user_logs_visualizer.py --logs_dir user_logs/2024-10-11/13-03-08
ffmpeg -i visualization/stitched_%03d.png -vf format=yuv420p replay.mp4
``` -->

## Running User Study UI
To start the user study server, run the following command:
```bash
python server/1p_websockets_server.py
```

Then, open ```client/index.html``` in a browser to start the user study.