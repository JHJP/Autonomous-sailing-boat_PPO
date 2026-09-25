# Autonomous-sailing-boat_PPO

Code, trained policies, and evaluation data for the paper

> **Navigating an Unmanned Boat to Reach Its Destination in Windy Environment Using Deep Reinforcement Learning**
> *(IEEE Access, under review)*

An unmanned boat is trained in a Unity ML-Agents simulator to reach a destination that changes every episode while a stochastic wind pushes it off course. The repository compares three deep RL algorithms on the same environment: **PPO**, **SAC** (both via ML-Agents), and **Rainbow DQN** (via Tianshou). Each is trained with 3 seeds.

![image](https://user-images.githubusercontent.com/78130703/168079365-a2f7700d-7449-4f7f-83ff-75bf58ca11e4.png)

## What you can reproduce

| Goal | Needs Unity build? | Time | Section |
|---|---|---|---|
| Regenerate the paper's figures from the saved evaluation data | No | ~1 min | [A](#a-regenerate-the-figures-no-unity-needed) |
| Re-run the frozen-policy evaluation (success rate, steps-to-goal) with the trained policies | Yes | not timed | [B](#b-re-run-the-evaluation-with-the-trained-policies) |
| Re-train all 9 runs (3 algorithms × 3 seeds, 5M steps each) from scratch | Yes | ~10 h in parallel on a 14-core Mac mini M4 Pro | [C](#c-re-train-from-scratch) |

Paper result (1,800 evaluation episodes = 3 algorithms × 3 seeds × 200 episodes; `paper_results/eval_algo_aggregate.csv`):

| Algorithm | Success rate, IQM [95% bootstrap CI] | Mean steps to goal |
|---|---|---|
| PPO | 99.5% [99.5, 99.5] | 42.0 |
| SAC | 99.2% [98.0, 100.0] | 42.1 |
| Rainbow DQN | 95.0% [92.0, 97.0] | 56.8 |

## Repository layout

```
Assets/                 Unity project: scene (Scenes/Buyoancy.unity), agent + wind scripts (Scripts/)
  Scripts/MoveToGoalAgent.cs   observation (12-d), action (2 discrete branches × 3), reward
  Scripts/WindEffect.cs        stochastic wind model
Config/
  moveToGoal.yaml       PPO hyperparameters (ML-Agents)
  sac_boat.yaml         SAC hyperparameters (ML-Agents)
training/
  rainbow_boat.py       Rainbow DQN trainer (Tianshou); hyperparameters are its CLI defaults
  run_retrain_obs12.sh  launches all 9 training runs (macOS)
analysis/
  evaluate.py           frozen-policy evaluation → success / wall / timeout / steps-to-goal
  figures.py            learning curves, success-rate plot, steps box plot, trajectories
  diagrams.py           policy / value network diagrams
  parse_results.py      training-time reward aggregation from TensorBoard logs
results/{ppo,sac,rainbow}_seed_{0,1,2}/
  MoveToGoal.onnx       trained PPO / SAC policy
  final_policy.pth      trained Rainbow DQN policy
  MoveToGoal/events.*   TensorBoard training log (used for the learning-curve figure)
paper_results/          evaluation CSVs + figures exactly as reported in the paper
```

## Setup

### 1. Unity

- Unity Editor **2022.3.62f3** (open this folder as a project in Unity Hub)
- ML-Agents Unity package `com.unity.ml-agents` **2.3.0-exp.2** (already pinned in `Packages/manifest.json`)

### 2. Python — two separate conda environments

ML-Agents 0.30 needs Python ≤3.10 and torch 1.11, while Tianshou 2.0.1 needs Python ≥3.11, so they cannot share one environment.

```bash
# (a) mlagents — PPO and SAC training
conda create -n mlagents python=3.9.13 -y
conda activate mlagents
conda install -c conda-forge grpcio -y   # the pip grpcio wheel fails to build on Apple Silicon
pip install mlagents==0.30.0
pip install protobuf==3.20.3 onnx packaging

# (b) tianshou — Rainbow DQN training, evaluation, figures
conda create -n tianshou python=3.11 -y
conda activate tianshou
pip install tianshou==2.0.1
pip install --ignore-requires-python mlagents-envs==0.30.0
pip install onnxruntime pandas scipy matplotlib tensorboard
```

Versions used for the paper: mlagents 0.30.0, torch 1.11.0, numpy 1.21.2 (env a); tianshou 2.0.1, torch 2.12.0, numpy 1.26.4, gymnasium 1.3.0, onnxruntime 1.26.0 (env b).

### 3. Build the headless simulator (needed for B and C)

The Python scripts launch the simulator as a standalone executable.

1. In Unity: `File → Build Settings…`
2. Make sure `Assets/Scenes/Buyoancy.unity` is in the scene list.
3. Choose your platform (the paper used macOS, Apple silicon) and turn on **Server Build / Headless Mode**.
4. Save the build as `Builds/BoatSailing_Mac` (so that `Builds/BoatSailing_Mac.app` exists).

On Linux or Windows, build for that platform and pass its path with `--binary` (e.g. `--binary Builds/BoatSailing_Linux`; ML-Agents adds `.x86_64` / `.exe` itself).

All commands below are run from the repository root.

## A. Regenerate the figures (no Unity needed)

```bash
conda activate tianshou
python analysis/figures.py --analysis-dir paper_results   # → paper_results/figures/
python analysis/diagrams.py                               # → outputs/figures/
```

## B. Re-run the evaluation with the trained policies

```bash
conda activate tianshou
python analysis/evaluate.py --episodes-per-seed 200 --log-trajectories   # → outputs/
python analysis/figures.py                                               # → outputs/figures/
```

The evaluation runs every policy in `results/` for 200 episodes per seed with exploration turned off (argmax action). Each episode ends as `goal`, `wall`, or `timeout`. The summary table printed at the end corresponds to `paper_results/eval_algo_aggregate.csv`. The episodes are randomized (wind and destination), so a re-run can differ slightly from the paper's numbers.

## C. Re-train from scratch

On macOS, run all 9 runs in parallel (at most `MAX` at once). On the paper's machine (Mac mini M4 Pro, CPU only) one run took about 1 h for PPO, 2 h for SAC, and 9 h for Rainbow DQN.

```bash
bash training/run_retrain_obs12.sh          # MAX=6 by default
```

Or run a single run:

```bash
# PPO / SAC
conda activate mlagents
mlagents-learn Config/moveToGoal.yaml --run-id=ppo_seed_0 --seed=0 --env=Builds/BoatSailing_Mac --no-graphics
mlagents-learn Config/sac_boat.yaml   --run-id=sac_seed_0 --seed=0 --env=Builds/BoatSailing_Mac --no-graphics

# Rainbow DQN
conda activate tianshou
python training/rainbow_boat.py --seed 0 --run-id rainbow_seed_0 --total-steps 5000000
```

Runs write to `results/<run-id>/`. ⚠️ With `--force`, a new run overwrites the trained policy of the same name that ships with this repository.

## Legacy folders

`results/MoveToGoal`, `results/TestParameters`, `results/ppo`, `results/randomParameters`, and `BoatSailing_RL_connectPy/` come from the original 2022 version of this project (Unity 2020.3, PPO only). The paper does not use them.

## Citation

The citation will be added here once the paper is published.
