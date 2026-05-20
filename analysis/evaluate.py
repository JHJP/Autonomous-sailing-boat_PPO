"""Frozen-policy evaluation on Mac Standalone Unity env.

Loads PPO/SAC .onnx (ml-agents) and Rainbow .pth (Tianshou) policies and runs N
episodes per (algo, seed) under condition E3 (stochastic wind + changing destination,
matching the training distribution). Decomposes outcomes that training-time mean
reward cannot: goal / wall / timeout / steps-to-goal.

Outputs:
    analysis/eval_per_episode.csv      one row per episode
    analysis/eval_per_seed_summary.csv aggregates per (algo, seed)
    analysis/eval_algo_aggregate.csv   IQM + 95% bootstrap CI per algo (Agarwal 2021)

Usage:
    conda activate tianshou
    python code/analysis/evaluate.py --episodes-per-seed 200

Notes:
    - Uses raw mlagents-envs API (16 agents in Buyoancy.unity scene; UnityToGymWrapper
      hangs on multi-agent).
    - PPO/SAC use ONNX `deterministic_discrete_actions` output (argmax, no exploration).
    - Rainbow uses Tianshou C51Policy in eval mode (NoisyLinear noise disabled).
    - Outcome decomposition: goal if terminal_reward >= +0.5, wall if <= -0.5, timeout otherwise.
"""
from __future__ import annotations

import argparse
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import onnxruntime as ort

warnings.filterwarnings("ignore")

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.data import Batch
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import NoisyLinear


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = PROJECT_ROOT / "code" / "Builds" / "BoatSailing_Mac"


# ─────────────────────── Policy abstractions ───────────────────────


@dataclass
class OnnxPolicy:
    """ml-agents PPO/SAC ONNX inference."""
    path: Path
    n_branches: int = 2
    branch_sizes: tuple = (3, 3)

    def __post_init__(self):
        self.sess = ort.InferenceSession(str(self.path))
        # action_masks size = sum of branch_sizes (3 + 3 = 6)
        self.mask_size = sum(self.branch_sizes)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """obs: (n, obs_dim) float32 → returns (n, 2) int32 joint discrete actions."""
        n = obs.shape[0]
        action_masks = np.ones((n, self.mask_size), dtype=np.float32)
        out = self.sess.run(
            ["deterministic_discrete_actions"],
            {"obs_0": obs.astype(np.float32), "action_masks": action_masks},
        )
        return out[0].astype(np.int32)  # (n, 2)


@dataclass
class RainbowPthPolicy:
    """Tianshou Rainbow .pth inference (eval mode → NoisyLinear noise off)."""
    path: Path
    obs_dim: int = 11
    n_actions: int = 9
    hidden_sizes: tuple = (128, 128)
    num_atoms: int = 51
    noisy_std: float = 0.1
    v_min: float = -1.0
    v_max: float = 1.0
    device: str = "cpu"

    def __post_init__(self):
        def noisy_linear(x, y):
            return NoisyLinear(x, y, noisy_std=self.noisy_std)

        net = Net(
            state_shape=(self.obs_dim,),
            action_shape=self.n_actions,
            hidden_sizes=self.hidden_sizes,
            softmax=True,
            num_atoms=self.num_atoms,
            dueling_param=(
                {"linear_layer": noisy_linear},
                {"linear_layer": noisy_linear},
            ),
        )
        self.policy = C51Policy(
            model=net,
            action_space=gym.spaces.Discrete(self.n_actions),
            observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32),
            num_atoms=self.num_atoms,
            v_min=self.v_min,
            v_max=self.v_max,
            eps_training=0.0,
            eps_inference=0.0,
        ).to(self.device)
        ckpt = torch.load(self.path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.policy.eval()  # disable NoisyLinear noise

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """obs: (n, obs_dim) → returns (n, 2) int32 joint (steer, motor)."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.policy(Batch(obs=obs_t, info={}))
        act = out.act
        action_int = (act.cpu().numpy() if hasattr(act, "cpu") else np.asarray(act)).astype(np.int64).reshape(-1)
        # Unflatten 0-8 → (steer, motor)
        joint = np.stack([action_int // 3, action_int % 3], axis=1).astype(np.int32)
        return joint


# ─────────────────────── Evaluation loop ───────────────────────


def classify(reward: float, interrupted: bool, goal_threshold: float = 0.5, wall_threshold: float = -0.5) -> str:
    if interrupted:
        return "timeout"
    if reward >= goal_threshold:
        return "goal"
    if reward <= wall_threshold:
        return "wall"
    return "timeout"  # fallback (shouldn't trigger given sparse ±1 reward)


def evaluate_one(
    policy,
    binary: Path,
    worker_id: int,
    seed: int,
    n_episodes: int,
    max_env_steps: int,
    algo: str,
    seed_label: str,
) -> list[dict]:
    """Run frozen policy until n_episodes accumulated across 16 agents."""
    env = UnityEnvironment(
        file_name=str(binary),
        worker_id=worker_id,
        seed=seed,
        no_graphics=True,
        additional_args=["-logFile", f"/tmp/unity_eval_{worker_id}.log"],
    )
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    ep_records: list[dict] = []
    ep_steps: dict[int, int] = {}
    env_steps = 0
    episode_idx_per_agent: dict[int, int] = {}

    while len(ep_records) < n_episodes and env_steps < max_env_steps:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # Process terminal: close episodes.
        for idx, agent_id in enumerate(terminal_steps.agent_id):
            agent_id = int(agent_id)
            term_rew = float(terminal_steps.reward[idx])
            interrupted = bool(terminal_steps.interrupted[idx])
            outcome = classify(term_rew, interrupted)
            ep_records.append({
                "algo": algo,
                "seed": seed_label,
                "agent_id": agent_id,
                "episode_idx": episode_idx_per_agent.get(agent_id, 0),
                "outcome": outcome,
                "steps": ep_steps.get(agent_id, 0),
                "terminal_reward": term_rew,
                "interrupted": interrupted,
            })
            episode_idx_per_agent[agent_id] = episode_idx_per_agent.get(agent_id, 0) + 1
            ep_steps[agent_id] = 0
            if len(ep_records) >= n_episodes:
                break

        if len(ep_records) >= n_episodes:
            break

        # Process decision: predict + send actions.
        if len(decision_steps) > 0:
            agent_ids = [int(a) for a in decision_steps.agent_id]
            obs = np.asarray(decision_steps.obs[0], dtype=np.float32)
            joint = policy.predict(obs)  # (n, 2) int32
            env.set_actions(behavior_name, ActionTuple(discrete=joint))
            for aid in agent_ids:
                ep_steps[aid] = ep_steps.get(aid, 0) + 1
            env_steps += len(agent_ids)

        env.step()

    env.close()
    return ep_records[:n_episodes]


# ─────────────────────── Aggregation ───────────────────────


def bootstrap_iqm_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                     rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    rng = rng or np.random.default_rng(0)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        lo_q, hi_q = np.quantile(sample, [0.25, 0.75])
        mask = (sample >= lo_q) & (sample <= hi_q)
        boots[i] = sample[mask].mean() if mask.any() else sample.mean()
    iqm = float(boots.mean())
    ci_lo, ci_hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return iqm, float(ci_lo), float(ci_hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "code" / "results")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "analysis")
    p.add_argument("--episodes-per-seed", type=int, default=200)
    p.add_argument("--max-env-steps", type=int, default=200_000,
                   help="Safety cap on env steps per (algo, seed) eval — avoid hangs.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--algos", type=str, nargs="+", default=["sac", "ppo", "rainbow"])
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_episodes: list[dict] = []
    worker_id_counter = 50  # avoid clash with training-time worker_ids 0,1,2

    for algo in args.algos:
        for seed in args.seeds:
            run_id = f"{algo}_seed_{seed}"
            run_dir = args.results_dir / run_id

            if algo == "rainbow":
                policy_path = run_dir / "final_policy.pth"
                if not policy_path.exists():
                    print(f"[skip] {run_id}: {policy_path} missing")
                    continue
                policy = RainbowPthPolicy(path=policy_path)
            else:
                policy_path = run_dir / "MoveToGoal.onnx"
                if not policy_path.exists():
                    print(f"[skip] {run_id}: {policy_path} missing")
                    continue
                policy = OnnxPolicy(path=policy_path)

            print(f"[eval] {run_id} (worker_id={worker_id_counter}) ...")
            recs = evaluate_one(
                policy=policy,
                binary=args.binary,
                worker_id=worker_id_counter,
                seed=seed + 1000,  # offset eval seeds from training seeds
                n_episodes=args.episodes_per_seed,
                max_env_steps=args.max_env_steps,
                algo=algo,
                seed_label=str(seed),
            )
            worker_id_counter += 1
            all_episodes.extend(recs)
            outcomes = [r["outcome"] for r in recs]
            n_goal = sum(1 for o in outcomes if o == "goal")
            n_wall = sum(1 for o in outcomes if o == "wall")
            n_to = sum(1 for o in outcomes if o == "timeout")
            print(f"  → n={len(recs)} goal={n_goal} wall={n_wall} timeout={n_to} succ_rate={n_goal/max(len(recs),1):.3f}")

    # Per-episode dump
    per_ep_path = args.out_dir / "eval_per_episode.csv"
    if all_episodes:
        with open(per_ep_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_episodes[0].keys()))
            w.writeheader()
            w.writerows(all_episodes)
    print(f"[done] per-episode → {per_ep_path}")

    # Per-seed summary
    summary_rows = []
    per_algo_seed_succ: dict[str, list[float]] = {}
    per_algo_seed_steps: dict[str, list[float]] = {}
    for algo in args.algos:
        for seed in args.seeds:
            recs = [r for r in all_episodes if r["algo"] == algo and r["seed"] == str(seed)]
            if not recs:
                continue
            outcomes = [r["outcome"] for r in recs]
            steps_to_goal = [r["steps"] for r in recs if r["outcome"] == "goal"]
            n = len(recs)
            n_goal = sum(1 for o in outcomes if o == "goal")
            n_wall = sum(1 for o in outcomes if o == "wall")
            n_to = sum(1 for o in outcomes if o == "timeout")
            row = {
                "algo": algo,
                "seed": seed,
                "n_episodes": n,
                "n_goal": n_goal,
                "n_wall": n_wall,
                "n_timeout": n_to,
                "success_rate": n_goal / max(n, 1),
                "wall_rate": n_wall / max(n, 1),
                "timeout_rate": n_to / max(n, 1),
                "mean_steps_to_goal": float(np.mean(steps_to_goal)) if steps_to_goal else float("nan"),
                "std_steps_to_goal": float(np.std(steps_to_goal, ddof=1)) if len(steps_to_goal) > 1 else float("nan"),
            }
            summary_rows.append(row)
            per_algo_seed_succ.setdefault(algo, []).append(row["success_rate"])
            per_algo_seed_steps.setdefault(algo, []).extend(steps_to_goal)

    summary_path = args.out_dir / "eval_per_seed_summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"[done] per-seed summary → {summary_path}")

    # Per-algo aggregate (IQM + bootstrap CI per Agarwal 2021)
    agg_rows = []
    for algo, succs in per_algo_seed_succ.items():
        arr = np.array(succs)
        iqm, lo, hi = bootstrap_iqm_ci(arr, rng=np.random.default_rng(hash(algo) & 0xFFFFFFFF))
        steps = per_algo_seed_steps.get(algo, [])
        agg_rows.append({
            "algo": algo,
            "n_seeds": len(succs),
            "mean_success": float(arr.mean()),
            "std_success": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
            "iqm_success": iqm,
            "ci_lo": lo,
            "ci_hi": hi,
            "mean_steps_to_goal_overall": float(np.mean(steps)) if steps else float("nan"),
            "std_steps_to_goal_overall": float(np.std(steps, ddof=1)) if len(steps) > 1 else float("nan"),
        })

    agg_path = args.out_dir / "eval_algo_aggregate.csv"
    if agg_rows:
        with open(agg_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            w.writeheader()
            w.writerows(agg_rows)
    print(f"[done] algo aggregate → {agg_path}")

    # Console summary
    print()
    print("=" * 70)
    print(f"{'algo':<10} {'n_seeds':<8} {'mean':<8} {'IQM':<8} {'95% CI':<22} {'steps':<10}")
    print("=" * 70)
    for r in agg_rows:
        ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        print(f"{r['algo']:<10} {r['n_seeds']:<8} {r['mean_success']:<8.4f} {r['iqm_success']:<8.4f} {ci:<22} {r['mean_steps_to_goal_overall']:<10.1f}")


if __name__ == "__main__":
    main()
