"""Parse training-time TensorBoard event files and aggregate per-algorithm metrics.

Handles both schemas:
- ml-agents (PPO + SAC): tags `Environment/Cumulative Reward`, `Environment/Episode Length`
- Tianshou (Rainbow):    tags `train/reward`, `train/len`, `test/reward`, `test/len`

Outputs:
- analysis/per_seed_summary.csv  one row per (algo, seed) with last-N-window mean + std
- analysis/algo_aggregate.csv    one row per algo with IQM + 95% stratified bootstrap CI
- analysis/learning_curves.png   per-algo mean curve + bootstrap-band
- analysis/per_seed_curves.png   thin per-seed curves overlay

Caveats:
- Cumulative reward alone CANNOT decompose success / wall / timeout. For exact final
  success rate, run a separate frozen-policy evaluation (evaluate.py, requires Mac
  Standalone). This script reports proxy success rate = (reward + 1) / 2 ASSUMING
  no timeouts; flag in plot legend.
- 'Last-N-window' = last `--last-n-windows` summary updates. ml-agents writes one
  summary per `summary_freq` env steps (config-side, default 2000 for sac_boat.yaml).
  Tianshou writes one per epoch. Different time bases; document in caption.

Setup — recommend a SEPARATE conda env to avoid bumping mlagents 0.30's pinned
numpy 1.21.2 / torch 1.11.0:

    conda create -n analysis python=3.11 -y
    conda activate analysis
    pip install numpy pandas matplotlib tensorboard

Usage:
    conda activate analysis
    python code/analysis/parse_results.py \\
        --results-dir code/results \\
        --runs ppo:ppo_seed_0,ppo_seed_1,ppo_seed_2 \\
        --runs sac:sac_seed_0,sac_seed_1,sac_seed_2 \\
        --runs rainbow:rainbow_seed_0,rainbow_seed_1,rainbow_seed_2 \\
        --out-dir analysis
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REWARD_TAG_CANDIDATES = (
    "Environment/Cumulative Reward",  # ml-agents
    "test/reward",                    # Tianshou test
    "train/reward",                   # Tianshou train (fallback)
)
LENGTH_TAG_CANDIDATES = (
    "Environment/Episode Length",
    "test/len",
    "train/len",
)


def find_event_dir(run_dir: Path) -> Path | None:
    """ml-agents nests events under <run_dir>/<behavior_name>/. Tianshou writes
    them directly under <run_dir>/. Return the dir containing the tfevents file."""
    if any(p.name.startswith("events.out.tfevents") for p in run_dir.iterdir() if p.is_file()):
        return run_dir
    for sub in run_dir.iterdir():
        if sub.is_dir() and any(p.name.startswith("events.out.tfevents") for p in sub.iterdir() if p.is_file()):
            return sub
    return None


def pick_tag(available: set[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in available:
            return c
    return None


def load_run(run_dir: Path) -> pd.DataFrame | None:
    event_dir = find_event_dir(run_dir)
    if event_dir is None:
        print(f"[warn] no events file under {run_dir}")
        return None
    ea = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    rt = pick_tag(tags, REWARD_TAG_CANDIDATES)
    lt = pick_tag(tags, LENGTH_TAG_CANDIDATES)
    if rt is None:
        print(f"[warn] no reward tag in {event_dir} (tags={sorted(tags)[:8]}...)")
        return None
    reward_events = ea.Scalars(rt)
    df = pd.DataFrame({
        "step": [e.step for e in reward_events],
        "reward": [e.value for e in reward_events],
    })
    if lt is not None:
        length_events = ea.Scalars(lt)
        len_df = pd.DataFrame({
            "step": [e.step for e in length_events],
            "length": [e.value for e in length_events],
        })
        df = df.merge(len_df, on="step", how="outer").sort_values("step")
    df["reward_tag"] = rt
    return df


def stratified_bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                            rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """IQM + (lo, hi) 95% bootstrap CI. Agarwal et al. 2021 protocol.
    values: 1-D array of per-seed scalar (e.g., last-N-window mean reward)."""
    rng = rng or np.random.default_rng(0)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        # Interquartile mean: mean of middle 50%.
        lo_q, hi_q = np.quantile(sample, [0.25, 0.75])
        mask = (sample >= lo_q) & (sample <= hi_q)
        boots[i] = sample[mask].mean() if mask.any() else sample.mean()
    iqm = np.mean(boots)
    ci_lo, ci_hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return iqm, ci_lo, ci_hi


@dataclass
class AlgoSpec:
    name: str
    runs: list[str]  # run-id subdirectory names under results-dir


def parse_algo_spec(spec: str) -> AlgoSpec:
    name, runs_csv = spec.split(":", 1)
    return AlgoSpec(name=name.strip(), runs=[r.strip() for r in runs_csv.split(",") if r.strip()])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("code/results"),
                   help="Directory containing run-id subdirs.")
    p.add_argument("--runs", action="append", required=True,
                   help="Repeat per algo: --runs algoname:run-id-1,run-id-2,run-id-3")
    p.add_argument("--out-dir", type=Path, default=Path("analysis"))
    p.add_argument("--last-n-windows", type=int, default=20,
                   help="Final summary windows averaged for per-seed final-perf scalar.")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--success-from-reward", action="store_true",
                   help="Compute proxy success rate = (reward + 1) / 2. ASSUMES no timeouts.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    algo_specs = [parse_algo_spec(s) for s in args.runs]

    per_seed_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    all_curves: dict[str, list[pd.DataFrame]] = {}

    for algo in algo_specs:
        seed_finals: list[float] = []
        algo_curves: list[pd.DataFrame] = []
        for run_id in algo.runs:
            run_dir = args.results_dir / run_id
            if not run_dir.exists():
                print(f"[skip] {algo.name}/{run_id} — dir missing")
                continue
            df = load_run(run_dir)
            if df is None:
                continue
            df["algo"] = algo.name
            df["run_id"] = run_id
            algo_curves.append(df)

            final_window = df["reward"].tail(args.last_n_windows)
            final_mean = float(final_window.mean())
            final_std = float(final_window.std())
            row = {
                "algo": algo.name,
                "run_id": run_id,
                "final_mean_reward": final_mean,
                "final_std_reward": final_std,
                "n_summaries": len(df),
                "last_step": int(df["step"].iloc[-1]),
                "reward_tag": df["reward_tag"].iloc[-1],
            }
            if args.success_from_reward:
                row["proxy_success_rate"] = (final_mean + 1.0) / 2.0
            per_seed_rows.append(row)
            seed_finals.append(final_mean)

        all_curves[algo.name] = algo_curves

        if len(seed_finals) >= 2:
            iqm, lo, hi = stratified_bootstrap_ci(
                np.array(seed_finals), n_boot=args.n_boot,
                rng=np.random.default_rng(hash(algo.name) & 0xFFFFFFFF),
            )
            aggregate_rows.append({
                "algo": algo.name,
                "n_seeds": len(seed_finals),
                "iqm_final_reward": iqm,
                "ci_lo": lo,
                "ci_hi": hi,
                "mean_final_reward": float(np.mean(seed_finals)),
                "std_final_reward": float(np.std(seed_finals, ddof=1)),
            })

    per_seed_df = pd.DataFrame(per_seed_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)
    per_seed_df.to_csv(args.out_dir / "per_seed_summary.csv", index=False)
    aggregate_df.to_csv(args.out_dir / "algo_aggregate.csv", index=False)
    print(f"[done] wrote {args.out_dir / 'per_seed_summary.csv'}")
    print(f"[done] wrote {args.out_dir / 'algo_aggregate.csv'}")

    # Learning curves with per-seed thin lines + per-algo mean bold line.
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_curves), 3)))
    for color, (algo_name, curves) in zip(colors, all_curves.items()):
        if not curves:
            continue
        for c in curves:
            ax.plot(c["step"], c["reward"], color=color, alpha=0.25, linewidth=0.8)
        merged = pd.concat(curves).groupby("step")["reward"].mean().reset_index()
        ax.plot(merged["step"], merged["reward"], color=color, linewidth=2.0, label=algo_name)
    ax.set_xlabel("env step")
    ax.set_ylabel("Cumulative reward (mean over summary window)")
    ax.set_title("Learning curves — per-seed (thin) + per-algo mean (bold)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "learning_curves.png", dpi=150)
    print(f"[done] wrote {args.out_dir / 'learning_curves.png'}")


if __name__ == "__main__":
    main()
