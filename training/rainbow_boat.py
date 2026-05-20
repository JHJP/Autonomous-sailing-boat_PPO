"""Rainbow DQN training for MoveToGoal sailboat env — MULTI-AGENT version.

Tianshou 2.0.1 + raw mlagents-envs API (no UnityToGymWrapper).
Buyoancy.unity Standalone scene has 16 parallel Environment instances sharing one Behavior.
We treat each agent as one slot in a PrioritizedVectorReplayBuffer(buffer_num=16) and
drive all 16 from a single shared C51/Rainbow policy.

Discrete action space: Unity exposes (3, 3) joint discrete. We flatten to Discrete(9):
    action_int = steer_idx * 3 + motor_idx,
    where steer_idx, motor_idx ∈ {0, 1, 2}.

Vector observation: 11-d (Environment.prefab VectorObservationSize=11). v1 code emits
12 floats but Behavior Parameters truncates the trailing component (Quaternion.w).

Usage:
    conda activate tianshou
    python code/training/rainbow_boat.py --seed 0 --run-id rainbow_seed_0 --total-steps 5000000

TB logs use ml-agents-compatible tags:
    Environment/Cumulative Reward   (per-episode return, logged on terminal)
    Environment/Episode Length      (per-episode steps, logged on terminal)
    Losses/Loss                     (rainbow loss per update)
so the analysis/parse_results.py script can aggregate across PPO/SAC/Rainbow uniformly.
"""
from __future__ import annotations

import argparse
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from tianshou.algorithm import RainbowDQN
from tianshou.algorithm.algorithm_base import policy_within_training_step
from tianshou.algorithm.modelfree.c51 import C51Policy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch, PrioritizedVectorReplayBuffer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import NoisyLinear


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = PROJECT_ROOT / "code" / "Builds" / "BoatSailing_Mac"  # no .app suffix; mlagents auto-appends


def build_rainbow_net(state_shape, action_shape, num_atoms: int, noisy_std: float, hidden_sizes):
    def noisy_linear(x: int, y: int) -> NoisyLinear:
        return NoisyLinear(x, y, noisy_std=noisy_std)

    return Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=hidden_sizes,
        softmax=True,
        num_atoms=num_atoms,
        dueling_param=(
            {"linear_layer": noisy_linear},
            {"linear_layer": noisy_linear},
        ),
    )


def make_env(binary_path: Path, worker_id: int, seed: int, unity_log_file: str = "/tmp/unity_rainbow.log"):
    # Redirect Unity's own log to a file. Without this Unity spams stdout with
    # warnings ("More observations (12) made than vector observation size (11)..."
    # from the v1 emit/spec mismatch — see project_revision_facts.md). The stdout
    # spam throttles Python's main loop to <5% CPU.
    return UnityEnvironment(
        file_name=str(binary_path),
        worker_id=worker_id,
        seed=seed,
        no_graphics=True,
        additional_args=["-logFile", unity_log_file],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", type=str, default="rainbow_seed_0")
    p.add_argument("--total-steps", type=int, default=5_000_000)
    p.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    p.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "code" / "results")
    p.add_argument("--device", type=str, default="cpu")
    # Rainbow hyperparameters
    p.add_argument("--num-atoms", type=int, default=51)
    p.add_argument("--v-min", type=float, default=-1.0)
    p.add_argument("--v-max", type=float, default=1.0)
    p.add_argument("--noisy-std", type=float, default=0.1)
    p.add_argument("--n-step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--target-update-freq", type=int, default=500)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--per-alpha", type=float, default=0.5)
    p.add_argument("--per-beta", type=float, default=0.4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    p.add_argument("--warmup-steps", type=int, default=5_000, help="random-policy buffer warmup")
    p.add_argument("--update-interval", type=int, default=10, help="gradient update every N env steps (per agent)")
    p.add_argument("--summary-freq", type=int, default=2_000)
    args = p.parse_args()

    if not args.binary.with_suffix(".app").exists():
        raise FileNotFoundError(f"Standalone not found: {args.binary}.app — build it first.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Unity env ---
    print(f"[init] launching Unity binary worker_id={args.seed}")
    env = make_env(args.binary, worker_id=args.seed, seed=args.seed)
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    obs_dim = spec.observation_specs[0].shape[0]
    # v1 env: 2 discrete branches × 3 actions each → flatten to Discrete(9)
    n_branches = len(spec.action_spec.discrete_branches)
    branch_sizes = spec.action_spec.discrete_branches
    assert n_branches == 2 and tuple(branch_sizes) == (3, 3), \
        f"expected (3,3) discrete branches, got {branch_sizes}"
    n_actions = int(np.prod(branch_sizes))  # 9
    decision_steps, _ = env.get_steps(behavior_name)
    n_agents = len(decision_steps)
    print(f"[init] behavior={behavior_name} obs_dim={obs_dim} n_actions={n_actions} n_agents={n_agents}")

    # --- Policy + algorithm ---
    action_space = gym.spaces.Discrete(n_actions)
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    net = build_rainbow_net(
        state_shape=(obs_dim,),
        action_shape=n_actions,
        num_atoms=args.num_atoms,
        noisy_std=args.noisy_std,
        hidden_sizes=tuple(args.hidden_sizes),
    )
    policy = C51Policy(
        model=net,
        action_space=action_space,
        observation_space=observation_space,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        eps_training=0.0,
        eps_inference=0.0,
    )
    optim_factory = AdamOptimizerFactory(lr=args.lr)
    algorithm = RainbowDQN(
        policy=policy,
        optim=optim_factory,
        gamma=args.gamma,
        n_step_return_horizon=args.n_step,
        target_update_freq=args.target_update_freq,
    ).to(args.device)

    # --- Replay buffer (one sub-buffer per agent for clean rollout boundaries) ---
    buffer = PrioritizedVectorReplayBuffer(
        total_size=args.buffer_size,
        buffer_num=n_agents,
        alpha=args.per_alpha,
        beta=args.per_beta,
    )

    # --- Logging ---
    log_dir = args.log_dir / args.run_id / behavior_name.split("?")[0]
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"[init] logging to {log_dir}")

    # --- Per-agent rolling state ---
    # agent_id → buffer slot index (stable mapping for this Unity env)
    agent_id_to_slot: dict[int, int] = {}
    prev_obs: dict[int, np.ndarray] = {}
    prev_act: dict[int, int] = {}
    ep_reward: dict[int, float] = {}
    ep_length: dict[int, int] = {}

    def slot_for(agent_id: int) -> int:
        if agent_id not in agent_id_to_slot:
            agent_id_to_slot[agent_id] = len(agent_id_to_slot)
        return agent_id_to_slot[agent_id]

    def make_batch(obs, act, rew, terminated, truncated, obs_next, info=None) -> Batch:
        return Batch(
            obs=np.asarray(obs, dtype=np.float32),
            act=np.asarray(act, dtype=np.int64),
            rew=np.asarray(rew, dtype=np.float32),
            terminated=np.asarray(terminated, dtype=bool),
            truncated=np.asarray(truncated, dtype=bool),
            done=np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool),
            obs_next=np.asarray(obs_next, dtype=np.float32),
            info=info if info is not None else {},
        )

    # --- Training loop ---
    total_env_steps = 0
    n_episodes = 0
    n_updates = 0
    last_summary_step = 0
    recent_returns: deque[float] = deque(maxlen=50)
    recent_lengths: deque[int] = deque(maxlen=50)

    print(f"[train] target={args.total_steps} env steps; warmup={args.warmup_steps} (random policy)")
    while total_env_steps < args.total_steps:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # --- Terminal (episode ends) ---
        for idx, agent_id in enumerate(terminal_steps.agent_id):
            agent_id = int(agent_id)
            term_obs = terminal_steps.obs[0][idx]
            term_rew = float(terminal_steps.reward[idx])
            interrupted = bool(terminal_steps.interrupted[idx])  # True if max_step truncated
            if agent_id in prev_obs:
                slot = slot_for(agent_id)
                buffer.add(
                    make_batch(
                        obs=[prev_obs[agent_id]],
                        act=[prev_act[agent_id]],
                        rew=[term_rew],
                        terminated=[not interrupted],
                        truncated=[interrupted],
                        obs_next=[term_obs],
                    ),
                    buffer_ids=np.array([slot], dtype=np.int64),
                )
                ep_reward[agent_id] = ep_reward.get(agent_id, 0.0) + term_rew
                ep_length[agent_id] = ep_length.get(agent_id, 0) + 1
                n_episodes += 1
                recent_returns.append(ep_reward[agent_id])
                recent_lengths.append(ep_length[agent_id])
                del prev_obs[agent_id], prev_act[agent_id]
                ep_reward.pop(agent_id, None)
                ep_length.pop(agent_id, None)

        # --- Decision (active agents need actions) ---
        if len(decision_steps) > 0:
            agent_ids = [int(a) for a in decision_steps.agent_id]
            obs_batch = np.asarray(decision_steps.obs[0], dtype=np.float32)  # (n, obs_dim)
            rewards = np.asarray(decision_steps.reward, dtype=np.float32)    # (n,)

            # Add interim (non-terminal) transitions for agents that already had a decision
            interim_obs, interim_act, interim_rew, interim_next, interim_slots = [], [], [], [], []
            for i, aid in enumerate(agent_ids):
                if aid in prev_obs:
                    interim_obs.append(prev_obs[aid])
                    interim_act.append(prev_act[aid])
                    interim_rew.append(rewards[i])
                    interim_next.append(obs_batch[i])
                    interim_slots.append(slot_for(aid))
                    ep_reward[aid] = ep_reward.get(aid, 0.0) + float(rewards[i])
                    ep_length[aid] = ep_length.get(aid, 0) + 1
            if interim_obs:
                buffer.add(
                    make_batch(
                        obs=interim_obs,
                        act=interim_act,
                        rew=interim_rew,
                        terminated=[False] * len(interim_obs),
                        truncated=[False] * len(interim_obs),
                        obs_next=interim_next,
                    ),
                    buffer_ids=np.array(interim_slots, dtype=np.int64),
                )

            # Choose actions: random during warmup, Rainbow policy after.
            # policy.train() before forward — NoisyLinear only applies parameter noise in
            # training mode (NoisyLinear.forward checks self.training). algorithm.update() may
            # leave policy in eval mode; without train() here, NoisyNet exploration silently dies.
            # Use policy(Batch(...)) instead of compute_action — compute_action flattens batched obs
            # (16, 11) into (1, 176) which mismatches the 11x128 first Linear layer.
            if total_env_steps < args.warmup_steps:
                actions_int = np.random.randint(0, n_actions, size=(len(agent_ids),))
            else:
                policy.train()
                obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=args.device)
                with torch.no_grad():
                    out = policy(Batch(obs=obs_t, info={}))
                act = out.act
                actions_int = (
                    act.cpu().numpy().astype(np.int64)
                    if hasattr(act, "cpu")
                    else np.asarray(act, dtype=np.int64)
                )
            actions_int = np.asarray(actions_int, dtype=np.int64).reshape(-1)

            # Save rolling state
            for i, aid in enumerate(agent_ids):
                prev_obs[aid] = obs_batch[i].copy()
                prev_act[aid] = int(actions_int[i])

            # Un-flatten Discrete(9) → (steer, motor) ∈ (3, 3)
            joint = np.stack(
                [actions_int // 3, actions_int % 3],
                axis=1,
            ).astype(np.int32)
            action_tuple = ActionTuple(discrete=joint)
            env.set_actions(behavior_name, action_tuple)

            total_env_steps += len(agent_ids)

        env.step()

        # --- Gradient update (after warmup) ---
        if (
            total_env_steps >= args.warmup_steps
            and len(buffer) >= args.batch_size
            and total_env_steps // args.update_interval > n_updates
        ):
            try:
                # Tianshou 2.x requires policy.is_within_training_step=True around .update()
                # when bypassing the built-in trainer.
                with policy_within_training_step(policy):
                    stats = algorithm.update(buffer=buffer, sample_size=args.batch_size)
                n_updates += 1
                # TrainingStats exposes losses via get_loss_stats_dict() → {name: value}.
                # Earlier code looked for .loss attribute and silently logged nothing.
                if n_updates % 20 == 0:
                    try:
                        loss_dict = stats.get_loss_stats_dict()
                        for k, v in loss_dict.items():
                            if isinstance(v, (int, float, np.floating)):
                                writer.add_scalar(f"Losses/{k}", float(v), total_env_steps)
                    except Exception:
                        pass
            except Exception as e:
                if n_updates < 3:
                    print(f"[update-error {n_updates}] {type(e).__name__}: {e}")
                n_updates += 1  # don't get stuck retrying every step

        # --- Periodic summary log + console ---
        if total_env_steps - last_summary_step >= args.summary_freq:
            last_summary_step = total_env_steps
            if recent_returns:
                writer.add_scalar("Environment/Cumulative Reward", float(np.mean(recent_returns)), total_env_steps)
                writer.add_scalar("Environment/Episode Length", float(np.mean(recent_lengths)), total_env_steps)
                writer.add_scalar("train/episodes", n_episodes, total_env_steps)
                writer.add_scalar("train/updates", n_updates, total_env_steps)
                print(
                    f"[step {total_env_steps:>8}] "
                    f"ep={n_episodes} upd={n_updates} "
                    f"reward(mean last50)={np.mean(recent_returns):+.3f} "
                    f"ep_len(mean last50)={np.mean(recent_lengths):.1f}"
                )

    # --- Save final ---
    save_path = log_dir.parent / "final_policy.pth"
    torch.save({"policy": policy.state_dict(), "env_step": total_env_steps}, save_path)
    print(f"[done] saved {save_path}")
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
