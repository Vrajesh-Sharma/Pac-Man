"""
Train every algorithm back-to-back with sensible default episode counts
(tabular methods are cheap and get more episodes; neural methods are more
expensive per-episode so get fewer). Adjust EPISODES below to fit your time
budget -- for a first run/demo, halving these is fine. If you have a CUDA
GPU, the neural agents (dqn/double_dqn/reinforce) will run noticeably
faster per episode -- consider bumping their episode counts up (e.g. 2500)
to take advantage of the extra headroom.

Usage:
    python train_all.py
    python train_all.py --quick                # fast smoke run to check everything works
    python train_all.py --device cuda           # force GPU for neural agents
    python train_all.py --device cuda --batch_size 256   # bigger batches, more GPU throughput
"""
import argparse

from train import train

EPISODES = {
    "qlearning": 6000,
    "sarsa": 6000,
    "montecarlo": 6000,
    "dqn": 1500,
    "double_dqn": 1500,
    "reinforce": 1500,
}

QUICK_EPISODES = {k: min(200, v) for k, v in EPISODES.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Run a fast smoke test (few episodes each) instead of full training.")
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"],
                         help="Only affects dqn/double_dqn/reinforce. Default: auto-detect CUDA.")
    parser.add_argument("--batch_size", type=int, default=None,
                         help="Replay batch size for dqn/double_dqn (default 128).")
    args = parser.parse_args()

    schedule = QUICK_EPISODES if args.quick else EPISODES

    for algo, n_episodes in schedule.items():
        print(f"\n{'='*70}\nTraining {algo} for {n_episodes} episodes\n{'='*70}")
        train(algo, n_episodes, log_every=max(1, n_episodes // 20),
              device=args.device, batch_size=args.batch_size)

    print("\nAll algorithms trained. Run `python dashboard.py` to compare them.")
