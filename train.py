"""
Train a single RL algorithm on the shared PacManEnv benchmark.

Usage:
    python train.py --algo qlearning    --episodes 6000
    python train.py --algo sarsa        --episodes 6000
    python train.py --algo montecarlo   --episodes 6000
    python train.py --algo dqn          --episodes 1500
    python train.py --algo double_dqn   --episodes 1500
    python train.py --algo reinforce    --episodes 1500

Every episode's outcome is appended to results/<algo>_metrics.csv and the
final model is saved to models/<algo>.pkl (tabular) or models/<algo>.pt
(neural). Dynamic difficulty automatically ramps up as the agent's rolling
average score improves (see utils/curriculum.py).
"""
import argparse
import os
import time

from env.pacman_env import PacManEnv
from utils.metrics_logger import MetricsLogger
from utils.curriculum import DifficultyCurriculum

TABULAR_ALGOS = {"qlearning", "sarsa", "montecarlo"}
NEURAL_ALGOS = {"dqn", "double_dqn", "reinforce"}


def build_agent(algo, input_dim, device=None, batch_size=None):
    if algo == "qlearning":
        from agents.q_learning import QLearningAgent
        return QLearningAgent()
    if algo == "sarsa":
        from agents.sarsa import SARSAAgent
        return SARSAAgent()
    if algo == "montecarlo":
        from agents.monte_carlo import MonteCarloAgent
        return MonteCarloAgent()
    if algo == "dqn":
        from agents.dqn import DQNAgent
        kwargs = {"device": device}
        if batch_size:
            kwargs["batch_size"] = batch_size
        return DQNAgent(input_dim=input_dim, **kwargs)
    if algo == "double_dqn":
        from agents.double_dqn import DoubleDQNAgent
        kwargs = {"device": device}
        if batch_size:
            kwargs["batch_size"] = batch_size
        return DoubleDQNAgent(input_dim=input_dim, **kwargs)
    if algo == "reinforce":
        from agents.reinforce import ReinforceAgent
        return ReinforceAgent(input_dim=input_dim, device=device)
    raise ValueError(f"unknown algo: {algo}")


def model_path(algo, models_dir="models"):
    ext = "pkl" if algo in TABULAR_ALGOS else "pt"
    return os.path.join(models_dir, f"{algo}.{ext}")


def model_meta_path(algo, models_dir="models"):
    return os.path.join(models_dir, f"{algo}_meta.json")


def _save_meta(algo, final_difficulty, models_dir="models"):
    """Record the difficulty tier training actually ended on, so
    visualize_play.py can default to showing the agent against the ghost
    roster it was actually trained against instead of always defaulting to
    difficulty 0 (which is why you may have only ever seen a single ghost
    in playback -- difficulty 0 has exactly one)."""
    import json
    with open(model_meta_path(algo, models_dir), "w") as f:
        json.dump({"algo": algo, "final_difficulty": final_difficulty}, f)


def load_meta(algo, models_dir="models"):
    import json
    path = model_meta_path(algo, models_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _autotune_epsilon_decay(agent, episodes, decay_fraction=0.5):
    """Recompute epsilon_decay so epsilon actually reaches epsilon_min by
    `decay_fraction` of the way through training, regardless of how many
    episodes were requested. A fixed decay constant (e.g. 0.9995) tuned for
    one episode count silently fails to converge for any other episode
    count -- with 6000 episodes and decay=0.9995, epsilon was still ~0.22
    at the halfway point and only reached ~0.05 in the last few episodes,
    meaning the agent spent almost the whole run acting near-randomly. This
    guarantees a real exploitation phase near the end of training instead."""
    if not hasattr(agent, "epsilon_decay"):
        return
    target_episode = max(1, int(episodes * decay_fraction))
    # epsilon_min = epsilon_start * decay^target_episode  =>  solve for decay
    ratio = agent.epsilon_min / max(agent.epsilon, 1e-8)
    agent.epsilon_decay = ratio ** (1.0 / target_episode)


def train(algo, episodes, max_steps=400, curriculum=True, log_every=100,
          results_dir="results", models_dir="models", verbose=True,
          device=None, batch_size=None):
    os.makedirs(models_dir, exist_ok=True)
    env = PacManEnv(difficulty=0, max_steps=max_steps)
    agent = build_agent(algo, input_dim=env.grid_state_dim, device=device, batch_size=batch_size)
    _autotune_epsilon_decay(agent, episodes)
    logger = MetricsLogger(algo, out_dir=results_dir)
    curr = DifficultyCurriculum() if curriculum else None

    use_grid = algo in NEURAL_ALGOS
    start = time.time()

    for ep in range(1, episodes + 1):
        env.reset()
        s = env.get_grid_state() if use_grid else env.get_tabular_state()
        done = False
        total_reward = 0

        # SARSA needs the *next* action chosen before it can bootstrap.
        a = agent.select_action(s) if algo == "sarsa" else None

        while not done:
            if algo == "sarsa":
                _, r, term, trunc, info = env.step(a)
                done = term or trunc
                s_next = env.get_tabular_state()
                a_next = agent.select_action(s_next) if not done else 0
                agent.update(s, a, r, s_next, a_next, done)
                s, a = s_next, a_next

            elif algo in ("qlearning",):
                act = agent.select_action(s)
                _, r, term, trunc, info = env.step(act)
                done = term or trunc
                s_next = env.get_tabular_state()
                agent.update(s, act, r, s_next, done)
                s = s_next

            elif algo == "montecarlo":
                act = agent.select_action(s)
                _, r, term, trunc, info = env.step(act)
                done = term or trunc
                agent.record(s, act, r)
                s = env.get_tabular_state()

            elif algo in ("dqn", "double_dqn"):
                act = agent.select_action(s)
                _, r, term, trunc, info = env.step(act)
                done = term or trunc
                s_next = env.get_grid_state()
                agent.remember(s, act, r, s_next, float(term))
                agent.train_step()
                s = s_next

            elif algo == "reinforce":
                act = agent.select_action(s)
                _, r, term, trunc, info = env.step(act)
                done = term or trunc
                agent.record_reward(r)
                s = env.get_grid_state()

            total_reward += r

        agent.on_episode_end()
        if algo != "reinforce":
            agent.decay_epsilon()

        logger.log(ep, algo, env.difficulty, total_reward, info["score"],
                    info["steps"], info["result"], getattr(agent, "epsilon", None))

        if curr is not None:
            leveled_up = curr.update(env, info["result"])
            if leveled_up and hasattr(agent, "epsilon"):
                # the ghost roster just got harder -- give the agent room to
                # re-explore instead of continuing to exploit a policy that
                # was only tuned for the easier setup it just graduated from.
                agent.epsilon = max(agent.epsilon, 0.3)

        if verbose and (ep % log_every == 0 or ep == episodes):
            elapsed = time.time() - start
            if algo == "reinforce":
                # REINFORCE has no epsilon -- show policy entropy instead
                # (higher = more exploratory/spread-out action distribution,
                # falling entropy over training is the useful signal here).
                ent = agent.last_mean_entropy
                explore_str = f"entropy={ent:.3f}" if ent is not None else "entropy=n/a "
            else:
                explore_str = f"eps={getattr(agent, 'epsilon', 0):.3f}"
            print(f"[{algo}] ep {ep}/{episodes} | score={info['score']:>5} "
                  f"reward={total_reward:>7.1f} steps={info['steps']:>3} "
                  f"result={info['result']:<8} diff={env.difficulty} "
                  f"{explore_str} elapsed={elapsed:.1f}s")

    agent.save(model_path(algo, models_dir))
    _save_meta(algo, env.difficulty, models_dir)
    logger.close()
    print(f"[{algo}] training complete. Model -> {model_path(algo, models_dir)} | "
          f"Metrics -> results/{algo}_metrics.csv | Final difficulty reached: {env.difficulty}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True,
                         choices=sorted(TABULAR_ALGOS | NEURAL_ALGOS))
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--no_curriculum", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"],
                         help="Only affects dqn/double_dqn/reinforce (neural agents). "
                              "Default: auto-detect CUDA if available. Tabular agents "
                              "(qlearning/sarsa/montecarlo) always run on CPU.")
    parser.add_argument("--batch_size", type=int, default=None,
                         help="Replay batch size for dqn/double_dqn. Default 128; "
                              "try 256 on an 8GB+ GPU for more throughput.")
    args = parser.parse_args()

    train(args.algo, args.episodes, max_steps=args.max_steps,
          curriculum=not args.no_curriculum, log_every=args.log_every,
          device=args.device, batch_size=args.batch_size)
