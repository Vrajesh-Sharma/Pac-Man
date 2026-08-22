import csv
import os


class MetricsLogger:
    """Appends one row per episode to results/<algo>_metrics.csv so
    dashboard.py can later compare every algorithm on equal footing."""

    FIELDS = ["episode", "algo", "difficulty", "total_reward", "score",
              "steps_survived", "result", "epsilon"]

    def __init__(self, algo_name, out_dir="results"):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{algo_name}_metrics.csv")
        self._f = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._writer.writeheader()

    def log(self, episode, algo, difficulty, total_reward, score,
             steps_survived, result, epsilon=None):
        self._writer.writerow({
            "episode": episode, "algo": algo, "difficulty": difficulty,
            "total_reward": total_reward, "score": score,
            "steps_survived": steps_survived, "result": result,
            "epsilon": epsilon if epsilon is not None else "",
        })
        self._f.flush()

    def close(self):
        self._f.close()
