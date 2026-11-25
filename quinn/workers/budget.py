"""Transactional, shared GPU-time admission across independent capture jobs."""
from contextlib import closing
import math
from pathlib import Path
import sqlite3


class Ledger:
    def __init__(self, path, ceiling_s):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not math.isfinite(ceiling_s) or ceiling_s <= 0:
            raise ValueError("GPU budget must be finite and positive")
        self.ceiling_s = ceiling_s
        with closing(self.connect()) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS head (id INTEGER PRIMARY KEY CHECK (id=1), ceiling REAL)")
            connection.execute("INSERT OR IGNORE INTO head VALUES (1, ?)", (ceiling_s,))
            stored = connection.execute("SELECT ceiling FROM head").fetchone()[0]
            if stored != ceiling_s:
                raise ValueError(f"Ledger ceiling is already {stored / 3600:g} GPU-hours")
            connection.execute("""CREATE TABLE IF NOT EXISTS attempt
                (id INTEGER PRIMARY KEY, job TEXT, phase TEXT, status TEXT,
                 reserved REAL, seconds REAL, detail TEXT)""")
            connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS completed
                ON attempt(job) WHERE status IN ('running', 'done')""")

    def connect(self):
        return sqlite3.connect(self.path, timeout=15, isolation_level="IMMEDIATE")

    def recover(self, job, complete_attempts=()):
        """Call only with the physical device and child leases held."""
        with closing(self.connect()) as connection, connection:
            row = connection.execute("SELECT id FROM attempt WHERE job=? AND status='running'",
                                     (job,)).fetchone()
            if row is None:
                return
            if row[0] in complete_attempts:
                status, detail = "done", "adopted complete raw capture; charged full reservation"
            else:
                status, detail = "interrupted", "unreconciled attempt charged full reservation"
            connection.execute("UPDATE attempt SET status=?, detail=? WHERE id=?",
                               (status, detail, row[0]))

    def completed_attempt(self, job):
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT id FROM attempt WHERE job=? AND status='done'",
                                     (job,)).fetchone()
        return None if row is None else row[0]

    def reserve(self, job, phase, seconds, resume=False):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("GPU reservation must be finite and positive")
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            complete = connection.execute(
                "SELECT 1 FROM attempt WHERE job=? AND status='done'", (job,)
            ).fetchone()
            if complete:
                if not resume:
                    raise ValueError("Completed capture already exists. Use --resume.")
                return None
            used = connection.execute("SELECT coalesce(sum(seconds), 0) FROM attempt").fetchone()[0]
            if used + seconds > self.ceiling_s:
                raise ValueError(f"GPU budget exhausted: charged {used:.1f}s, reservation {seconds:.1f}s")
            cursor = connection.execute("""INSERT INTO attempt(job, phase, status, reserved, seconds)
                VALUES (?, ?, 'running', ?, ?)""", (job, phase, seconds, seconds))
            return cursor.lastrowid

    def finish(self, attempt, seconds, succeeded, detail):
        with closing(self.connect()) as connection, connection:
            row = connection.execute("SELECT reserved FROM attempt WHERE id=? AND status='running'",
                                     (attempt,)).fetchone()
            if row is None:
                raise RuntimeError("attempt has no live reservation")
            exceeded = seconds > row[0]
            status = "overrun" if exceeded else ("done" if succeeded else "failed")
            connection.execute("UPDATE attempt SET status=?, seconds=?, detail=? WHERE id=?",
                               (status, seconds, detail, attempt))
        if exceeded:
            raise RuntimeError("Attempt exceeded its reservation; capture halted")

    def summary(self):
        with closing(self.connect()) as connection:
            phases = dict(connection.execute("SELECT phase, sum(seconds) FROM attempt GROUP BY phase"))
        return {"budget_gpu_hours": self.ceiling_s / 3600,
                "charged_gpu_hours": sum(phases.values()) / 3600,
                "phases_gpu_hours": {k: phases.get(k, 0) / 3600 for k in
                                     ("model_rollout", "policy_optimization", "reward_evaluation")}}
