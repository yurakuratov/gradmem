"""Batch experiment scheduler for the test-time-gd / gradmem repo.

This schedules many ``run_from_config.py`` invocations (the repo's standard
entry point) from a single YAML *manifest*, with checkpointing, optional
parallelism, auto-retry, and resume-after-interrupt.

Manifest schema
---------------
The manifest is a YAML file with these top-level keys (only ``experiments``
is required):

.. code-block:: yaml

    max_parallel: 1            # global cap on concurrent runs (default 1)
    schedule: sequential       # 'sequential' or 'interleaved' (default sequential)
    spawn_delay: 0.0           # seconds between consecutive launches (default 0)
    max_retries: 0             # auto-retry budget per run (default 0 = off)
    experiments:
      - name: my_grid          # required, used in checkpoint keys / logs
        command: "python run_from_config.py"   # required, the launch binary
        base_config: "configs/gradmemgpt/kv_retrieval/default.yaml"  # required
        max_parallel: 1        # per-experiment cap (default = global max_parallel)
        overrides:             # applied to every run in this experiment
          gradmem.inner_lr: 0.02
          training.seed: 143
        grid:                  # cartesian product -> one run per combination
          gradmem.K: [1, 2, 4]
          training.warmup_steps: [1000, 10000]

How commands are built
----------------------
For each grid combination the scheduler emits::

    <command> <base_config> <key>=<val> <key>=<val> ...

The ``key=val`` tokens are **positional** overrides consumed by
``run_from_config.py``'s ``overrides`` nargs collector (which ``eval()``s the
value). Dotted keys (``section.subkey``) route into a config section; bare keys
are matched against any section. Booleans/numbers/strings are formatted so they
round-trip through that ``eval()``. There is intentionally **no** ``--`` prefix
— that is what distinguishes an override from a flag like ``--debug`` or
``--config`` here. To pass a flag through verbatim, list it in ``command``
(e.g. ``command: "python run_from_config.py --debug"``).

See the bottom of ``--help`` output and ``manifests/`` for examples.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


# ==================== Signal-aware cleanup ====================

class SchedulerInterrupted(Exception):
    """Raised to unwind run loops after Ctrl-C / SIGTERM handling."""


def _sigterm_handler(signum, frame):
    """Translate SIGTERM into KeyboardInterrupt so the same code path handles it."""
    raise KeyboardInterrupt


def _pid_alive(pid: int) -> bool:
    """Return True if `pid` exists. Used to monitor re-attached children."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by another user
    return True


def _killpg_quietly(pid: int, sig: int) -> None:
    """Send a signal to the process group of `pid`; ignore already-dead errors."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def _wait_for_spawn_slot(last_spawn_time: float, spawn_delay: float) -> None:
    """Sleep so at least `spawn_delay` seconds separate consecutive spawns.

    No-op on the first spawn (`last_spawn_time == 0`) or when `spawn_delay`
    is non-positive. Used to stagger launches when many processes starting at
    once contend for the GPU or trip accelerator init races.
    """
    if spawn_delay > 0 and last_spawn_time > 0:
        remaining = spawn_delay - (time.time() - last_spawn_time)
        if remaining > 0:
            time.sleep(remaining)


def _should_respawn(ret: int | str | None, attempts: int, max_retries: int) -> bool:
    """True if a finished run should be re-queued for another attempt.

    `ret` is the poll() result: 0 = success (never respawn), a non-zero int =
    failure, or 'unknown' (resumed child disappeared). `attempts` is the number
    of retries already performed for this run (0 = it was the first attempt).
    Respawn is allowed while `attempts < max_retries`.
    """
    if max_retries <= 0:
        return False
    if ret == 0:
        return False
    if attempts >= max_retries:
        return False
    # ret is a non-zero int (fail) or 'unknown' — both are retryable.
    return True


def _prompt_terminate() -> str:
    """Ask whether to terminate running children on interrupt.

    Returns 'terminate' or 'leave'. On non-interactive input (EOF), defaults
    to 'terminate' so a Ctrl-C from a non-TTY doesn't strand children.
    """
    try:
        answer = input(
            "\nScheduler interrupted. Terminate running child processes? [Y/n] "
        )
    except EOFError:
        return "terminate"
    return "leave" if answer.strip().lower() in ("n", "no") else "terminate"


class _Child:
    """A tracked run.

    Encapsulates the dual nature of a tracked run: a `subprocess.Popen` we
    launched this session (`proc is not None`), or a PID left running by a
    previous session that we're merely polling (`proc is None`).

    Both run-loop functions track children as `dict[pid -> _Child]`.
    """

    def __init__(
        self,
        name: str,
        idx: int,
        start_iso: str,
        start: float,
        exp_i: int | None,
        proc: subprocess.Popen | None,
        pid: int,
        attempts: int = 0,
    ):
        self.name = name
        self.idx = idx
        self.start_iso = start_iso
        self.start = start
        self.exp_i = exp_i
        self.proc = proc
        self.pid = pid
        # How many retries preceded this launch (0 = first attempt). Used for
        # the auto-respawn budget and recorded in the checkpoint on finalize.
        self.attempts = attempts

    def poll(self) -> int | None | str:
        """None if still running; an int exit code if we own the proc and it
        finished; or the literal 'unknown' if a resumed child has disappeared
        (we cannot reap its exit code since we are not its parent)."""
        if self.proc is not None:
            return self.proc.poll()
        return None if _pid_alive(self.pid) else "unknown"

    def finalize(
        self,
        checkpoint: dict,
        command: list[str],
        exit_code: int | None,
        status: str,
    ) -> None:
        """Record a terminal state in the checkpoint (ok/fail/unknown)."""
        end_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        entry: dict = {
            "command": command,
            "status": status,
            "start_time": self.start_iso,
            "end_time": end_iso,
            "duration_s": round(time.time() - self.start, 1),
        }
        if exit_code is not None:
            entry["exit_code"] = exit_code
        if self.attempts > 0:
            entry["attempts"] = self.attempts
        key = f"{self.name}/{self.idx}"
        checkpoint["runs"][key] = entry

    def leave(self, checkpoint: dict, command: list[str]) -> None:
        """On 'leave', persist the pid so a restarted scheduler can resume
        polling; keep status 'running'."""
        checkpoint["runs"][f"{self.name}/{self.idx}"] = {
            "command": command,
            "status": "running",
            "pid": self.pid,
            "start_time": self.start_iso,
        }

    def terminate(self) -> int | None:
        """SIGTERM the whole process group, wait up to ~5s, then SIGKILL.

        Returns the child's exit code (negative if killed by a signal), or
        None for a resumed child we don't own (we can signal its group but
        cannot `wait()` to reap the result). Children are launched with
        `start_new_session=True`, so the group leader's pid equals the group
        id; this also takes down grandchildren (e.g. the `accelerate launch`
        worker processes each run spawns).
        """
        if self.proc is None:
            # Resumed child: signal its group but don't wait (not our child).
            _killpg_quietly(self.pid, signal.SIGTERM)
            return None
        _killpg_quietly(self.pid, signal.SIGTERM)
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _killpg_quietly(self.pid, signal.SIGKILL)
            try:
                return self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return -signal.SIGKILL


def _handle_interrupt(
    running: dict[int, _Child],
    checkpoint: dict,
    cp_path: Path,
    on_interrupt: str,
) -> None:
    """Shared Ctrl-C handler.

    Asks the user (unless `on_interrupt` pins a choice), then either kills
    every running child (recording ok/fail) or leaves them running (recording
    their pids). Always raises SchedulerInterrupted so run loops unwind.
    """
    action = on_interrupt
    if action == "prompt":
        action = _prompt_terminate()

    if action == "terminate":
        for child in running.values():
            command = checkpoint["runs"][f"{child.name}/{child.idx}"]["command"]
            code = child.terminate()
            if code is None:
                # Resumed child we don't own: signaled but exit code unrecoverable.
                child.finalize(checkpoint, command, None, "unknown")
                label = "UNKNOWN"
            else:
                status = "ok" if code == 0 else "fail"
                child.finalize(checkpoint, command, code, status)
                label = "OK" if code == 0 else f"FAIL(code={code})"
            print(
                f"  [{child.name} {child.idx + 1}] terminated -> {label} "
                f"(PID {child.pid})"
            )
    else:  # leave
        for child in running.values():
            command = checkpoint["runs"][f"{child.name}/{child.idx}"]["command"]
            child.leave(checkpoint, command)
            print(f"  [{child.name} {child.idx + 1}] left running (PID {child.pid})")

    save_checkpoint(checkpoint, cp_path)
    raise SchedulerInterrupted()


# ==================== Manifest ====================

def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        manifest = yaml.safe_load(f)

    if "experiments" not in manifest:
        raise ValueError("Manifest must contain 'experiments' key")

    for i, exp in enumerate(manifest["experiments"]):
        for required in ("name", "command", "base_config"):
            if required not in exp:
                raise ValueError(
                    f"Experiment {i} missing required field '{required}'"
                )

    return manifest


# ==================== Grid expansion & override formatting ====================

def expand_grid(grid: dict | None) -> list[dict]:
    if not grid:
        return [{}]

    keys = list(grid.keys())
    values = []
    for k in keys:
        v = grid[k]
        values.append(v if isinstance(v, list) else [v])

    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def format_value(value) -> str:
    """Render a YAML override value as ``run_from_config.py`` expects.

    ``run_from_config.py`` collects positional ``key=value`` tokens and runs
    ``eval()`` on the value half, so the rendering must be a valid Python
    literal. Numbers and bools use their ``repr()``; strings are quoted so
    ``eval`` reproduces a ``str`` (not interpreted as an identifier); ``None``
    becomes the literal ``None``.
    """
    if isinstance(value, bool):
        return repr(value)            # True / False
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return "None"
    if isinstance(value, str):
        return repr(value)            # quoted so eval() -> str
    # Lists/dicts: json round-trips through eval as the matching Python literal.
    return json.dumps(value)


def format_override(key: str, value) -> str:
    """A positional ``key=value`` override for ``run_from_config.py``.

    No ``--`` prefix: ``run_from_config.py`` reads overrides from its nargs
    ``overrides`` collector, not from argparse flags. Dotted keys
    (``section.subkey``) route into a config section inside the launched run.
    """
    return f"{key}={format_value(value)}"


def build_commands(experiment: dict) -> list[list[str]]:
    base_cmd = experiment["command"].split()
    base_config = experiment["base_config"]

    overrides = experiment.get("overrides") or {}
    grid_combos = expand_grid(experiment.get("grid"))

    commands = []
    for combo in grid_combos:
        merged = {**overrides, **combo}
        parts = base_cmd + [base_config]
        for k, v in merged.items():
            parts.append(format_override(k, v))
        commands.append(parts)

    return commands


# ==================== Checkpointing ====================

def checkpoint_path(manifest_path: str | Path) -> Path:
    return Path(str(manifest_path) + ".checkpoint.json")


def compute_manifest_hash(experiments: list[tuple[str, list[list[str]], int]]) -> str:
    h = hashlib.sha256()
    for name, commands, _ in experiments:
        h.update(name.encode())
        h.update(str(len(commands)).encode())
        for cmd in commands:
            for arg in cmd:
                h.update(arg.encode())
    return h.hexdigest()[:16]


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: Could not load checkpoint {path}: {e}")
        print("Starting from scratch.")
        return None


def save_checkpoint(checkpoint: dict, path: Path) -> None:
    tmp = path.with_suffix(".checkpoint.json.tmp")
    with tmp.open("w") as f:
        json.dump(checkpoint, f, indent=2)
    tmp.rename(path)


def init_checkpoint(
    experiments: list[tuple[str, list[list[str]], int]],
    manifest_hash: str,
) -> dict:
    runs = {}
    for name, commands, _ in experiments:
        for i, cmd in enumerate(commands):
            runs[f"{name}/{i}"] = {
                "command": cmd,
                "status": "pending",
            }
    return {"manifest_hash": manifest_hash, "runs": runs}


def merge_checkpoint(
    checkpoint: dict,
    experiments: list[tuple[str, list[list[str]], int]],
    manifest_hash: str,
) -> tuple[dict, list[str]]:
    old_runs = checkpoint.get("runs", {})
    warnings = []
    new_runs = {}

    for name, commands, _ in experiments:
        n_kept = 0
        n_changed = 0
        n_new = 0
        for i, cmd in enumerate(commands):
            key = f"{name}/{i}"
            if key in old_runs:
                if old_runs[key].get("command", []) == cmd:
                    new_runs[key] = old_runs[key]
                    n_kept += 1
                else:
                    new_runs[key] = {"command": cmd, "status": "pending"}
                    n_changed += 1
            else:
                new_runs[key] = {"command": cmd, "status": "pending"}
                n_new += 1

        if n_changed or n_new:
            parts = []
            if n_changed:
                parts.append(f"{n_changed} changed")
            if n_new:
                parts.append(f"{n_new} new")
            warnings.append(
                f"  {name}: {', '.join(parts)} run(s) reset to pending"
            )

    removed = len(set(old_runs.keys()) - set(new_runs.keys()))
    if removed:
        warnings.append(
            f"  {removed} run(s) removed from checkpoint "
            f"(no longer in manifest)"
        )

    return {"manifest_hash": manifest_hash, "runs": new_runs}, warnings


# ==================== Execution ====================

def run_experiment(
    name: str,
    commands: list[list[str]],
    max_parallel: int,
    checkpoint: dict,
    cp_path: Path,
    restart_failed: bool = False,
    restart_unknown: bool = False,
    on_interrupt: str = "prompt",
    spawn_delay: float = 0.0,
    max_retries: int = 0,
) -> None:
    total = len(commands)
    skip_ok = 0
    skip_fail = 0
    skip_unknown = 0
    queue: list[int] = []
    resumed: list[_Child] = []
    last_spawn_time = 0.0
    # Retries already performed per run idx (0 until first respawn).
    attempts: dict[int, int] = {}

    for i in range(total):
        key = f"{name}/{i}"
        entry = checkpoint["runs"][key]
        status = entry.get("status", "pending")
        if status == "ok":
            skip_ok += 1
        elif status == "fail":
            if restart_failed:
                queue.append(i)
            else:
                skip_fail += 1
        elif status == "unknown":
            if restart_unknown:
                queue.append(i)
            else:
                skip_unknown += 1
        elif status == "running":
            pid = entry.get("pid")
            if pid is not None and _pid_alive(pid):
                # Resume a child left running by a previous session: track it
                # without relaunching. Its exit code is unrecoverable (we are
                # not its parent), so on disappearance we mark it 'unknown'.
                resumed.append(
                    _Child(
                        name=name, idx=i, start_iso=entry.get("start_time", ""),
                        start=time.time(), exp_i=None, proc=None, pid=pid,
                    )
                )
            else:
                entry["status"] = "unknown"
                if restart_unknown:
                    queue.append(i)
                else:
                    skip_unknown += 1
        else:
            queue.append(i)

    print(f"\n{'='*60}")
    print(
        f"Experiment: {name} | {total} runs | max_parallel={max_parallel}"
        + (f" | spawn_delay={spawn_delay}s" if spawn_delay > 0 else "")
        + (f" | max_retries={max_retries}" if max_retries > 0 else "")
    )
    if skip_ok or skip_fail or skip_unknown:
        parts = []
        if skip_ok:
            parts.append(f"{skip_ok} ok")
        if skip_fail:
            parts.append(f"{skip_fail} failed")
        if skip_unknown:
            parts.append(f"{skip_unknown} unknown")
        print(f"  Skipping: {', '.join(parts)}")
        if skip_fail and not restart_failed:
            print(
                f"  (use --restart-failed to retry "
                f"{skip_fail} failed run(s))"
            )
        if skip_unknown and not restart_unknown:
            print(
                f"  (use --restart-unknown to retry "
                f"{skip_unknown} unknown run(s))"
            )
    if resumed:
        print(f"  Resuming: {len(resumed)} (live PID from previous session)")
    print(f"  Running: {len(queue)}/{total}")
    print(f"{'='*60}")

    if not queue and not resumed:
        print(f"  Experiment '{name}' — nothing to run.")
        return

    running: dict[int, _Child] = {c.pid: c for c in resumed}
    for c in resumed:
        print(f"  [{c.idx+1}/{total}] Resumed (PID {c.pid})")

    try:
        while queue or running:
            while queue and len(running) < max_parallel:
                idx = queue.pop(0)
                cmd = commands[idx]
                key = f"{name}/{idx}"
                _wait_for_spawn_slot(last_spawn_time, spawn_delay)
                start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
                checkpoint["runs"][key] = {
                    "command": cmd,
                    "status": "running",
                    "start_time": start_iso,
                }
                save_checkpoint(checkpoint, cp_path)
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, start_new_session=True,
                )
                last_spawn_time = time.time()
                running[proc.pid] = _Child(
                    name=name, idx=idx, start_iso=start_iso,
                    start=time.time(), exp_i=None, proc=proc, pid=proc.pid,
                    attempts=attempts.get(idx, 0),
                )
                print(
                    f"  [{idx+1}/{total}] Started (PID {proc.pid})"
                    + (f" attempt {attempts.get(idx, 0)+1}"
                       if attempts.get(idx, 0) > 0 else "")
                )

            finished = []
            for pid, child in running.items():
                ret = child.poll()
                if ret is not None:
                    elapsed = time.time() - child.start
                    # Auto-respawn: re-queue if the budget allows and the run
                    # didn't succeed. Respawn counts against `attempts[idx]`,
                    # not the launch-time attempts counter, so retries add up.
                    n_attempts = attempts.get(child.idx, 0)
                    if _should_respawn(ret, n_attempts, max_retries):
                        attempts[child.idx] = n_attempts + 1
                        queue.append(child.idx)
                        if ret == "unknown":
                            label = "UNKNOWN"
                        else:
                            label = f"FAIL(code={ret})"
                        print(
                            f"  [{child.idx+1}/{total}] {label} "
                            f"({elapsed:.1f}s) (PID {pid}) -> "
                            f"retrying ({n_attempts + 1}/{max_retries})"
                        )
                        finished.append(pid)
                        continue
                    if ret == "unknown":
                        child.finalize(
                            checkpoint, commands[child.idx], None, "unknown"
                        )
                        label = "UNKNOWN"
                    else:
                        status = "ok" if ret == 0 else "fail"
                        child.finalize(
                            checkpoint, commands[child.idx], ret, status
                        )
                        label = "OK" if ret == 0 else f"FAIL(code={ret})"
                        if ret != 0 and ret != "unknown" and n_attempts > 0:
                            label = (
                                f"FAIL(code={ret}) after "
                                f"{n_attempts} retr{'y' if n_attempts == 1 else 'ies'}"
                            )
                    save_checkpoint(checkpoint, cp_path)
                    print(
                        f"  [{child.idx+1}/{total}] {label} "
                        f"({elapsed:.1f}s) (PID {pid})"
                    )
                    finished.append(pid)

            for pid in finished:
                del running[pid]

            if running:
                time.sleep(0.5)
    except KeyboardInterrupt:
        _handle_interrupt(running, checkpoint, cp_path, on_interrupt)

    print(f"  Experiment '{name}' complete.")


def run_interleaved(
    experiments: list[tuple[str, list[list[str]], int]],
    global_max_parallel: int,
    checkpoint: dict,
    cp_path: Path,
    restart_failed: bool = False,
    restart_unknown: bool = False,
    on_interrupt: str = "prompt",
    spawn_delay: float = 0.0,
    max_retries: int = 0,
) -> None:
    total_runs = sum(len(cmds) for _, cmds, _ in experiments)

    skip_ok_total = 0
    skip_fail_total = 0
    skip_unknown_total = 0
    run_total = 0
    resume_total = 0

    for name, cmds, _ in experiments:
        for i in range(len(cmds)):
            key = f"{name}/{i}"
            entry = checkpoint["runs"][key]
            status = entry.get("status", "pending")
            if status == "ok":
                skip_ok_total += 1
            elif status == "fail":
                if restart_failed:
                    run_total += 1
                else:
                    skip_fail_total += 1
            elif status == "unknown":
                if restart_unknown:
                    run_total += 1
                else:
                    skip_unknown_total += 1
            elif status == "running":
                pid = entry.get("pid")
                if pid is not None and _pid_alive(pid):
                    resume_total += 1
                else:
                    entry["status"] = "unknown"
                    if restart_unknown:
                        run_total += 1
                    else:
                        skip_unknown_total += 1
            else:
                run_total += 1

    print(f"\n{'='*60}")
    print(
        f"Interleaved schedule | {total_runs} total runs | "
        f"global max_parallel={global_max_parallel}"
        + (f" | spawn_delay={spawn_delay}s" if spawn_delay > 0 else "")
        + (f" | max_retries={max_retries}" if max_retries > 0 else "")
    )
    if skip_ok_total or skip_fail_total or skip_unknown_total:
        parts = []
        if skip_ok_total:
            parts.append(f"{skip_ok_total} ok")
        if skip_fail_total:
            parts.append(f"{skip_fail_total} failed")
        if skip_unknown_total:
            parts.append(f"{skip_unknown_total} unknown")
        print(f"  Skipping: {', '.join(parts)}")
        if skip_fail_total and not restart_failed:
            print(
                f"  (use --restart-failed to retry "
                f"{skip_fail_total} failed run(s))"
            )
        if skip_unknown_total and not restart_unknown:
            print(
                f"  (use --restart-unknown to retry "
                f"{skip_unknown_total} unknown run(s))"
            )
    if resume_total:
        print(f"  Resuming: {resume_total} (live PID from previous session)")
    print(f"  Running: {run_total}/{total_runs}")
    print(f"{'='*60}")

    n_exp = len(experiments)
    queues: list[list[int]] = []
    exp_max: list[int] = []

    for name, cmds, mp in experiments:
        q = []
        for i in range(len(cmds)):
            key = f"{name}/{i}"
            entry = checkpoint["runs"][key]
            status = entry.get("status", "pending")
            if status == "ok":
                continue
            elif status == "fail" and not restart_failed:
                continue
            elif status == "unknown":
                if restart_unknown:
                    q.append(i)
                continue
            elif status == "running":
                # Skip here; live ones are seeded into `running` below.
                continue
            q.append(i)
        queues.append(q)
        exp_max.append(mp)

    per_exp_running: list[int] = [0] * n_exp
    per_exp_ok: list[int] = [0] * n_exp
    per_exp_fail: list[int] = [0] * n_exp

    # Retries already performed per (exp_i, run idx).
    attempts: dict[tuple[int, int], int] = {}

    running: dict[int, _Child] = {}

    last_spawn_time = 0.0

    # Seed resumed children (live PIDs left by a previous session).
    for exp_i, (name, cmds, _) in enumerate(experiments):
        for i in range(len(cmds)):
            entry = checkpoint["runs"][f"{name}/{i}"]
            if entry.get("status") != "running":
                continue
            pid = entry.get("pid")
            if pid is None or not _pid_alive(pid):
                continue
            running[pid] = _Child(
                name=name, idx=i, start_iso=entry.get("start_time", ""),
                start=time.time(), exp_i=exp_i, proc=None, pid=pid,
            )
            per_exp_running[exp_i] += 1
            print(f"  [{name} {i+1}/{len(cmds)}] Resumed (PID {pid})")

    def try_launch(rr_idx: int) -> bool:
        nonlocal last_spawn_time
        for offset in range(n_exp):
            i = (rr_idx + offset) % n_exp
            if not queues[i]:
                continue
            if per_exp_running[i] >= exp_max[i]:
                continue
            idx = queues[i].pop(0)
            name, cmds, _ = experiments[i]
            cmd = cmds[idx]
            key = f"{name}/{idx}"
            n_attempts = attempts.get((i, idx), 0)
            _wait_for_spawn_slot(last_spawn_time, spawn_delay)
            start_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
            checkpoint["runs"][key] = {
                "command": cmd,
                "status": "running",
                "start_time": start_iso,
            }
            save_checkpoint(checkpoint, cp_path)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
            last_spawn_time = time.time()
            running[proc.pid] = _Child(
                name=name, idx=idx, start_iso=start_iso,
                start=time.time(), exp_i=i, proc=proc, pid=proc.pid,
                attempts=n_attempts,
            )
            per_exp_running[i] += 1
            print(
                f"  [{name} {idx+1}/{len(cmds)}] "
                f"Started (PID {proc.pid})"
                + (f" attempt {n_attempts + 1}" if n_attempts > 0 else "")
            )
            return True
        return False

    if run_total == 0 and not running:
        print("  Nothing to run.")
        print_summary(experiments, checkpoint)
        return

    rr = 0
    try:
        while any(q for q in queues) or running:
            while sum(per_exp_running) < global_max_parallel:
                if not try_launch(rr):
                    break
                rr = (rr + 1) % n_exp

            finished = []
            for pid, child in running.items():
                ret = child.poll()
                if ret is not None:
                    elapsed = time.time() - child.start
                    name, cmds, _ = experiments[child.exp_i]
                    per_exp_running[child.exp_i] -= 1
                    # Auto-respawn: re-queue if the budget allows and the run
                    # didn't succeed.
                    akey = (child.exp_i, child.idx)
                    n_attempts = attempts.get(akey, 0)
                    if _should_respawn(ret, n_attempts, max_retries):
                        attempts[akey] = n_attempts + 1
                        queues[child.exp_i].append(child.idx)
                        if ret == "unknown":
                            label = "UNKNOWN"
                        else:
                            label = f"FAIL(code={ret})"
                        print(
                            f"  [{name} {child.idx+1}/{len(cmds)}] {label} "
                            f"({elapsed:.1f}s) (PID {pid}) -> "
                            f"retrying ({n_attempts + 1}/{max_retries})"
                        )
                        finished.append(pid)
                        continue
                    if ret == "unknown":
                        child.finalize(checkpoint, cmds[child.idx], None, "unknown")
                        label = "UNKNOWN"
                    else:
                        status = "ok" if ret == 0 else "fail"
                        child.finalize(
                            checkpoint, cmds[child.idx], ret, status
                        )
                        if ret == 0:
                            per_exp_ok[child.exp_i] += 1
                        else:
                            per_exp_fail[child.exp_i] += 1
                        label = "OK" if ret == 0 else f"FAIL(code={ret})"
                        if ret != 0 and n_attempts > 0:
                            label = (
                                f"FAIL(code={ret}) after "
                                f"{n_attempts} retr{'y' if n_attempts == 1 else 'ies'}"
                            )
                    save_checkpoint(checkpoint, cp_path)
                    print(
                        f"  [{name} {child.idx+1}/{len(cmds)}] {label} "
                        f"({elapsed:.1f}s) (PID {pid})"
                    )
                    finished.append(pid)

            for pid in finished:
                del running[pid]

            if running:
                time.sleep(0.5)
    except KeyboardInterrupt:
        _handle_interrupt(running, checkpoint, cp_path, on_interrupt)

    print_summary(experiments, checkpoint)


def print_summary(
    experiments: list[tuple[str, list[list[str]], int]],
    checkpoint: dict,
) -> None:
    print(f"\n{'='*60}")
    print("All experiments complete.")
    print(f"{'='*60}")
    for name, cmds, _ in experiments:
        total = len(cmds)
        ok = fail = pending = running = unknown = 0
        for i in range(total):
            key = f"{name}/{i}"
            status = checkpoint["runs"][key].get("status", "pending")
            if status == "ok":
                ok += 1
            elif status == "fail":
                fail += 1
            elif status == "running":
                running += 1
            elif status == "unknown":
                unknown += 1
            else:
                pending += 1
        parts = [f"{ok}/{total} OK"]
        if fail:
            parts.append(f"{fail} FAIL")
        if unknown:
            parts.append(f"{unknown} unknown")
        if pending:
            parts.append(f"{pending} pending")
        if running:
            parts.append(f"{running} interrupted")
        print(f"  {name}: {', '.join(parts)}")


# ==================== Main ====================

def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python run_scheduler.py <manifest.yaml> [--dry-run] "
            "[--restart-failed] [--restart-unknown] "
            "[--on-interrupt=prompt|terminate|leave] "
            "[--spawn-delay=SECONDS] [--max-retries=N] [--force]"
        )
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv
    restart_failed = "--restart-failed" in sys.argv
    restart_unknown = "--restart-unknown" in sys.argv
    force = "--force" in sys.argv

    on_interrupt = "prompt"
    for a in sys.argv[1:]:
        if a.startswith("--on-interrupt="):
            on_interrupt = a.split("=", 1)[1]
    if on_interrupt not in ("prompt", "terminate", "leave"):
        print(
            f"Error: --on-interrupt must be one of prompt|terminate|leave, "
            f"got '{on_interrupt}'"
        )
        sys.exit(1)

    spawn_delay = None
    for a in sys.argv[1:]:
        if a.startswith("--spawn-delay="):
            try:
                spawn_delay = float(a.split("=", 1)[1])
            except ValueError:
                print(
                    f"Error: --spawn-delay must be a number of seconds, "
                    f"got '{a.split('=', 1)[1]}'"
                )
                sys.exit(1)
            if spawn_delay < 0:
                print("Error: --spawn-delay must not be negative")
                sys.exit(1)

    max_retries = None
    for a in sys.argv[1:]:
        if a.startswith("--max-retries="):
            try:
                max_retries = int(a.split("=", 1)[1])
            except ValueError:
                print(
                    f"Error: --max-retries must be a non-negative integer, "
                    f"got '{a.split('=', 1)[1]}'"
                )
                sys.exit(1)
            if max_retries < 0:
                print("Error: --max-retries must not be negative")
                sys.exit(1)

    # Treat SIGTERM like Ctrl-C so `kill <scheduler_pid>` unwinds cleanly.
    signal.signal(signal.SIGTERM, _sigterm_handler)

    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not positional:
        print("Error: manifest path is required")
        sys.exit(1)
    manifest_path = positional[0]

    manifest = load_manifest(manifest_path)
    global_max_parallel = manifest.get("max_parallel", 1)
    schedule = manifest.get("schedule", "sequential")

    # CLI --spawn-delay overrides the manifest's spawn_delay (if any).
    if spawn_delay is None:
        spawn_delay = float(manifest.get("spawn_delay", 0.0))

    # CLI --max-retries overrides the manifest's max_retries (if any).
    if max_retries is None:
        max_retries = int(manifest.get("max_retries", 0))

    if schedule not in ("sequential", "interleaved"):
        raise ValueError(
            f"Unknown schedule mode '{schedule}'. "
            f"Must be 'sequential' or 'interleaved'."
        )

    experiments: list[tuple[str, list[list[str]], int]] = []
    for exp in manifest["experiments"]:
        name = exp["name"]
        commands = build_commands(exp)
        max_parallel = exp.get("max_parallel", global_max_parallel)
        experiments.append((name, commands, max_parallel))

    cp_path = checkpoint_path(manifest_path)
    current_hash = compute_manifest_hash(experiments)
    checkpoint = load_checkpoint(cp_path)

    if checkpoint is None:
        checkpoint = init_checkpoint(experiments, current_hash)
        save_checkpoint(checkpoint, cp_path)
    elif checkpoint.get("manifest_hash", "") != current_hash:
        if not force:
            expected_keys = set()
            for name, commands, _ in experiments:
                for i in range(len(commands)):
                    expected_keys.add(f"{name}/{i}")
            cp_keys = set(checkpoint.get("runs", {}).keys())
            new_count = len(expected_keys - cp_keys)
            removed_count = len(cp_keys - expected_keys)

            print(
                "ERROR: Manifest has changed since last run "
                "(checkpoint hash mismatch).\n"
                f"  New runs: {new_count}, "
                f"Removed runs: {removed_count}\n\n"
                "Options:\n"
                f"  1. Delete the checkpoint and re-run:\n"
                f"     rm {cp_path}\n"
                "  2. Use --force to merge (keep existing states, "
                "new runs as pending)"
            )
            sys.exit(1)
        else:
            checkpoint, warnings = merge_checkpoint(
                checkpoint, experiments, current_hash
            )
            save_checkpoint(checkpoint, cp_path)
            print(
                "WARNING: Manifest changed, merging checkpoint "
                "(--force):"
            )
            for w in warnings:
                print(w)
            print()
    else:
        for name, commands, _ in experiments:
            for i, cmd in enumerate(commands):
                key = f"{name}/{i}"
                checkpoint["runs"][key]["command"] = cmd
        save_checkpoint(checkpoint, cp_path)

    if dry_run:
        total_runs = sum(len(cmds) for _, cmds, _ in experiments)
        print(
            f"DRY RUN — {total_runs} total run(s) across "
            f"{len(experiments)} experiment(s)"
        )
        print(f"Schedule: {schedule}")

        n_ok = n_fail = n_running = n_unknown = n_pending = 0
        for v in checkpoint["runs"].values():
            s = v.get("status", "pending")
            if s == "ok":
                n_ok += 1
            elif s == "fail":
                n_fail += 1
            elif s == "running":
                n_running += 1
            elif s == "unknown":
                n_unknown += 1
            else:
                n_pending += 1

        if n_ok or n_fail or n_running or n_unknown:
            parts = []
            if n_ok:
                parts.append(f"{n_ok} ok")
            if n_fail:
                parts.append(f"{n_fail} failed")
            if n_unknown:
                parts.append(f"{n_unknown} unknown")
            if n_running:
                parts.append(f"{n_running} interrupted")
            if n_pending:
                parts.append(f"{n_pending} pending")
            print(f"Checkpoint: {', '.join(parts)}")

        print()
        for name, commands, max_parallel in experiments:
            print(
                f"--- {name} ({len(commands)} runs, "
                f"max_parallel={max_parallel}) ---"
            )
            for i, cmd in enumerate(commands):
                key = f"{name}/{i}"
                entry = checkpoint["runs"].get(key, {})
                status = entry.get("status", "pending")
                cmd_str = " ".join(cmd)
                if status == "ok":
                    dur = entry.get("duration_s")
                    dur_str = f" ({dur}s)" if dur is not None else ""
                    print(f"  [{i+1}] OK{dur_str}  {cmd_str}")
                elif status == "fail":
                    code = entry.get("exit_code", "?")
                    dur = entry.get("duration_s")
                    dur_str = f" ({dur}s)" if dur is not None else ""
                    if restart_failed:
                        print(
                            f"  [{i+1}] RETRY(code={code}){dur_str}  "
                            f"{cmd_str}"
                        )
                    else:
                        print(
                            f"  [{i+1}] SKIP(code={code}){dur_str}  "
                            f"{cmd_str}"
                        )
                elif status == "unknown":
                    if restart_unknown:
                        print(f"  [{i+1}] RETRY(unknown)  {cmd_str}")
                    else:
                        print(f"  [{i+1}] SKIP(unknown)  {cmd_str}")
                elif status == "running":
                    pid = entry.get("pid")
                    pid_str = f" pid={pid}" if pid is not None else ""
                    print(f"  [{i+1}] INTERRUPTED{pid_str}  {cmd_str}")
                else:
                    print(f"  [{i+1}] {cmd_str}")
            print()
        return

    try:
        if schedule == "sequential":
            for name, commands, max_parallel in experiments:
                run_experiment(
                    name, commands, max_parallel,
                    checkpoint, cp_path, restart_failed, restart_unknown,
                    on_interrupt, spawn_delay, max_retries,
                )
            print_summary(experiments, checkpoint)
        elif schedule == "interleaved":
            run_interleaved(
                experiments, global_max_parallel,
                checkpoint, cp_path, restart_failed, restart_unknown,
                on_interrupt, spawn_delay, max_retries,
            )
    except SchedulerInterrupted:
        print(
            "\nInterrupted. Re-run the same command to resume: runs left "
            "running can be resumed via their stored PID, finished runs keep "
            "their status, and the rest will be re-queued."
        )
        sys.exit(130)


if __name__ == "__main__":
    main()