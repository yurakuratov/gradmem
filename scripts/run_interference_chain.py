#!/usr/bin/env python
"""Checkpoint-chained single-key KV interference curriculum (one arm x one seed).

Executes the 5-phase N in {8,16,32,64,96} curriculum of ONE (arm, seed) chain
for the interference-curve experiment (tasks/INTERFERENCE_CURVE_TASK.md):
train N=8 from scratch, warm-start N=16 from the N=8 checkpoint, and so on.
The scheduler has no dependency support, so `run_scheduler.py` launches THIS
wrapper once per (arm, seed) via manifests/kv_interference_curve.yaml and the
wrapper runs the phases sequentially itself, one `run_from_config.py` call per
phase, resolving the previous phase's run dir after each phase.

Interface (scheduler-compatible positional key=val overrides):
    chain.arm=<interface|interface_softmax|adaptive|exact_softmax|linear_all|hopfield>
                                (required; the first two are the Revision-2
                                arms of the RUNNING interference job
                                (manifests/kv_interference_curve.yaml), the
                                third is the Revision-3 carried-memory
                                baseline (manifests/kv_interference_adaptive.yaml,
                                separate job), the rest are DEFERRED to a
                                later cycle -- definitions kept so a follow-up
                                manifest can run them without code changes)
    chain.seed=<int>                                (required)
    chain.phases=<int>          first-k phases only (default all 5)
    chain.base_model=<name>     backbone override (default llama; gpt2 for smoke)
    chain.max_steps / chain.eval_steps / chain.warmup_steps_1 / chain.warmup_steps_rest
    chain.per_device_batch_size / chain.total_batch_size
    chain.stop_on_em_threshold  (default 0.99)
    chain.data_root=<dir>       dir holding <data_name>/ DatasetDicts
                                (default: repo data/, else the remote absolute
                                data root -- see resolve_data_path)
    chain.run_name_prefix=<s>   prefix on run_name AND the state file (smoke
                                isolation; production runs use no prefix)

Also accepts `--dry-run` (print per-phase child commands, execute nothing) and
ignores a leading positional token without '=' (the scheduler's required
`base_config` placeholder -- this wrapper owns its configs).

Chain state lives in runs/interference_curve/<prefix><arm>_<seed>.chain.json;
phases recorded 'done' are skipped on rerun (idempotent resume). A phase that
exits non-zero marks the chain failed and stops it (other chains proceed).
A phase that finishes below the EM threshold still chains -- the curve records
the shortfall. If a rerun left several run_<seed>_* dirs, the newest mtime is
used (per-seed glob, so sibling seeds' runs are never picked up).

OOM note: per-phase per-device batch sizes in PHASES were chosen from the
local N=96 GPU memory probe (see the manifest header / task report);
total_batch_size stays 64 via gradient accumulation
(run_from_config.py derives grad_acc = total // per_device).
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / 'runs' / 'interference_curve'
REMOTE_DATA_ROOT = Path('/home/jovyan/dzhivelikian/test_time_gd/data')

# --------------------------------------------------------------------------- #
# Arms (same WRITE path everywhere; deltas exactly per the task brief).
#
# Revision 2 (2026-10-26, user-directed): the SUBMITTED arms of the RUNNING
# interference job are the two interface reads below. `linear_all`,
# all-positions `exact_softmax`, and `hopfield` are DEFERRED to a later cycle
# (not cancelled) -- their entries stay so a follow-up manifest can run them
# without code changes.
#
# Revision 3 (2026-11-02, user-directed): NEW arm `adaptive` -- the adaptive
# fork's native CROSS-SEGMENT MEMORY CARRY baseline (every segment's inner
# loop continues from the previous segment's memory; the running-sum
# m_t = m_{t-1} - lr*g overwrites earlier segments). Delivered as a SEPARATE
# job (manifests/kv_interference_adaptive.yaml + run_adaptive.yaml) on the
# identical curriculum/protocol/seeds; do NOT add it to the running job's
# manifest. Base config = adaptive_mamba.yaml with the gated bits stripped:
# memory_update_rule sgd builds NO gate heads (grad_memgpt_adaptive.py only
# constructs them for rule != "sgd"), so the leftover gate_* YAML keys are
# inert. adaptive_stage0.yaml was rejected as the base because it carries
# star_weight (a STAR key; Revision 3 says no STAR keys). segment_size must
# be nulled: adaptive_mamba.yaml sets it and the fork's config validation
# rejects pair_aware/n_segments>1 together with segment_size. The forgetting
# matrix (training.per_segment_eval, already true in the base config) is a
# PRIMARY readout for this arm -- cadence aligned to the interface arms
# (2500 steps). NOTE the budget asymmetry is the design: 2 carried tokens
# vs N stored tokens for the interface arms (architecture axis, not matched
# capacity).
# --------------------------------------------------------------------------- #

ARMS = {
    # (d') interface read, linear kernel: per-token keys, d_h=32, bias only at
    # the M=2 learnable interface tokens in a two-pass conditioning layout --
    # the deployable read (ladder's fastest, most seed-consistent solver)
    'interface': {
        'base_config': 'configs/gradmemgpt/kv_retrieval/linear_state_read_interface.yaml',
        'overrides': {},
    },
    # interface + exact softmax over the unfolded bank at the SAME interface
    # sites and two-pass conditioning -- the no-superposition control (O(N)
    # read; arm1 of manifests/kv_interface_softmax.yaml riding the curriculum)
    'interface_softmax': {
        'base_config': 'configs/gradmemgpt/kv_retrieval/linear_state_read_interface.yaml',
        'overrides': {'read_linear.read_state_exact_softmax': True},
    },
    # ---- Revision-3 carried-memory baseline (SEPARATE job, do not mix into
    # manifests/kv_interference_curve.yaml): adaptive fork, memory_update_rule
    # sgd (plain gradient inner loop; NO mamba/convex gating, NO STAR, NO
    # replay), memory-prefix READ as-is. Write hyperparameters matched to the
    # curve's arms: n_mem_tokens 2, K 2, inner_lr 0.04, grad_mode second,
    # use_write_head true (adaptive_mamba.yaml already has K/inner_lr/
    # grad_mode/use_write_head/add_inner_loss_to_outer identical to the
    # interface config; only n_mem_tokens 8->2 differs). Segmentation
    # identical: pair-aware, 2 pairs per segment via the per-phase
    # segmentation.n_segments = N/2 override (same mechanism as the interface
    # arms). per_phase_batch: the carry retains ONE autograd graph over all
    # n_segments*K second-order steps, so this arm carries its own batch table
    # from the 2026-11-02 N=96 probe (B=16 2.99 / B=8 1.51 GiB peak alloc =>
    # ~0.185 GiB/sample, linear in B with ~0.03 GiB intercept, ~14% LIGHTER
    # per sample than the interface arms' 0.215 -- the one-graph sgd carry
    # with M=2 beats the two-pass interface read). The ladder is IDENTICAL to
    # the interface arms' PHASES table: uniform per-phase batch protocol
    # across arms, and every entry's A100 projection for 3 concurrent chains
    # is <= ~12 GB -- see the header of manifests/kv_interference_adaptive.yaml.
    'adaptive': {
        'base_config': 'configs/gradmemgpt/kv_retrieval/adaptive_mamba.yaml',
        'overrides': {
            'adaptive.memory_update_rule': 'sgd',        # strip the mamba gated recurrence
            'gradmem.n_mem_tokens': 2,                   # match the curve's arms (M=2)
            'segmentation.pair_aware_segmentation': True,
            'segmentation.segment_size': None,           # mamba file sets 16; mutually exclusive w/ pair-aware
            'training.per_segment_eval_steps': 2500,     # forgetting matrix = primary readout; interface cadence
        },
        # measured 2026-11-02 (see manifest header): N=96 fits at B=16 (2.99
        # GiB) and not B>=32 on the 4GB probe card; identical ladder to the
        # interface arms (PHASES) -> uniform protocol across the arm family.
        'per_phase_batch': {8: 64, 16: 64, 32: 64, 64: 32, 96: 16},
    },
    # ---- deferred arms (keep for a later cycle) ------------------------------ #
    # (c) exact softmax over the unfolded bank -- the no-superposition control
    'exact_softmax': {
        'base_config': 'configs/gradmemgpt/kv_retrieval/linear_state_read.yaml',
        'overrides': {'read_linear.read_state_exact_softmax': True},
    },
    # (d) the superposed linear-attention state (O(1) read) -- proven config
    'linear_all': {
        'base_config': 'configs/gradmemgpt/kv_retrieval/linear_state_read.yaml',
        'overrides': {},
    },
    # (a) hopfield one-shot convex mixture -- the ladder hopfield_control deltas
    'hopfield': {
        'base_config': 'configs/gradmemgpt/kv_retrieval/hopfield.yaml',
        'overrides': {
            'gradmem.use_mem_proj': False,
            'gradmem.mem_proj_mode': 'none',
            'gradmem.add_inner_loss_to_outer': False,
            'gradmem.n_mem_tokens': 2,
        },
    },
}

# --------------------------------------------------------------------------- #
# Phase table: (N, segmentation.n_segments, per_device_batch_size)
#
# 2 pairs per segment into M=2 memory tokens everywhere (per-segment load is
# CONSTANT; only cross-talk between stored items grows). per-device batch from
# the local N=96 memory probe (RTX 3050 4GB, gpt2 backbone; extrapolated to
# the A100 with 3 concurrent chains -- see the manifest header): batch 64 fits
# comfortably through N=32; N=64/N=96 step down to 32/16 so three concurrent
# chains stay well inside the A100's 80GB. total_batch_size stays 64 via
# gradient accumulation in every phase.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Phase:
    n_pairs: int
    n_segments: int
    per_device_batch_size: int


PHASES = [
    Phase(8, 4, 64),
    Phase(16, 8, 64),
    Phase(32, 16, 64),
    Phase(64, 32, 32),
    Phase(96, 48, 16),
]

DEFAULTS = dict(
    base_model='llama',
    phases=None,              # None -> all 5; smoke passes 2 (N8->N16)
    max_steps=50000,
    eval_steps=500,          # config default, set explicitly for the record
    warmup_steps_1=10000,
    warmup_steps_rest=1000,
    per_device_batch_size=None,  # None -> per-phase table value
    total_batch_size=64,
    stop_on_em_threshold=0.99,
    data_root=None,
    run_name_prefix='',
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def format_value(value) -> str:
    """Render an override value as a Python literal for run_from_config's eval()."""
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return 'None'
    if isinstance(value, str):
        return repr(value)
    return json.dumps(value)


def resolve_data_path(data_name: str, data_root) -> Path:
    """Locate the DatasetDict dir for `data_name`.

    Search order: an explicit chain.data_root (smoke miniature data), the repo
    data/ dir (local machine), then the remote project's absolute data root
    (cluster jobs run from a .cryri copy whose ./data is excluded, so they must
    read the real project dir -- see docs/REMOTE.md).
    """
    candidates = []
    if data_root:
        p = Path(str(data_root))
        if not p.is_absolute():
            p = REPO_ROOT / p
        candidates.append(p / data_name)
    candidates.append(REPO_ROOT / 'data' / data_name)
    candidates.append(REMOTE_DATA_ROOT / data_name)
    for c in candidates:
        if (c / 'dataset_dict.json').exists():
            return c
    raise SystemExit(
        f'[chain] dataset dir for {data_name!r} not found; looked in: '
        + ', '.join(str(c) for c in candidates))


def phase_run_name(arm: str, phase: Phase, prefix: str) -> str:
    return f'{prefix}interf_{arm}_N{phase.n_pairs}'


def resolve_prev_run_dir(arm: str, phase: Phase, seed: int, prefix: str):
    """Newest run_<seed>_* dir under the phase's run-NAME dir (never another seed)."""
    run_name = phase_run_name(arm, phase, prefix)
    pattern = str(REPO_ROOT / 'runs' / f'N{phase.n_pairs}-K2V2-V62_1M' / run_name / f'run_{seed}_*')
    matches = [Path(m) for m in glob.glob(pattern)]
    if not matches:
        return None
    return max(matches, key=lambda d: d.stat().st_mtime)


def read_phase_metrics(run_dir: Path):
    """Best eval exact_match + final step count from the run dir."""
    best_em, steps = None, None
    ts = run_dir / 'trainer_state.json'
    if ts.exists():
        try:
            d = json.loads(ts.read_text())
            steps = d.get('global_step')
            ems = [e['eval_exact_match'] for e in d.get('log_history', [])
                   if isinstance(e, dict) and 'eval_exact_match' in e]
            if ems:
                best_em = max(ems)
        except (json.JSONDecodeError, OSError):
            pass
    if best_em is None:
        ar = run_dir / 'all_results.json'
        if ar.exists():
            try:
                best_em = json.loads(ar.read_text()).get('exact_match')
            except (json.JSONDecodeError, OSError):
                pass
    return best_em, steps


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {'phases': {}}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(state_path)


def parse_chain_args(argv):
    """Positional key=val tokens -> (chain dict, ignored positional tokens)."""
    chain, ignored = {}, []
    for tok in argv:
        if '=' not in tok:
            if tok.startswith('--'):
                continue  # unknown flag; argparse handles the ones we know
            ignored.append(tok)
            continue
        key, val = tok.split('=', 1)
        try:
            val = eval(val)  # noqa: S307 -- same convention as run_from_config.py
        except Exception:
            pass
        chain[key] = val
    return chain, ignored


# --------------------------------------------------------------------------- #
# chain execution
# --------------------------------------------------------------------------- #

def phase_per_device_batch(arm_cfg: dict, phase: Phase, knobs: dict) -> int:
    """Per-device batch for one phase: explicit chain knob > arm's per-phase
    table (arms whose memory profile differs from the global PHASES table,
    e.g. the Revision-3 carried-memory `adaptive` arm) > global PHASES value.
    Existing arms carry no `per_phase_batch` key, so their commands are
    unchanged."""
    if knobs['per_device_batch_size']:
        return knobs['per_device_batch_size']
    return arm_cfg.get('per_phase_batch', {}).get(phase.n_pairs,
                                                  phase.per_device_batch_size)


def build_phase_command(arm_cfg: dict, arm: str, seed: int, phase: Phase,
                        phase_idx: int, prev_run_dir, knobs) -> list[str]:
    data_name = f'N{phase.n_pairs}-K2V2-V62_1M'
    per_device = phase_per_device_batch(arm_cfg, phase, knobs)
    overrides = {
        'dataset.data_name': data_name,
        'dataset.data_path': str(resolve_data_path(data_name, knobs['data_root'])),
        'segmentation.n_segments': phase.n_segments,
        'model.base_model': knobs['base_model'],
        'training.max_steps': knobs['max_steps'],
        'training.stop_on_em_threshold': knobs['stop_on_em_threshold'],
        'training.early_stopping_patience': 50,
        'training.warmup_steps': (knobs['warmup_steps_1'] if phase_idx == 0
                                  else knobs['warmup_steps_rest']),
        'training.eval_steps': knobs['eval_steps'],
        'training.seed': seed,
        'training.per_device_batch_size': per_device,
        'training.total_batch_size': knobs['total_batch_size'],
        'run_name': phase_run_name(arm, phase, knobs['run_name_prefix']),
    }
    if prev_run_dir is not None:
        # weights-only warm start: the runner loads model.safetensors into the
        # fresh model (no optimizer state); M=2 is constant so no mem slicing.
        overrides['init_checkpoint'] = str(prev_run_dir)
    overrides.update(arm_cfg['overrides'])
    cmd = [sys.executable, 'run_from_config.py', '--config', arm_cfg['base_config']]
    cmd += [f'{k}={format_value(v)}' for k, v in overrides.items()]
    return cmd


def run_chain(arm: str, seed: int, knobs: dict, dry_run: bool) -> int:
    if arm not in ARMS:
        raise SystemExit(f'[chain] unknown arm {arm!r}; expected one of {sorted(ARMS)}')
    arm_cfg = ARMS[arm]
    phases = PHASES[:knobs['n_phases']]

    state_path = STATE_DIR / f"{knobs['run_name_prefix']}{arm}_{seed}.chain.json"
    state = load_state(state_path)

    print(f"[chain] arm={arm} seed={seed} phases={[p.n_pairs for p in phases]} "
          f"state={state_path}")
    if dry_run:
        prev = None
        for i, phase in enumerate(phases):
            cmd = build_phase_command(arm_cfg, arm, seed, phase, i, prev, knobs)
            print(f"[chain] phase {i + 1} (N={phase.n_pairs}) command:\n  {' '.join(cmd)}")
            prev = resolve_prev_run_dir(arm, phase, seed, knobs['run_name_prefix'])
            if prev is None:
                # show the chaining token even before the run dir exists
                prev = (REPO_ROOT / 'runs' / f'N{phase.n_pairs}-K2V2-V62_1M'
                        / phase_run_name(arm, phase, knobs['run_name_prefix'])
                        / f'run_{seed}_<uid8>')
        return 0

    prev_run_dir = None
    for i, phase in enumerate(phases):
        key = f'N{phase.n_pairs}'
        rec = state['phases'].get(key, {})
        if rec.get('status') == 'done':
            print(f"[chain] phase {i + 1} (N={phase.n_pairs}): already done "
                  f"(best_em={rec.get('best_em')}), skipping")
            prev_run_dir = rec.get('run_dir')
            continue

        if i > 0:
            prev_run_dir = resolve_prev_run_dir(arm, phases[i - 1], seed,
                                                knobs['run_name_prefix'])
            if prev_run_dir is None:
                prev_run_dir = state['phases'].get(f'N{phases[i - 1].n_pairs}', {}).get('run_dir')
            if prev_run_dir is None or not Path(prev_run_dir).exists():
                rec = dict(rec, status='failed',
                           error=f'cannot resolve previous phase N={phases[i - 1].n_pairs} run dir')
                state['phases'][key] = rec
                save_state(state_path, state)
                print(f"[chain] FAIL: {rec['error']}")
                return 1

        cmd = build_phase_command(arm_cfg, arm, seed, phase, i, prev_run_dir, knobs)
        rec = dict(rec, status='running', start=time.strftime('%Y-%m-%dT%H:%M:%S'),
                   command=cmd, init_from=str(prev_run_dir) if prev_run_dir else None)
        state['phases'][key] = rec
        save_state(state_path, state)
        print(f"[chain] phase {i + 1} (N={phase.n_pairs}): running\n  {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        wall = round(time.time() - t0, 1)

        run_dir = resolve_prev_run_dir(arm, phase, seed, knobs['run_name_prefix'])
        if proc.returncode != 0 or run_dir is None:
            rec = dict(rec, status='failed', wall_s=wall, exit_code=proc.returncode,
                       run_dir=str(run_dir) if run_dir else None)
            state['phases'][key] = rec
            save_state(state_path, state)
            print(f"[chain] phase {i + 1} (N={phase.n_pairs}) FAILED "
                  f"(exit={proc.returncode}, run_dir={run_dir}); stopping chain")
            return 1

        best_em, steps = read_phase_metrics(run_dir)
        rec = dict(rec, status='done', wall_s=wall, exit_code=0, run_dir=str(run_dir),
                   best_em=best_em, steps=steps,
                   end=time.strftime('%Y-%m-%dT%H:%M:%S'))
        state['phases'][key] = rec
        save_state(state_path, state)
        print(f"[chain] phase {i + 1} (N={phase.n_pairs}) done in {wall}s: "
              f"best_em={best_em} steps={steps} run_dir={run_dir}")

    print(f"[chain] arm={arm} seed={seed} complete: "
          + ', '.join(f"N{p.n_pairs}={state['phases'][f'N{p.n_pairs}'].get('best_em')}"
                      for p in phases))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dry-run', action='store_true',
                        help='print per-phase child commands, execute nothing')
    known, rest = parser.parse_known_args()

    chain, ignored = parse_chain_args(rest)
    if ignored:
        print(f"[chain] ignoring non key=val positional token(s) (scheduler "
              f"base_config placeholder): {ignored}")
    bad = [k for k in chain
           if k not in ('chain.arm', 'chain.seed')
           and not (k.startswith('chain.') and k[len('chain.'):] in DEFAULTS)]
    if bad:
        raise SystemExit(f'[chain] unknown override key(s): {bad} '
                         f'(expected chain.arm, chain.seed, chain.<'
                         + '|'.join(DEFAULTS) + '>)')

    for req in ('chain.arm', 'chain.seed'):
        if req not in chain:
            raise SystemExit(f'[chain] missing required override {req}')

    knobs = dict(DEFAULTS)
    for k, v in chain.items():
        if k not in ('chain.arm', 'chain.seed'):
            knobs[k[len('chain.'):]] = v
    knobs['n_phases'] = int(knobs.get('phases') or len(PHASES))

    return run_chain(chain['chain.arm'], int(chain['chain.seed']), knobs, known.dry_run)


if __name__ == '__main__':
    sys.exit(main())
