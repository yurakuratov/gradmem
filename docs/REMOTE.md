# REMOTE.md — working on the remote GPU cluster + cryri job system

`cloud-lolora` (ssh-target) is a **remote shared machine** (login node, no
local GPU, no `nvidia-smi`).  Compute runs as containerized cloud jobs via
**cryri** (`cr` CLI).  The project lives at
`/home/jovyan/dzhivelikian/test_time_gd/` (sync from the laptop with tar over
ssh — the login node has no rsync).

## Working submission recipe

From the laptop, in one ssh call per step:

```bash
# 1. sync code (repo root -> remote project dir). data/ and runs/ live on the
#    remote and are NOT overwritten by this tar (they are excluded); .venv is
#    rebuilt by the job's `uv sync`. *.checkpoint.json is excluded because a
#    LOCAL dry-run's checkpoint tar-synced to the project dir makes the job's
#    scheduler abort on a manifest-hash mismatch.
tar czf - --exclude=.venv --exclude=.git --exclude=runs --exclude=data \
    --exclude=notebooks --exclude=__pycache__ --exclude=.pytest_cache \
    --exclude=.idea --exclude=.cometml-runs --exclude='*.log' \
    --exclude='*.checkpoint.json' . \
  | ssh -o RemoteCommand=none cloud-lolora 'tar xzf - -C /home/jovyan/dzhivelikian/test_time_gd'

# 2. submit. The shell MUST have the HF_*/COMET_*/MY_HOME env — .dzhivelikian.bashrc
#    has an INTERACTIVITY GUARD (case $- in *i*) ;; *) return;; esac), so sourcing it
#    from a plain `ssh host bash -c '...'` silently sets NOTHING and run.yaml's
#    "$HF_TOKEN" stays a literal string in the job env (-> 401 GatedRepoError on the
#    llama config, comet auth broken). Run the submit through an interactive shell:
ssh -o RemoteCommand=none cloud-lolora 'bash -i -c "
  source /home/jovyan/dzhivelikian/.dzhivelikian.bashrc
  cd /home/jovyan/dzhivelikian/test_time_gd
  bash /home/jovyan/dzhivelikian/.cr_env.sh submit run.yaml -y"'

#    !! SUBMIT-TIMEOUT FAN-OUT (seen 2026-10-19) !! When the API gateway is
#    slow, `cr submit` prints "Failed to submit job: API error 0: ... Read
#    timed out (read timeout=30)" but has CREATED a job per internal retry
#    attempt server-side (4-8 jobs observed, plus late spawns MINUTES after
#    the client exited) — all pinned to the ONE run copy made before the
#    retry loop, so the racing `uv sync`s corrupt the shared .venv.
#    - NEVER resubmit on that error without first auditing `cr jobs`
#      (ours are identifiable by description in the platform UI, or by
#      creation-time clustering right after the submit).
#    - run.yaml's command starts with a flock guard: `exec 9>.job_lock &&
#      { flock -n 9 || { echo DUPLICATE_SPAWN_LOCK_HELD_EXITING; exit 0; } }
#      && ...`. Duplicates self-exit (status "Completed" in ~40 s); exactly
#      one job runs. KEEP THE GUARD when editing the command.
#    - Do not `cr kill` jobs without checking the description — the
#      namespace is shared; other users' jobs look identical in `cr jobs`.

# 3. monitor / fetch artifacts (cr logs can hang; read the run copy instead)
ssh -o RemoteCommand=none cloud-lolora 'bash /home/jovyan/dzhivelikian/.cr_env.sh status <hash-prefix>'
ssh -o RemoteCommand=none cloud-lolora 'latest=$(ls -td /home/jovyan/dzhivelikian/.cryri/run_* | head -1); ls $latest/runs/ 2>/dev/null; tail -50 $latest/*_output 2>/dev/null'
```

`/home/jovyan/dzhivelikian/.cr_env.sh` sources the user bashrc, exports
`TMUX_TMPDIR`, and pulls `GWAPI_ADDR/GWAPI_KEY/GWAPI_V2_URL/NAMESPACE` from
the jupyter-lab process (`/proc/<pid>/environ`) when tmux has no server.

**Selecting the sweep:** this repo's `run.yaml` hardcodes the manifest in its
command (`uv run python run_scheduler.py manifests/kv_hopfield.yaml`) — unlike
gradmem_rl there is no `MANIFEST` env var. To run a different manifest, edit
the manifest path in `run.yaml` before submitting (the edit rides along in the
job copy), or override the whole command with `cr submit run.yaml -c '<cmd>'`.
Instance: `a100plus.1gpu.80vG.12C.96G` (region `SR006`, image
`cuda12.3-torch2-py310`).

## Environment / shell setup

A fresh shell needs one thing:

```bash
source ../.dzhivelikian.bashrc   # relative to the project root
```

- `../.dzhivelikian.bashrc` exports `HF_HOME` (`/home/jovyan/.cache/huggingface`),
  `HF_TOKEN`, `COMET_API_KEY`, `COMET_WORKSPACE`, `TRITON_CACHE_DIR`,
  `UV_CACHE_DIR` (`/home/jovyan/dzhivelikian/.cache/uv`), `ME`, `MY_HOME`, …
  These variables are also **expanded inside `run.yaml`** at submit time, so
  `cr submit` must run from a shell that has them set.
- `uv` binary: `/home/jovyan/.local/bin/uv` (also a zsh wrapper function);
  Python 3.11 is the system interpreter and the project env
  (`uv sync` — the pin lives in `pyproject.toml`).
- **Datasets stay on the remote**: `data/` (KV-retrieval, bAbI) is excluded
  from both the tar sync and the job copy (`exclude_from_copy`), and the
  manifests reference it by absolute path
  (`/home/jovyan/dzhivelikian/test_time_gd/data/...`), so jobs read it
  directly from the real project dir. Do not delete it there. Fetch new
  datasets with `scripts/download_kv_retrieval.sh` /
  `scripts/download_babi.sh` inside the remote project dir.
- Comet project is baked into `run.yaml` as
  `COMET_PROJECT_NAME=gradmem-mem-stage0-inter`.

## cryri crash course

| Command | What it does |
|---|---|
| `cr submit run.yaml` | Submit a job. `-y` skips the confirmation prompt; `-c`/`-i`/`-e`/`-w` override command/instance/env/workers. |
| `cr jobs [-n N] [-s STATUS] [--since 30m]` | List jobs (default 10, `-n 0` = all). The namespace is **shared** — many other users' `lm-mpi-job-*` jobs appear; find yours by creation time or use the hash printed at submit. |
| `cr status <hash>` | Status of one job. `<hash>` is any prefix of the job name (the UUID part). |
| `cr logs <hash>` | Print job logs. `-f` follows until the job finishes, `--raw` disables filtering. Without a hash it opens an interactive picker. |
| `cr kill <hash>` | Terminate a job. |
| `cr instances` | List instance types per region (region default `SR006`). |

Typical loop:

```bash
source ../.dzhivelikian.bashrc
cr submit run.yaml                 # prints "Job submitted: lm-mpi-job-<uuid>"
cr logs -f lm-mpi-job-<uuid>       # or a unique prefix of the uuid
```

Save the job name printed at submission — old jobs scroll out of `cr jobs`,
and the name is the only way to reach `status`/`logs`/`kill` later.

## Where files live (`.cryri/` run copies)

With `run_from_copy: True` (this repo's setting, `cry_copy_dir: $MY_HOME/.cryri`),
every `cr submit` copies the project (minus `exclude_from_copy`: `.venv`,
`.git`, `notebooks`, `data`) into

```
$MY_HOME/.cryri/run_<YYYY_MM_DD_HHMM>_<hash>/   # sibling area, not inside the project
```

The job's container runs inside that copy (its `work_dir: '.'`), so:

- **Never edit anything under `.cryri/`** — it's a snapshot of submission time.
  Fix code in the real project and resubmit; each submit makes a fresh copy.
- **Artifacts sync back** into the run copy while/after the job runs:
  `runs/` (per-experiment metrics/checkpoints — `run_from_config.py` exp dirs)
  and `<jobhash>_output` (the saved console log). Pull results back to the
  laptop with tar over ssh from the run copy, or read metrics straight off the
  login node.
- A failed job leaves its `.venv` behind (the cleanup `rm -r .venv` at the end
  of `run.yaml`'s command never runs) — safe to delete to save space.

## Job configuration (run.yaml, already set up)

- Image `cr.ai.cloud.ru/aicloud-base-images/cuda12.3-torch2-py310:0.0.37`,
  instance `a100plus.1gpu.80vG.12C.96G` (1×A100 80GB), region `SR006`.
- The command installs `uv` (pip `--user` → `$HOME/.local/bin`), runs
  `uv sync`, then `uv run python run_scheduler.py manifests/kv_hopfield.yaml`,
  and removes `.venv` on success.
- Env passthrough: `HF_HOME`, `HF_TOKEN`, `TRITON_CACHE_DIR`,
  `COMET_API_KEY`, `COMET_WORKSPACE`, `COMET_PROJECT_NAME`, `UV_CACHE_DIR`
  (shared caches → `uv sync` is fast after the first run).

## Local (CPU) verification before submitting

The login node has no GPU, but config/scheduler plumbing is CPU-testable:

```bash
cd /home/jovyan/dzhivelikian/test_time_gd
/home/jovyan/.local/bin/uv sync    # UV_CACHE_DIR=... to reuse the shared cache
uv run python run_from_config.py --config configs/gradmemgpt/kv_retrieval/hopfield.yaml --dry-run
uv run python run_scheduler.py manifests/kv_hopfield.yaml --dry-run
```
