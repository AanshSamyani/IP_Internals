# IP_Internals — Language steering vectors on Mistral-Small-24B-Instruct-2501

This repo runs a small interpretability experiment: build a **language steering
vector** for Spanish (and separately French) on
`Mistral-Small-24B-Instruct-2501` using parallel English / target-language
GSM8K responses, then sweep over steering strengths `lambda` to find a soft
spot that flips the model's output language without pushing activations
out-of-distribution.

## Pipeline

For each target language `L in {Spanish, French}`:

1. **`scripts/generate_steering_vector.py`** — picks 50 paired
   `(English-response, L-response)` samples from `data/gsm8k.jsonl` and
   `data/gsm8k_<L>_only.jsonl`. For each pair we run two forward passes (one
   with the English assistant response, one with the L response), mean the
   residual-stream activations across **only the assistant-content tokens** at
   a chosen middle/late decoder layer, take `mean_L - mean_English`, and
   finally average the per-sample directions over the 50 samples. The vector
   plus a sidecar `*.meta.json` (containing the questions used) are saved
   under `outputs/steering_vectors/`.

2. **`scripts/apply_steering.py`** — picks **another 50** GSM8K questions
   (excluding any used in step 1), registers a forward hook on the same
   decoder layer that adds `lambda * steering_vector` to the residual stream,
   and generates a completion for each `(question, lambda)` pair. Rollouts
   are written to `outputs/rollouts/`.

3. **`scripts/judge_language.py`** — word-level language judge (no LLM
   involved). Each completion is tokenized into words; each word is looked up
   in the `wordfreq` corpora for English and the target language and
   classified as `english` / `target` / `ambiguous` / `unknown`. The
   per-completion record reports the label (`english`, `target`, `both`, or
   `unknown`) along with `english_pct` and `target_pct`. A companion summary
   JSON aggregates per-lambda label histograms and average target-language
   percentage so you can eyeball the soft-spot lambda.

## Repo layout

```
.
├── data/
│   ├── gsm8k.jsonl                  # English Q + English A
│   ├── gsm8k_spanish_only.jsonl     # English Q + Spanish A (parallel)
│   └── gsm8k_french_only.jsonl      # English Q + French  A (parallel)
├── scripts/
│   ├── data_utils.py
│   ├── generate_steering_vector.py
│   ├── apply_steering.py
│   └── judge_language.py
├── outputs/
│   ├── steering_vectors/
│   ├── rollouts/
│   ├── judgements/
│   └── logs/
├── pyproject.toml
└── README.md
```

## Setup on an SSH server with a persistent `/workspace` volume

> **Persistence warning.** Only the `/workspace` directory survives across
> sessions on this box. `$HOME`, `/tmp`, and the default uv/pip/HF caches do
> **not** persist. Everything below — the repo clone, uv binary, uv cache,
> Python virtualenv, HuggingFace cache, model weights, and experiment
> outputs — is deliberately placed under `/workspace` so you don't have to
> redo the install after a restart.

### 0. Pick your persistent locations (one-time, copy-paste this block)

Add these to `/workspace/env.sh` (or paste at the top of every new shell).
Every subsequent command in this README assumes they are set.

```bash
# Persistent workspace root
export WS=/workspace

# Where uv lives (binary + cache)
export UV_INSTALL_DIR=$WS/bin
export UV_CACHE_DIR=$WS/.cache/uv
export PATH=$UV_INSTALL_DIR:$PATH

# Where HuggingFace caches downloads (tokenizer files, any fetched artefacts)
export HF_HOME=$WS/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# Point at wherever you have the Mistral weights staged inside /workspace
export MODEL_PATH=$WS/models/Mistral-Small-24B-Instruct-2501

mkdir -p $UV_INSTALL_DIR $UV_CACHE_DIR $HF_HOME $WS/models
```

Save the block as `/workspace/env.sh` so you can `source /workspace/env.sh`
on every new SSH session:

```bash
cat > /workspace/env.sh <<'EOF'
export WS=/workspace
export UV_INSTALL_DIR=$WS/bin
export UV_CACHE_DIR=$WS/.cache/uv
export PATH=$UV_INSTALL_DIR:$PATH
export HF_HOME=$WS/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export MODEL_PATH=$WS/models/Mistral-Small-24B-Instruct-2501
EOF
source /workspace/env.sh
```

### 1. Clone the repo into `/workspace`

```bash
cd /workspace
git clone https://github.com/AanshSamyani/IP_Internals.git
cd /workspace/IP_Internals
```

### 2. Install `uv` into `/workspace/bin`

The official installer honours `UV_INSTALL_DIR`, so the binary lands in a
persistent location:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version   # sanity check — should print a version string
```

If your server has no outbound `curl` but does have pip:

```bash
pip install --prefix=$WS --target=$UV_INSTALL_DIR uv
```

### 3. Create the virtualenv *inside the repo* (under `/workspace`)

Because the repo lives at `/workspace/IP_Internals`, its `.venv` directory
persists automatically. `uv sync` reads `pyproject.toml` and installs Torch,
Transformers, Accelerate, wordfreq, etc. into that venv.

```bash
cd /workspace/IP_Internals
uv venv --python 3.10 .venv
source .venv/bin/activate
uv sync
```

If the default Torch wheel is wrong for your CUDA version, install the right
one first and then re-run `uv sync`:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv sync
```

### 4. Stage the model weights under `/workspace`

Make sure the Mistral weights are actually on the persistent volume:

```bash
ls $MODEL_PATH   # should list config.json, tokenizer files, *.safetensors, ...
```

If the weights are somewhere ephemeral (e.g. `/root/models`), move or
symlink them first:

```bash
mv /root/models/Mistral-Small-24B-Instruct-2501 $WS/models/
# or
ln -s /root/models/Mistral-Small-24B-Instruct-2501 $WS/models/Mistral-Small-24B-Instruct-2501
```

### 5. Resuming in a fresh SSH session

After a disconnect / server restart, everything is still on disk under
`/workspace`. You only need to re-source the env file and re-activate the
venv:

```bash
source /workspace/env.sh
cd /workspace/IP_Internals
source .venv/bin/activate
```

No re-install required.

## Running the experiments (nohup)

All commands below assume:

- you are in `/workspace/IP_Internals`
- `/workspace/env.sh` has been sourced (so `MODEL_PATH` is set)
- `.venv` is activated

Logs go to `outputs/logs/` (which is under `/workspace/IP_Internals/outputs`
and therefore persistent). Every step is independent and resumable —
re-running a step overwrites its output file.

> **Layer choice.** The default decoder layer is `25`, which sits in the
> later half of `Mistral-Small-24B-Instruct-2501` (40 layers). Pass
> `--layer` to override.

Each `nohup ... &` line is followed by `disown` so the job keeps running if
the SSH session drops. Wait for step 1 to finish before launching step 2
(step 2 depends on the steering vector file produced by step 1).

### Spanish

```bash
cd /workspace/IP_Internals
source /workspace/env.sh
source .venv/bin/activate

# 1) build the Spanish steering vector
nohup python -m scripts.generate_steering_vector \
    --model-path "$MODEL_PATH" \
    --english-file /workspace/IP_Internals/data/gsm8k.jsonl \
    --target-file  /workspace/IP_Internals/data/gsm8k_spanish_only.jsonl \
    --layer 25 \
    --num-samples 50 \
    --seed 0 \
    --output /workspace/IP_Internals/outputs/steering_vectors/spanish_layer25.pt \
    > /workspace/IP_Internals/outputs/logs/spanish_steering.out 2>&1 &
disown

# Wait until spanish_layer25.pt exists, then:

# 2) generate steered rollouts across a lambda sweep
nohup python -m scripts.apply_steering \
    --model-path "$MODEL_PATH" \
    --english-file /workspace/IP_Internals/data/gsm8k.jsonl \
    --steering-vector /workspace/IP_Internals/outputs/steering_vectors/spanish_layer25.pt \
    --num-questions 50 \
    --question-seed 42 \
    --lambdas 0 1 2 3 4 5 6 8 \
    --max-new-tokens 300 \
    --output /workspace/IP_Internals/outputs/rollouts/spanish_rollouts.jsonl \
    > /workspace/IP_Internals/outputs/logs/spanish_rollouts.out 2>&1 &
disown

# 3) judge the rollouts (no LLM involved — uses wordfreq)
nohup python -m scripts.judge_language \
    --rollouts /workspace/IP_Internals/outputs/rollouts/spanish_rollouts.jsonl \
    --target-lang es \
    --output-jsonl /workspace/IP_Internals/outputs/judgements/spanish_judgements.jsonl \
    --output-summary /workspace/IP_Internals/outputs/judgements/spanish_summary.json \
    > /workspace/IP_Internals/outputs/logs/spanish_judge.out 2>&1 &
disown
```

### French

```bash
cd /workspace/IP_Internals
source /workspace/env.sh
source .venv/bin/activate

# 1) build the French steering vector
nohup python -m scripts.generate_steering_vector \
    --model-path "$MODEL_PATH" \
    --english-file /workspace/IP_Internals/data/gsm8k.jsonl \
    --target-file  /workspace/IP_Internals/data/gsm8k_french_only.jsonl \
    --layer 25 \
    --num-samples 50 \
    --seed 0 \
    --output /workspace/IP_Internals/outputs/steering_vectors/french_layer25.pt \
    > /workspace/IP_Internals/outputs/logs/french_steering.out 2>&1 &
disown

# Wait until french_layer25.pt exists, then:

# 2) generate steered rollouts across a lambda sweep
nohup python -m scripts.apply_steering \
    --model-path "$MODEL_PATH" \
    --english-file /workspace/IP_Internals/data/gsm8k.jsonl \
    --steering-vector /workspace/IP_Internals/outputs/steering_vectors/french_layer25.pt \
    --num-questions 50 \
    --question-seed 42 \
    --lambdas 0 1 2 3 4 5 6 8 \
    --max-new-tokens 300 \
    --output /workspace/IP_Internals/outputs/rollouts/french_rollouts.jsonl \
    > /workspace/IP_Internals/outputs/logs/french_rollouts.out 2>&1 &
disown

# 3) judge the rollouts
nohup python -m scripts.judge_language \
    --rollouts /workspace/IP_Internals/outputs/rollouts/french_rollouts.jsonl \
    --target-lang fr \
    --output-jsonl /workspace/IP_Internals/outputs/judgements/french_judgements.jsonl \
    --output-summary /workspace/IP_Internals/outputs/judgements/french_summary.json \
    > /workspace/IP_Internals/outputs/logs/french_judge.out 2>&1 &
disown
```

### Watch progress / check if a job is still running

```bash
tail -f /workspace/IP_Internals/outputs/logs/spanish_steering.out
tail -f /workspace/IP_Internals/outputs/logs/spanish_rollouts.out

# See which of your Python jobs are still alive
ps -ef | grep -E "scripts\.(generate_steering_vector|apply_steering|judge_language)" | grep -v grep
```

After the judge step finishes, look at
`outputs/judgements/<lang>_summary.json`. The `per_lambda` block lists, for
each `lambda`, how many of the 50 completions were labelled `english` /
`target` / `both` and the average target-language percentage. The "soft spot"
lambda is the smallest value where most rollouts are labelled `target` (or
`both` with high `target_pct`) **without** the completions degenerating into
gibberish — a quick sanity check is to skim a couple of completions in
`outputs/rollouts/<lang>_rollouts.jsonl` for that lambda.

## Notes

- Everything that needs to persist lives under `/workspace`: the repo clone,
  the `uv` binary (`/workspace/bin`), the uv cache (`/workspace/.cache/uv`),
  the HuggingFace cache (`/workspace/.cache/huggingface`), the venv
  (`/workspace/IP_Internals/.venv`), the model weights
  (`/workspace/models/...`), and all outputs
  (`/workspace/IP_Internals/outputs/...`). After a fresh SSH session, a
  `source /workspace/env.sh && cd /workspace/IP_Internals && source
  .venv/bin/activate` is all you need.
- `--device-map auto` shards the model across all visible GPUs. For a single
  H100/A100 node, `--device-map cuda` is fine.
- All decoding defaults to greedy (`--temperature 0`). If you want sampled
  rollouts, pass `--temperature 0.7 --top-p 0.95`.
- `generate_steering_vector.py` writes a sidecar `*.meta.json` listing the 50
  questions it used so the rollout script can guarantee a disjoint test set.
- The judge is intentionally simple: per-word `wordfreq` lookups with a
  configurable `--ratio-margin` (default 2.0) and `--label-threshold` (default
  0.80). It is not perfect but is enough to find the soft-spot lambda.
