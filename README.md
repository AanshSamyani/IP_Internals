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

## Setup with `uv`

[`uv`](https://docs.astral.sh/uv/) is used for dependency management. It is a
single static binary, much faster than pip, and has reproducible env handling.

### 1. Install `uv`

On Linux / macOS (recommended for the SSH server):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# then make it available in the current shell
source $HOME/.local/bin/env   # or: export PATH="$HOME/.local/bin:$PATH"
uv --version
```

If you'd rather install through pip:

```bash
pip install --user uv
```

### 2. Create the virtual environment and install deps

From the repo root:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync
```

`uv sync` reads `pyproject.toml` and installs everything into `.venv`,
including `torch`, `transformers`, `accelerate`, `wordfreq`, etc. If you need a
CUDA-specific Torch wheel, install it explicitly first:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv sync
```

### 3. Make sure the model weights are reachable

Put / symlink the local weights at some path, e.g.
`/data/models/Mistral-Small-24B-Instruct-2501`, and export it once for
convenience:

```bash
export MODEL_PATH=/data/models/Mistral-Small-24B-Instruct-2501
```

## Running the experiments (nohup)

All commands assume you are in the repo root with `.venv` activated and
`MODEL_PATH` exported. Logs go to `outputs/logs/`. Every step is independent
and resumable — re-running a step overwrites its output file.

> **Layer choice.** The default decoder layer is `25`, which sits in the
> later half of `Mistral-Small-24B-Instruct-2501` (40 layers). Pass
> `--layer` to override.

### Spanish

```bash
# 1) build the Spanish steering vector
nohup python -m scripts.generate_steering_vector \
    --model-path "$MODEL_PATH" \
    --english-file data/gsm8k.jsonl \
    --target-file  data/gsm8k_spanish_only.jsonl \
    --layer 25 \
    --num-samples 50 \
    --seed 0 \
    --output outputs/steering_vectors/spanish_layer25.pt \
    > outputs/logs/spanish_steering.out 2>&1 &

# 2) generate steered rollouts across a lambda sweep
nohup python -m scripts.apply_steering \
    --model-path "$MODEL_PATH" \
    --english-file data/gsm8k.jsonl \
    --steering-vector outputs/steering_vectors/spanish_layer25.pt \
    --num-questions 50 \
    --question-seed 42 \
    --lambdas 0 1 2 3 4 5 6 8 \
    --max-new-tokens 300 \
    --output outputs/rollouts/spanish_rollouts.jsonl \
    > outputs/logs/spanish_rollouts.out 2>&1 &

# 3) judge the rollouts (no LLM involved — uses wordfreq)
nohup python -m scripts.judge_language \
    --rollouts outputs/rollouts/spanish_rollouts.jsonl \
    --target-lang es \
    --output-jsonl outputs/judgements/spanish_judgements.jsonl \
    --output-summary outputs/judgements/spanish_summary.json \
    > outputs/logs/spanish_judge.out 2>&1 &
```

### French

```bash
# 1) build the French steering vector
nohup python -m scripts.generate_steering_vector \
    --model-path "$MODEL_PATH" \
    --english-file data/gsm8k.jsonl \
    --target-file  data/gsm8k_french_only.jsonl \
    --layer 25 \
    --num-samples 50 \
    --seed 0 \
    --output outputs/steering_vectors/french_layer25.pt \
    > outputs/logs/french_steering.out 2>&1 &

# 2) generate steered rollouts across a lambda sweep
nohup python -m scripts.apply_steering \
    --model-path "$MODEL_PATH" \
    --english-file data/gsm8k.jsonl \
    --steering-vector outputs/steering_vectors/french_layer25.pt \
    --num-questions 50 \
    --question-seed 42 \
    --lambdas 0 1 2 3 4 5 6 8 \
    --max-new-tokens 300 \
    --output outputs/rollouts/french_rollouts.jsonl \
    > outputs/logs/french_rollouts.out 2>&1 &

# 3) judge the rollouts
nohup python -m scripts.judge_language \
    --rollouts outputs/rollouts/french_rollouts.jsonl \
    --target-lang fr \
    --output-jsonl outputs/judgements/french_judgements.jsonl \
    --output-summary outputs/judgements/french_summary.json \
    > outputs/logs/french_judge.out 2>&1 &
```

### Watch progress

```bash
tail -f outputs/logs/spanish_steering.out
tail -f outputs/logs/spanish_rollouts.out
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

- `--device-map auto` shards the model across all visible GPUs. For a single
  H100/A100 node, `--device-map cuda` is fine.
- All decoding defaults to greedy (`--temperature 0`). If you want sampled
  rollouts, pass `--temperature 0.7 --top-p 0.95`.
- `generate_steering_vector.py` writes a sidecar `*.meta.json` listing the 50
  questions it used so the rollout script can guarantee a disjoint test set.
- The judge is intentionally simple: per-word `wordfreq` lookups with a
  configurable `--ratio-margin` (default 2.0) and `--label-threshold` (default
  0.80). It is not perfect but is enough to find the soft-spot lambda.
