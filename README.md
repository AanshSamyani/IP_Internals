# IP_Internals — Language steering vectors on Mistral-Small-24B-Instruct-2501

This repo runs a small interpretability experiment: build a **language steering
vector** for Spanish (and separately French) on
`Mistral-Small-24B-Instruct-2501` using parallel English / target-language
GSM8K responses, then sweep over steering strengths `lambda` to find a soft
spot that flips the model's output language without pushing activations
out-of-distribution.

Model loading prefers **unsloth** (auto dtype, no quantization) when installed,
with an automatic fallback to plain `transformers` (`torch_dtype="auto"`).

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

### Finetuning pipeline (CAFT-style steering injection)

Inspired by **CAFT** (Casademunt et al., 2025 — *Concept Ablation
Fine-Tuning*), this pipeline tests whether injecting a steering vector
*during training* (not just at inference) changes what the model learns.

4. **`scripts/prepare_finetune_data.py`** — builds a mixed training set
   (half Spanish responses, half French responses, different questions
   from the GSM8K training split, excluding steering-vector questions)
   and downloads the GSM8K test split for evaluation.

5. **`scripts/finetune.py`** — LoRA SFT via unsloth + `SFTTrainer`.
   Two modes:
   - **Baseline**: standard SFT on the mixed data.
   - **Steered**: registers a forward hook that adds
     `lambda * steering_vector` at a decoder layer during every forward
     pass.  Gradients flow through the hook (the vector is a frozen
     constant).  After training the hook is removed.

6. **`scripts/generate_test_rollouts.py`** — loads the base model +
   LoRA adapter and generates completions on the held-out GSM8K test set.

7. **`scripts/judge_multilingual.py`** — three-way word-frequency judge
   (English / Spanish / French) that reports per-language percentages
   for each completion and an aggregate summary.

## Repo layout

```
.
├── data/
│   ├── gsm8k.jsonl                  # English Q + English A
│   ├── gsm8k_spanish_only.jsonl     # English Q + Spanish A (parallel)
│   ├── gsm8k_french_only.jsonl      # English Q + French  A (parallel)
│   ├── finetune_train.jsonl         # (generated) mixed ES/FR training set
│   └── gsm8k_test.jsonl            # (generated) GSM8K test split
├── scripts/
│   ├── data_utils.py
│   ├── download_model.py            # download weights from HF Hub
│   ├── generate_steering_vector.py
│   ├── apply_steering.py
│   ├── judge_language.py
│   ├── prepare_finetune_data.py     # build mixed train set + test set
│   ├── finetune.py                  # baseline + steered SFT
│   ├── generate_test_rollouts.py    # rollouts from finetuned model
│   └── judge_multilingual.py        # EN/ES/FR three-way judge
├── outputs/
│   ├── steering_vectors/
│   ├── rollouts/
│   ├── judgements/
│   ├── checkpoints/                 # LoRA adapters from finetuning
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
export MODEL_PATH=$WS/models/unsloth_Mistral_Small_24B_Instruct_2501

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
export MODEL_PATH=$WS/models/unsloth_Mistral_Small_24B_Instruct_2501
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
persists automatically.

```bash
cd /workspace/IP_Internals
uv venv --python 3.10 .venv
source .venv/bin/activate
```

#### 3a. Install a CUDA-matched Torch wheel **first**

> **What `nvidia-smi` tells you.** The "CUDA Version" shown by `nvidia-smi`
> (e.g. `CUDA Version: 13.0` on driver `580.126.16`) is the **maximum** CUDA
> runtime the installed driver can support, not a runtime you must match
> exactly. NVIDIA drivers are forward-compatible: a driver that advertises
> CUDA 13 can happily run PyTorch binaries built against CUDA 12.x (or even
> 11.x). You do **not** need a CUDA-13-specific Torch build — and as of
> today there is not a stable one published anyway.
>
> **What to install.** For this box (A100-SXM4-80GB, SM_80, driver 580 /
> CUDA 13) install the CUDA 12.4 PyTorch wheel. It is the most recent
> stable line PyTorch ships, works out of the box on every driver ≥ 525,
> and A100 is fully supported.

**Recommended — install from PyPI (much faster than `download.pytorch.org`).**
Since Torch 2.5, the default PyPI wheel for Linux x86_64 is already
CUDA-enabled (cu12x bundled), and PyPI is served by Fastly which is
typically 10–20x faster than `download.pytorch.org`'s CDN. Just omit
`--index-url`:

```bash
uv pip install torch torchvision torchaudio
```

Verify before moving on:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: 2.x.y+cu12x  True  NVIDIA A100-SXM4-80GB
```

Any `cu121` / `cu124` / `cu126` build works on driver 580 — forward
compatibility guarantees it.

<details>
<summary><b>Alternative — install directly from <code>download.pytorch.org</code></b> (only use if PyPI doesn't have a CUDA wheel for your Python version)</summary>

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision torchaudio
```

> **Warning — the `download.pytorch.org` CDN is slow.** The full install is
> ~2.6 GB of wheels (torch 732 MB, cudnn 634 MB, cublas 347 MB, etc.) and
> the CDN throttles each connection, so the big wheels can crawl at
> 1–2 MB/s. `uv`'s progress bar only redraws when chunks land, so it can
> *look* frozen while bytes are still flowing. To verify from a **second**
> SSH session:
>
> ```bash
> source /workspace/env.sh
> ps -ef | grep -E "uv pip" | grep -v grep          # process still alive?
> watch -n 2 'du -sh /workspace/.cache/uv'          # cache still growing?
> UVPID=$(pgrep -f "uv pip install" | head -1) && \
>     ls -l /proc/$UVPID/fd 2>/dev/null | grep -c socket   # open sockets > 0?
> ```
>
> If `du` is growing, leave it alone — it's slow, not stuck. If it's flat
> for >60 s, Ctrl+C and retry with tamer settings (uv caches partial
> downloads, so retries resume):
>
> ```bash
> export UV_CONCURRENT_DOWNLOADS=2
> export UV_HTTP_TIMEOUT=600
> uv pip install --index-url https://download.pytorch.org/whl/cu124 \
>     torch torchvision torchaudio
> ```
>
> **Still slow? Use `aria2c` for multi-connection byte-range downloads.**
> This bypasses per-connection CDN throttling completely:
>
> ```bash
> apt-get update && apt-get install -y aria2
>
> mkdir -p /workspace/wheels && cd /workspace/wheels
>
> TORCH_WHL=$(curl -s https://download.pytorch.org/whl/cu124/torch/ \
>     | grep -oE 'torch-[0-9.]+%2Bcu124-cp310-cp310-linux_x86_64\.whl' | sort -u | tail -1)
> TV_WHL=$(curl -s https://download.pytorch.org/whl/cu124/torchvision/ \
>     | grep -oE 'torchvision-[0-9.]+%2Bcu124-cp310-cp310-linux_x86_64\.whl' | sort -u | tail -1)
> TA_WHL=$(curl -s https://download.pytorch.org/whl/cu124/torchaudio/ \
>     | grep -oE 'torchaudio-[0-9.]+%2Bcu124-cp310-cp310-linux_x86_64\.whl' | sort -u | tail -1)
>
> aria2c -x 16 -s 16 -k 1M "https://download.pytorch.org/whl/cu124/torch/$TORCH_WHL"
> aria2c -x 16 -s 16 -k 1M "https://download.pytorch.org/whl/cu124/torchvision/$TV_WHL"
> aria2c -x 16 -s 16 -k 1M "https://download.pytorch.org/whl/cu124/torchaudio/$TA_WHL"
>
> cd /workspace/IP_Internals && source .venv/bin/activate
> uv pip install --index-url https://download.pytorch.org/whl/cu124 \
>     /workspace/wheels/torch-*.whl \
>     /workspace/wheels/torchvision-*.whl \
>     /workspace/wheels/torchaudio-*.whl
> ```
>
> `aria2c -x 16 -s 16` opens 16 parallel byte-range connections per file;
> typical speedup is 10–20x over a single-connection download. If `apt-get`
> is unavailable on the box, a statically-linked `aria2c` binary can be
> dropped into `/workspace/bin/` from the aria2 GitHub releases page.

</details>

#### 3b. Install `unsloth`

The experiment scripts prefer **unsloth** for model loading (auto dtype
detection, tokenizer fixes, faster inference).  Install it before
`uv sync`:

```bash
uv pip install unsloth
```

> If `unsloth` cannot be installed (e.g. older CUDA driver), the scripts
> fall back to plain `transformers` automatically — no code changes needed.

#### 3c. Install everything else with `uv sync`

`uv sync` will now read `pyproject.toml`, see that `torch` is already
satisfied by the cu124 wheel you just installed, and only pull in
`transformers`, `accelerate`, `safetensors`, `sentencepiece`, `protobuf`,
`wordfreq`, `huggingface-hub`, `numpy`, and `tqdm`:

```bash
uv sync
```

> **If `uv sync` tries to replace your Torch with a CPU build** (rare, but
> it can happen when resolver preferences disagree), use this instead:
>
> ```bash
> uv sync --no-install-package torch
> ```
>
> That tells `uv` to keep the Torch it already has and only install the
> remaining dependencies.

### 4. Download the model weights

Use the included download script to pull the unsloth Mistral weights from
HuggingFace Hub:

```bash
python -m scripts.download_model \
    --repo-id unsloth/Mistral-Small-24B-Instruct-2501 \
    --output-dir $MODEL_PATH
```

Verify the download:

```bash
ls $MODEL_PATH   # should list config.json, tokenizer files, *.safetensors, ...
```

> If you already have the weights elsewhere (e.g. `/root/models`), symlink
> instead:
>
> ```bash
> ln -s /root/models/unsloth_Mistral_Small_24B_Instruct_2501 $MODEL_PATH
> ```

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

> **Custom lambda values.** The default sweep is
> `0 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2 3 4 5 6 8`.
> Override with `--lambdas`, e.g. `--lambdas 0 0.5 1 1.5 2` for a
> quick coarse sweep, or `--lambdas 1.8 1.9 2.0 2.1 2.2` to zoom into
> a narrow range around the soft spot.

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

# 2) generate steered rollouts across a lambda sweep (uses default lambdas)
nohup python -m scripts.apply_steering \
    --model-path "$MODEL_PATH" \
    --english-file /workspace/IP_Internals/data/gsm8k.jsonl \
    --steering-vector /workspace/IP_Internals/outputs/steering_vectors/spanish_layer25.pt \
    --num-questions 50 \
    --question-seed 42 \
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

# 2) generate steered rollouts across a lambda sweep (uses default lambdas)
nohup python -m scripts.apply_steering \
    --model-path "$MODEL_PATH" \
    --english-file /workspace/IP_Internals/data/gsm8k.jsonl \
    --steering-vector /workspace/IP_Internals/outputs/steering_vectors/french_layer25.pt \
    --num-questions 50 \
    --question-seed 42 \
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

## Finetuning pipeline (CAFT-style)

> **Prerequisites.** You need a Spanish steering vector
> (`outputs/steering_vectors/spanish_layer25.pt`) from the steering
> pipeline above.  Install extra deps: `uv sync` (datasets, trl, peft
> are now in `pyproject.toml`).

### Step 0 — Prepare data

```bash
cd /workspace/IP_Internals
source /workspace/env.sh && source .venv/bin/activate

python -m scripts.prepare_finetune_data \
    --exclude-meta outputs/steering_vectors/spanish_layer25.meta.json \
                   outputs/steering_vectors/french_layer25.meta.json
```

This creates `data/finetune_train.jsonl` (~7 400 examples, half Spanish
half French) and downloads the GSM8K test split to `data/gsm8k_test.jsonl`
(1 319 questions).

### Step 1a — Baseline finetune

```bash
nohup python -m scripts.finetune \
    --model-path "$MODEL_PATH" \
    --train-file data/finetune_train.jsonl \
    --output-dir outputs/checkpoints/baseline \
    > outputs/logs/finetune_baseline.out 2>&1 &
disown
```

### Step 1b — Steered finetune (Spanish vector injection, lambda=1)

```bash
nohup python -m scripts.finetune \
    --model-path "$MODEL_PATH" \
    --train-file data/finetune_train.jsonl \
    --steering-vector outputs/steering_vectors/spanish_layer25.pt \
    --steering-lambda 1.0 \
    --output-dir outputs/checkpoints/steered_spanish \
    > outputs/logs/finetune_steered.out 2>&1 &
disown
```

> **Tuning knobs.** `--lora-r 32 --lora-alpha 64` (default), `--batch-size 2`,
> `--grad-accum-steps 8` (effective batch 16), `--num-epochs 1`,
> `--learning-rate 1e-5`.  Override any of these on the command line.
> To inject at a different layer: `--steering-layer 20`.

### Step 2 — Generate test rollouts

Run after **each** finetune finishes:

```bash
# Baseline rollouts
nohup python -m scripts.generate_test_rollouts \
    --model-path "$MODEL_PATH" \
    --adapter-path outputs/checkpoints/baseline \
    --test-file data/gsm8k_test.jsonl \
    --num-questions 100 \
    --output outputs/rollouts/baseline_test_rollouts.jsonl \
    > outputs/logs/baseline_test_rollouts.out 2>&1 &
disown

# Steered rollouts
nohup python -m scripts.generate_test_rollouts \
    --model-path "$MODEL_PATH" \
    --adapter-path outputs/checkpoints/steered_spanish \
    --test-file data/gsm8k_test.jsonl \
    --num-questions 100 \
    --output outputs/rollouts/steered_test_rollouts.jsonl \
    > outputs/logs/steered_test_rollouts.out 2>&1 &
disown
```

### Step 3 — Evaluate language mix

```bash
# Baseline
python -m scripts.judge_multilingual \
    --rollouts outputs/rollouts/baseline_test_rollouts.jsonl \
    --output-jsonl outputs/judgements/baseline_multilingual.jsonl \
    --output-summary outputs/judgements/baseline_multilingual_summary.json

# Steered
python -m scripts.judge_multilingual \
    --rollouts outputs/rollouts/steered_test_rollouts.jsonl \
    --output-jsonl outputs/judgements/steered_multilingual.jsonl \
    --output-summary outputs/judgements/steered_multilingual_summary.json
```

Compare the two `*_summary.json` files to see if steering injection during
training shifted the English / Spanish / French distribution relative to the
baseline.

## Notes

- **Model loading.** Both `generate_steering_vector.py` and `apply_steering.py`
  try to import `unsloth.FastLanguageModel` first.  If unsloth is installed the
  model is loaded with `dtype=None` (auto-detect: bfloat16 on Ampere+) and
  `load_in_4bit=False` (full precision, no quantization).  If unsloth is *not*
  installed the scripts fall back to
  `AutoModelForCausalLM.from_pretrained(..., torch_dtype="auto",
  device_map="auto")`.
- Everything that needs to persist lives under `/workspace`: the repo clone,
  the `uv` binary (`/workspace/bin`), the uv cache (`/workspace/.cache/uv`),
  the HuggingFace cache (`/workspace/.cache/huggingface`), the venv
  (`/workspace/IP_Internals/.venv`), the model weights
  (`/workspace/models/...`), and all outputs
  (`/workspace/IP_Internals/outputs/...`). After a fresh SSH session, a
  `source /workspace/env.sh && cd /workspace/IP_Internals && source
  .venv/bin/activate` is all you need.
- All decoding defaults to greedy (`--temperature 0`). If you want sampled
  rollouts, pass `--temperature 0.7 --top-p 0.95`.
- `generate_steering_vector.py` writes a sidecar `*.meta.json` listing the 50
  questions it used so the rollout script can guarantee a disjoint test set.
- The judge is intentionally simple: per-word `wordfreq` lookups with a
  configurable `--ratio-margin` (default 2.0) and `--label-threshold` (default
  0.80). It is not perfect but is enough to find the soft-spot lambda.
