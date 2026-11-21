# Offline OPD and matched teacher-response SFT

## Runtime

Use a compatible **Slime 0.4.0 container** with its Megatron, SGLang, Torch, and CUDA packages. This project uses native Megatron language layers plus the Transformers vision module, TP4/CP1, micro-batch 1, full activation recomputation and exact multimodal teacher scoring. `training/mm_native_provider.py` contains the CP1 indexing correction; `qwen_exact_ids.py` preserves exact input IDs in the teacher's Qwen processor. Use the pinned runtime APIs for these integration hooks.

The formal recipes use Qwen3.5-4B or 9B students and Qwen3.5-27B teacher; exact revisions are in `configs/models.json`. Student/teacher tokenizers and visual preprocessing must match. The learning rate is 1e-6, Adam betas 0.9/0.98, weight decay 0.1, 100 updates × batch 64, response cap 32,768, temperature/top-p 1 and top-k −1. The three context limits differ; use the corresponding recipe.

Create a Ray cluster on your allocated training/inference nodes before launch. The default launcher requests 8 actor GPUs and 4 rollout GPUs; a separately launched TP2 teacher needs 2 more GPUs. Ray manages placement across the training and inference nodes. Reserve disjoint resources for an externally launched teacher rather than letting Ray place an actor on it. All nodes must see the same code, model, data and output paths.

```bash
export SLIME_ROOT=/path/to/slime
export STUDENT_MODEL=/models/Qwen3.5-4B
export TEACHER_MODEL=/models/Qwen3.5-27B
export PYTHONPATH="$PWD:$PWD/training:$SLIME_ROOT:$PYTHONPATH"
export RAY_ADDRESS=auto
export WANDB_PROJECT=mosaic-distill
```

For 9B, additionally set `MODEL_ARGS_FILE=$PWD/training/models/qwen3.5-9B.sh`. The default 4B model arguments come from the Slime checkout. Set GPU allocation and memory budgets for the selected model and context length.

## OPD

1. Supply already selected trajectories. Alternatively, use `--demo` for the bundled Sokoban examples.

```bash
python scripts/prepare_data.py --episodes /data/selected.json --assets /data/assets \
  --recipe configs/sokoban.json --student "$STUDENT_MODEL" --output outputs/opd
python training/validate_inputs.py outputs/opd
```

2. On reserved teacher GPUs, start the native SGLang service through the exact-ID entrypoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 python training/launch_exact_mm_teacher.py \
  --model-path "$TEACHER_MODEL" --tp-size 2 --host 0.0.0.0 --port 30004 \
  --context-length 114696 --mem-fraction-static 0.78 --max-running-requests 32
```

Use the recipe's training context + 8 for the teacher; 114696 above is Sokoban. Keep automatic prompt truncation disabled and expose the teacher endpoint to the training nodes.

3. On the Ray head:

```bash
export TEACHER_URL=http://TEACHER_HOST:30004/generate
bash training/train.sh opd outputs/opd
```

The launcher records W&B data offline and saves the final native checkpoint and HF export. Live monitoring evaluates the current weights every five updates.

## Matched SFT

Use the unchanged OPD input directory. A teacher is needed to generate labels, but not to train SFT.

```bash
python scripts/prepare_sft.py request --opd-run outputs/opd --labels outputs/labels \
  --teacher "$TEACHER_MODEL" --teacher-urls http://TEACHER_HOST:30004
python training/collect_teacher_labels.py outputs/labels
python scripts/prepare_sft.py attach --opd-run outputs/opd --labels outputs/labels --output outputs/sft
python training/validate_inputs.py outputs/sft
bash training/train.sh sft outputs/sft
```

Use a fresh label directory. The teacher generates one response per draw without outcome filtering. Training checks label hashes and matching teacher/student prefix identities. Default SFT has 8 actor GPUs and no inference engines; adding live evaluation starts inference engines as described next.

## Live monitoring every five updates

`prepare_evaluation.py` attaches a frozen panel schedule. Each descriptor has `name`, `scope` (`monitor` or `final`), `environment`, and paths to the environment protocol and slot list. Paths are relative to the descriptor file. `configs/evaluations/*.json` provide the paper panels. Final evaluation uses the fixed update-100 weights. Navigation's 16 monitoring tasks are a subset of its 60 standard final tasks, and SpatialGym reuses its four held-out rooms for monitoring.

```bash
python scripts/prepare_evaluation.py --run outputs/opd \
  --panels configs/evaluations/sokoban.json --inference-ip INFERENCE_IP --rollout-gpus 4
# Separate environment process; set MOSAIC_RENDER_HOST on its permitted host.
python scripts/run_with_render_guard.py --run-dir outputs/eval-render -- \
  python scripts/evaluation_worker.py outputs/opd
bash training/train.sh opd outputs/opd          # separate training process
```

For SFT, attach the same schedule to `outputs/sft`, use `--rollout-gpus 8` and `ROLLOUT_GPUS=8`. Both commands require the compatible environment runtime. Navigation workers must run under the rendering guard. The evaluator checks exact task/repeat identities, weight version before/after evaluation, and completion. A failure stops evaluation rather than treating missing episodes as unsuccessful outcomes.

The launchers use the Ray cluster and environment services configured above.
