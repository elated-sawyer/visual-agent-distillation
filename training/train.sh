#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help || $# -lt 2 ]]; then
  echo 'Usage: bash training/train.sh opd|sft|online RUN_DIR [extra native Slime arguments]'
  echo 'Required: SLIME_ROOT, STUDENT_MODEL, RAY_ADDRESS; OPD: TEACHER_URL'
  exit 0
fi
method=$1; shift
export MM_OPD_RUN; MM_OPD_RUN=$(realpath "$1"); shift
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MM_OPD_SOURCE="$repo/training"
: "${SLIME_ROOT:?Set the compatible Slime checkout}" "${STUDENT_MODEL:?Set the local HF model}" "${RAY_ADDRESS:?Start a Ray cluster first}"
export PYTHONPATH="$repo:$repo/training:$SLIME_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
mapfile -t cfg < <(python3 - "$MM_OPD_RUN" <<'PY'
import json,sys
from pathlib import Path
c=json.loads((Path(sys.argv[1])/'config.json').read_text())
for k in ['updates','batch_size','context_limit','response_limit','learning_rate','seed','evaluation_context_limit']:print(c[k])
PY
)
[[ ${#cfg[@]} == 7 ]]
source "${MODEL_ARGS_FILE:-$SLIME_ROOT/scripts/models/qwen3.5-4B.sh}"
MODEL_ARGS[1]=mm_native_provider
MODEL_ARGS[2]=get_qwen3_5_vl_model_provider
common=(--actor-num-nodes 1 --actor-num-gpus-per-node "${ACTOR_GPUS:-8}"
 "${MODEL_ARGS[@]}" --hf-checkpoint "$STUDENT_MODEL" --load "$STUDENT_MODEL"
 --prompt-data "$MM_OPD_RUN/prefixes.jsonl" --input-key messages --apply-chat-template
 --apply-chat-template-kwargs '{"enable_thinking":true}'
 --num-rollout "${cfg[0]}" --rollout-batch-size "${cfg[1]}" --n-samples-per-prompt 1 --global-batch-size "${cfg[1]}"
 --rollout-max-context-len "${cfg[2]}" --rollout-max-response-len "${cfg[3]}"
 --rollout-temperature 1 --rollout-top-p 1 --rollout-top-k -1 --rollout-skip-special-tokens
 --balance-data --seed "${cfg[5]}" --advantage-estimator grpo --kl-coef 0 --kl-loss-coef 0 --entropy-coef 0
 --optimizer adam --lr "${cfg[4]}" --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
 --tensor-model-parallel-size "${TRAIN_TP:-4}" --sequence-parallel --pipeline-model-parallel-size 1
 --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
 --micro-batch-size 1 --log-probs-chunk-size 1024 --attention-dropout 0 --hidden-dropout 0
 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
 --use-wandb --wandb-mode offline --wandb-dir "$MM_OPD_RUN" --wandb-project "${WANDB_PROJECT:-mosaic-distill}"
 --wandb-group "$(basename "$MM_OPD_RUN")" --disable-wandb-random-suffix)
save=(--save "$MM_OPD_RUN/checkpoint" --save-hf "$MM_OPD_RUN/final-hf" --save-interval "${cfg[0]}" --no-save-optim --no-save-rng)
if [[ $method == sft ]]; then
 extra=(--rollout-num-gpus 0 --debug-train-only --rollout-function-path mm_teacher_sft.generate_rollout
 --loss-type sft_loss --disable-compute-advantages-and-returns --rollout-data-postprocess-path mm_teacher_sft.audit_batch)
elif [[ $method == opd || $method == online ]]; then
 : "${TEACHER_URL:?Set exact-token teacher /generate endpoint}"
 extra=(--rollout-num-gpus "${ROLLOUT_GPUS:-4}" --rollout-num-gpus-per-engine 1
 --custom-rm-path mm_opd_checks.reward_func --reward-key opd_task_reward --custom-reward-post-process-path mm_opd_checks.post_process_rewards
 --rm-url "$TEACHER_URL" --rollout-data-postprocess-path mm_opd_checks.audit_signal --use-opd --opd-type sglang --opd-kl-coef 1
 --sglang-mem-fraction-static 0.68 --sglang-context-length "$((${cfg[6]}+8))" --sglang-max-running-requests 32
 --sglang-cuda-graph-max-bs 32 --sglang-server-concurrency 16 --sglang-chunked-prefill-size 8192
 --sglang-router-port "${ROUTER_PORT:-27500}" --sglang-served-model-name mm-student --check-weight-update-equal)
 if [[ $method == online ]]; then
  export MOSAIC_ONLINE=1
  save=()
  extra+=(--data-source-path native_async_turns.TurnDataSource --rollout-function-path native_async_turns.generate_rollout
   --custom-generate-function-path native_async_turns.generate_turn --custom-rm-path native_async_turns.score_turn
   --rollout-sample-filter-path native_async_turns.accept_batch --rollout-data-transport object-store --use-tis --tis-clip 2 --tis-clip-low 0)
 fi
else echo "Unknown method: $method" >&2; exit 2; fi
if [[ -f "$MM_OPD_RUN/eval-dataset.jsonl" ]]; then
 interval=$(python3 -c 'import json,os;from pathlib import Path;print(json.loads((Path(os.environ["MM_OPD_RUN"])/"config.json").read_text())["evaluation_interval"])')
 if [[ $method == sft ]]; then
  extra=(--rollout-num-gpus "${ROLLOUT_GPUS:-8}" --rollout-num-gpus-per-engine 1
   --rollout-function-path mm_teacher_sft.generate_rollout --loss-type sft_loss --disable-compute-advantages-and-returns
   --rollout-data-postprocess-path mm_teacher_sft.audit_batch --sglang-context-length "$((${cfg[6]}+8))"
   --sglang-mem-fraction-static 0.68 --sglang-max-running-requests 32 --sglang-cuda-graph-max-bs 32
   --sglang-server-concurrency 16 --sglang-chunked-prefill-size 8192 --sglang-router-port "${ROUTER_PORT:-27500}"
   --sglang-served-model-name mm-student --check-weight-update-equal)
 fi
 eval_hook=live_eval.generate
 [[ $method != online ]] || eval_hook=native_async_turns.evaluate
 extra+=(--eval-interval "$interval" --eval-function-path "$eval_hook" --eval-prompt-data live "$MM_OPD_RUN/eval-dataset.jsonl" --skip-eval-before-train)
elif [[ $method == online ]]; then
 echo 'Online runs require live evaluation panels: prepare_evaluation.py first (no online checkpoint is saved).' >&2; exit 2
fi
exec python3 -B "$repo/training/entry.py" "${common[@]}" "${extra[@]}" "${save[@]}" "$@"
