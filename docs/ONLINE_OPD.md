# Online OPD baseline

## Task budgets

The completed 4B study uses **1, 4, 16, or all training tasks**. Full means 120 Navigation tasks, 128 Sokoban maps, or 16 SpatialGym rooms. SpatialGym K16 and full-task results refer to the same run. The 9B study currently includes full-task online OPD, not the smaller online task budgets.

At K4, the online catalogue contains the exact tasks underlying the offline K4 trajectories. The online agent executes fresh responses and receives new observations. Offline replay keeps the original observations fixed. Both use 100 updates × 64 responses, but their generated-token counts and compute costs can differ. See [results](../results/README.md).

The number of environment lanes remains 16 even when the catalogue contains fewer than 16 tasks; lanes can run different episodes of the same task. Supply the desired task subset through the catalogue below. This does not require changing the OPD loss or the per-update response budget.

## Runtime

The published adapter is the experimental **native-background-worker, turn-level** implementation: 16 persistent environment lanes, 64 executed responses per update, at most one batch of credit ahead, lag in [0,1], and native importance correction clipped to [0,2]. It is not the obsolete implementation that completes whole episodes and discards most turns.

The core is `training/native_async_turns.py`; generation, scoring, stale-policy checks and evaluation draining are retained from the validated run. Each lane has a single active decision. The environment executes the sampled response before it is admitted to training. A stop due to context exhaustion does not create a synthetic supervised response. There is no environment-reward filter.

This is an advanced distributed entrypoint. First configure the offline OPD runtime, an input-audited bootstrap run, and live evaluation as in [TRAINING.md](TRAINING.md). Supply `online-catalogue.json` containing:

```text
environment: sokoban | navigation | spatial
student: local HF model path
protocol: complete environment protocol
protocol_sha256: canonical SHA256 from mosaic_distill.records.fingerprint(protocol)
tasks: unique training task records, including original reset seed and task_id
```

Build the catalogue from collected training episodes, not final evaluation slots. It is an input contract, not an implementation of skill-based task selection. All nodes must see the same filesystem paths. Navigation's renderer must be deployed and guarded; the persistent supervisor is also guarded in the released baseline.

```bash
python scripts/prepare_online.py --run outputs/online --catalogue /data/online-catalogue.json \
  --render-hostname your-approved-render-hostname
# Attach live panels with prepare_evaluation.py before launch.
export PYTHONPATH="$PWD:$PWD/training:$SLIME_ROOT:$PYTHONPATH"
export MOSAIC_RENDER_HOST=your-approved-render-hostname
python scripts/run_with_render_guard.py --run-dir outputs/online-render -- \
  python training/turn_stream_supervisor.py outputs/online
# On the Ray head, after the supervisor is ready:
bash training/train.sh online outputs/online
```

`outputs/online` must first be prepared and natively input-audited like an offline run; its frozen prefixes serve only the startup preflight. Training responses thereafter come from environment lanes. Start an independent fresh run; resuming the persistent environment stream is not implemented. Evaluation pauses/drains the native worker, checks that all 16 lanes are idle, and evaluates the current weights. The online launcher saves no weight checkpoints and requires live final evaluation.

The release is a portable extraction, not the authors' cluster controller. CPU tests validate contracts; an optional `integration_tests/test_native_async.py` exercises the actual compatible Slime worker with fake environment/model I/O. A fresh full GPU run of the extracted package is not claimed. The authors' original completed online experiments used these core hooks.
