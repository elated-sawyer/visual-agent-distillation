# Training and data contracts

The public boundary starts **after trajectory selection**. `prepare_data.py --episodes selected.json` reads every path supplied by the caller, without ranking trajectories or filtering by success. It rejects incomplete/infrastructure-error episodes and evaluation-split episodes. Selection labels never become model input.

A prefix ends at observation t: system, observation 0, original response 0, …, observation t. The response at t, future observations, rewards and final outcome are absent. Original earlier thinking and original image bytes are preserved. The prefix validator compares native token IDs, image pixels and grid shapes, and rejects excessive length rather than silently truncating the pool.

For offline OPD, sample a trajectory uniformly, then a turn uniformly. The student samples a new response y from its current policy. The frozen teacher scores exactly these response token IDs on the same multimodal prefix. The native Slime token advantage is:

```text
A_t = log p_teacher(y_t | prefix, y_<t) − log p_student_old(y_t | prefix, y_<t)
```

The coefficient is 1; there is no environment reward or additional reference KL. `mm_opd_checks.py` checks the signal against that difference. Native Slime performs the policy update; this release does not replace its optimizer/loss with a lookalike implementation. The tokenizer compatibility assumptions are specific to the released Qwen3.5 student/teacher pairing.

Matched SFT uses the same ordered prefix draws. A frozen teacher generates a response for each draw, using sampling seed `20261022 + draw_index`; the full response, including thinking and length-limited completions, is supervised. Prompt tokens have zero supervised weight. SFT uses native `sft_loss`, with advantage computation disabled. The saved collection response is **not** the SFT target.

An online OPD sample is an actually executed response, not a completed episode. Sixteen environment lanes retain their causal histories. Slime's native background worker overlaps generation and training, with at most one batch of unconsumed credit. Sampling-policy lag is checked to be at most one update; native truncated importance sampling uses bounds [0, 2]. Histories can cross policy versions. No outcome-based turn filtering is applied.

## Input schemas

- `mm-episode-v1`: `system`, `task`, `source`, `protocol_sha256`, `status`, and ordered `turns` with `observation`, exact `response`, and `next_observation`.
- Each text/image message is explicitly typed. Images use content-addressed `image_ref` records in stored episodes and absolute image paths in native training JSONL.
- `selected.json`: JSON array of episode paths relative to the manifest, supplied by the user. This list is the selection-module interface.
- `pool.jsonl`: every causal prefix of the selected episodes. `prefixes.jsonl`: the precommitted draw order.
- `config.json`: model path, recipe and budgets. `input-audit.json`: native processor checks, pool and draw SHA256.

The compact walkthrough uses anonymized IDs. Its prefix text/images are original, but its display JSON is not the internal episode schema. `--demo` converts that representation without interpreting or rewriting responses.
