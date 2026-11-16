# Collection and evaluation

The adapters use the pinned VAGEN environment implementations, with the same observations, actions and scoring as the experiments. Fetch the exact upstream sources first:

```bash
python scripts/fetch_upstreams.py
pip install -r requirements-env.txt
```

Install environment packages in a separate environment on a CPU task or suitable simulation host. Models may be served remotely. No source checkout or environment command is run by merely importing the lightweight data package.

| Environment | Protocol | Metric |
|---|---|---|
| Sokoban | 6×6, one box, 5 outer turns, up to 3 actions per turn, wm_think | Binary task success |
| Navigation | AI2-THOR, 30 turns, up to 5 actions per turn, free_think | Binary task success |
| SpatialGym | 10 exploration steps plus final map, full history | Native continuous cognitive-map score |

`configs/environments/` contains the actual formal settings, including the versioned SpatialGym initial-image clarification. `configs/panels/` contains frozen final task/repeat/seed lists, not newly randomized replacements. Sokoban final: 64 maps × 4; Navigation: 60 standard and 60 long-horizon tasks × 4 each; SpatialGym: 20 single-room and 20 double-room scenes × 4. Report the 16 training and 4 held-out single rooms separately. Those four held-out rooms are a development split; the double-room panel is the cross-scene test.

Monitoring uses 16 Navigation tasks drawn from the 60 standard final tasks; final evaluations execute separately. SpatialGym monitors the same four held-out single rooms. These original panels were inspected during development. The additional independent Navigation panel is recorded separately in the released result snapshot; its task manifest is not bundled.

## Sokoban example

```bash
export MOSAIC_RENDER_HOST=your-approved-simulation-hostname
python scripts/run_with_render_guard.py --run-dir outputs/sokoban-render -- \
  python scripts/collect.py --protocol configs/environments/sokoban.json \
  --panel configs/panels/sokoban.json --endpoints http://STUDENT_HOST:30000 \
  --served-model mm-student --context 196608 --workers 32 --output outputs/eval
python scripts/summarize.py outputs/eval/results.json
```

The retained Sokoban adapter also uses the guard; its board rendering itself is CPU-based.

The model service must support the SGLang chat options used here, including `separate_reasoning=false`, enabled thinking, and sampling seeds. Complete original responses are retained. GPU/model generation and infrastructure failures are not counted as zero rewards. The collector writes original episodes and content-addressed images; for training collection, supply a **train-only** slot list and pass its selected episode paths to `prepare_data.py`.

## Navigation deployment

Install AI2-THOR 5.0.0 and deploy its `f0825767cd50d69f666c7f282e54abfe58f1e917` CloudRendering build under `vendor/navigation/ai2thor/releases/`. Install the corresponding headless/Vulkan system libraries on a suitable rendering machine; the model-training container alone does not supply a renderer. No binary simulator is redistributed here.

The renderer must be a descendant of the supplied fail-closed supervisor. Explicitly set `MOSAIC_RENDER_HOST` to the permitted simulation host, then wrap collection/evaluation:

```bash
export MOSAIC_RENDER_HOST=your-approved-render-hostname
python scripts/run_with_render_guard.py --run-dir outputs/render-session -- \
  python scripts/collect.py --protocol configs/environments/navigation.json \
    --panel configs/panels/navigation.json --endpoints http://STUDENT_HOST:30000 \
    --context 262136 --output outputs/navigation-eval
```

The guard monitors Xid109, kernel-log continuity, descendant ownership and heartbeat, and leaves a persistent latch on failure. It requires readable kernel logs and Linux process supervision. It has no default permitted hostname and does not clear a latch automatically. This portable copy does not alter any guard on the authors' running cluster.

## SpatialGym deployment

Obtain `yw12356/spatial_gym_dataset` revision `5299760c9d153ba2b5956527f38e730ea92f9165` and place its `1-room`/`2-room` directories under `data/spatial_gym/<revision>/`. Install VAGEN's SpatialGym dependencies from the pinned checkout. The committed protocol includes per-file hashes and room assignments; don't silently substitute another dataset revision. Asset licenses remain those of the upstream dataset.

Use `spatial-single.json` / `spatial-two-room.json` protocols and panels with `--context 262136`. The summary command retains continuous scores; it never thresholds SpatialGym into binary success. Frozen full histories and the 32K response limit are shared across teacher, base student and trained student evaluation.
