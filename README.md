# MosaicDistill

**Multimodal Agent Distillation from a Few Representative Trajectories**

Companion code for **Distilling Visual Agents from a Few Representative Trajectories**.

This repository provides visual-environment adapters, trajectory collection, causal prefix replay, native Slime OPD and teacher-response SFT, the online OPD adapter, evaluation, and aggregate results. **The trajectory-selection front end is not included.** Supply an already selected trajectory manifest, or start with the four released Sokoban examples.

```text
Already selected trajectories → original visual prefixes
                              ├─ fresh student responses → teacher token scores → OPD
                              └─ frozen teacher responses → response-only SFT
Current student ↔ environment → executed turns → teacher token scores → online OPD
```

Skill annotations are selection metadata. They are not injected into student or teacher prompts. Offline OPD generates a new response each time a prefix is replayed; it does not train on the historical response as a target. Failed trajectories are retained. Thinking remains enabled.

## Start here

| Goal | Entry |
|---|---|
| Read the original compact demo | [Four trajectories and ten prefixes](docs/WALKTHROUGH.md) |
| Understand inputs and losses | [Method and data contracts](docs/METHOD.md) |
| Prepare replay data and run OPD/SFT | [Training guide](docs/TRAINING.md) |
| Collect trajectories or evaluate | [Environment guide](docs/ENVIRONMENTS.md) |
| Use the online baseline | [Online OPD guide](docs/ONLINE_OPD.md) |
| Inspect results and curves | [Results](results/README.md) |
| Check what was tested | [Release validation](docs/VALIDATION.md) |

## Small CPU-only example

Use a CPU task or a development machine with adequate resources:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,plots]'
python scripts/prepare_data.py --demo --recipe configs/sokoban.json \
  --student /path/to/Qwen3.5-4B --output outputs/demo-opd
python -m pytest -q tests
python scripts/plot_results.py
```

The data command needs no model loading, network access, or environment interaction. It writes 10 unique visual prefixes and 6,400 deterministic draws from four trajectories (100 updates × batch 64). The model path is used later by the GPU preflight. This does **not** claim that training has run.

## Released code

```text
mosaic_distill/   Original-image storage, causal records, limits, three environments
training/        Native Slime hooks, exact-token teacher scoring, SFT, online adapter
scripts/         Data preparation, collection, live evaluation, summaries and plotting
configs/         Formal recipes, model revisions, frozen environment/evaluation panels
tests/           CPU contracts and release checks
integration_tests/  Optional tests requiring the compatible training container
data/, images/   Compact, precomputed Sokoban examples
results/         100-update aggregate results and measured learning curves
```

The model/runtime-dependent training hooks come from tested experiments. Portable launchers and packaging are validated separately; see the validation report for the actual scope. This is not a claim of a fresh end-to-end GPU reproduction on arbitrary hardware. Slime, Megatron, SGLang, model weights, and simulator assets are external dependencies.

The source release excludes skill-author prompts, visual decision-description generation, feature/similarity construction, coverage/representativeness scoring, and the selector. Precomputed example annotations and choices remain viewable. There is no website, HTML application, checkpoint, private cluster launcher, or credential in this repository.

See [third-party notices](THIRD_PARTY.md). A general license for the authors' source code has not yet been assigned; third-party licenses remain in effect.
