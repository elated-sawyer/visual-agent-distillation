# Sokoban trajectory walkthrough

Examples accompanying **Distilling Visual Agents from Limited Interaction Experience**.

Explore the four selected trajectories, their screenshots, decision annotations, and causal prefixes. See the [README](../README.md) to prepare these records for training.

## Walkthrough

1. Start with a pool of recorded visual interactions. This example has **128 Sokoban candidate trajectories**.
2. Inspect the cached skill/context annotations and visual decision descriptions used for selection.
3. Follow the **four recorded representative choices** below.
4. Inspect the **10 fixed prefixes** extracted from those trajectories. Each prefix ends at the current observation, before its response.

In the training method, a student generates a new response at a sampled prefix, and a frozen teacher supplies token-level guidance. The linked files below show the saved inputs used by this process.

## Four representative trajectories

These are the four recorded histories used in the paper's Sokoban K4 setting. Follow a trajectory link to see its observations, extracted actions, cached decision descriptions, and prefix links.

| Recorded rank | Trajectory | Initial observation | Saved prefixes | Newly covered skill–context pairs | Recorded representativeness gain |
|---:|---|:---:|---:|---:|---:|
| 1 | [C023](../examples/C023.md) | ![C023](../images/board-023.png) | 5 | 6 | 9.184 |
| 2 | [C073](../examples/C073.md) | ![C073](../images/board-073.png) | 2 | 1 | 2.646 |
| 3 | [C016](../examples/C016.md) | ![C016](../images/board-016.png) | 1 | 0 | 3.345 |
| 4 | [C114](../examples/C114.md) | ![C114](../images/board-114.png) | 2 | 0 | 1.892 |

The histories preserve the original Qwen3.5-4B responses, actions, and observations. Selection gains measure the coverage and representativeness objective. Full-precision records are in [selection.json](../data/selection.json).

## What a fixed prefix contains

A prefix contains the original system message, earlier observations and responses, and the current observation. Earlier reasoning is retained. The current response and future observations are excluded. See [P01](../data/prefixes/P01.json) for a first-turn input and [P02](../data/prefixes/P02.json) for an input with interaction history.

Skill labels and cached decision descriptions are used for selection only. They are **not inserted into either the student or distillation-teacher prefix**. Training samples a selected trajectory uniformly, then samples one of its turns uniformly to generate a fresh response.

## Contents

| File or directory | Contents |
|---|---|
| [examples/](../examples/) | Four readable trajectory walkthroughs |
| [data/candidates.json](../data/candidates.json) | Summaries and cached annotation previews for all 128 candidates |
| [data/selection.json](../data/selection.json) | Four precomputed choices and recorded gains |
| [data/vocabulary.json](../data/vocabulary.json) | Saved skill and context vocabulary |
| [data/trajectories/](../data/trajectories/) | Complete stored responses and observations for the four selected trajectories |
| [data/prefixes/](../data/prefixes/) | Ten saved model inputs |
| [images/](../images/) | Original observation images referenced by the selected examples |
| [THIRD_PARTY.md](../THIRD_PARTY.md) | Environment and asset acknowledgments |

Image paths in JSON are relative to the repository root. Displayed turn numbers start at 1; JSON `turn_index` starts at 0. Candidate IDs are aliases. The four selected trajectories include their original images and full responses; the remaining candidates have annotation previews.

Continue with the [data preparation and training guide](../README.md#quick-start).
