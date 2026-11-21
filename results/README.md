# Experimental results

All tables use **100 updates × 64 responses**, fixed update-100 evaluation, native thinking, and a frozen Qwen3.5-27B teacher. The [README main table](../README.md#results) presents the symmetric 4B/9B comparison. The files below contain the paper's result tables and learning curves.

## Result files

| File | Contents |
|---|---|
| [main.json](main.json) | Existing full-precision 4B core table; retained for plotting compatibility |
| [paper_results_4b.json](paper_results_4b.json) | 4B panel scores/counts and original collaborator-reported selection/data-budget controls |
| [student_scale_results.json](student_scale_results.json) | 9B original, K4 offline OPD/SFT, and full-task online OPD |
| [online_task_budget_results.json](online_task_budget_results.json) | 4B K1/K4/K16/full-task online results, including task matching and run reuse |
| [data_budget.csv](data_budget.csv) | Plot-ready offline/online data-volume results with explicit units and source types |
| [collaborator_followup_results.json](collaborator_followup_results.json) | All four supplementary training repetitions, independent Navigation panel, and additional ablations |
| [learning_curves.csv](learning_curves.csv) | 300 measured points from 15 core 4B training runs |
| [student_scale_curves.csv](student_scale_curves.csv) | 180 measured points from nine 9B training runs |

JSON success rates use fractions in [0,1]; `data_budget.csv` explicitly uses success percentages for Navigation/Sokoban and mapping scores for SpatialGym.

## Task breadth and fixed-history budgets: 4B

Offline K counts **fixed trajectories**; online K counts **tasks with fresh interactions**. The full offline pools contain 120/128/32 Navigation/Sokoban/SpatialGym trajectories; full online catalogues contain 120/128/16 tasks. K4 online uses the same task identities as K4 replay. Equal updates and responses do not imply equal tokens or compute.

| Evaluation panel | Training | K1 | K4 | K16 | Full |
|---|---|---:|---:|---:|---:|
| Navigation · standard | Offline OPD | 73.75% | 82.08% | 81.67% | 77.50% |
| | Offline SFT | 70.42% | 74.17% | 75.42% | 64.17% |
| | Online OPD | 80.42% | 82.50% | 76.25% | 78.75% |
| Navigation · long horizon | Offline OPD | 19.17% | 22.08% | 23.75% | 22.50% |
| | Offline SFT | 17.50% | 20.42% | 21.25% | 15.42% |
| | Online OPD | 14.17% | 22.50% | 20.42% | 21.25% |
| Sokoban · 64 maps | Offline OPD | 80.47% | 85.55% | 83.98% | 83.59% |
| | Offline SFT | 75.00% | 76.17% | 77.34% | 77.73% |
| | Online OPD | 80.47% | 84.38% | 85.16% | 85.94% |
| SpatialGym · held-out rooms | Offline OPD | 0.36 | 0.4112 | 0.46 | 0.4664 |
| | Offline SFT | 0.33 | 0.3996 | 0.42 | 0.4243 |
| | Online OPD | 0.4057 | 0.4032 | 0.4763† | 0.4763 |
| SpatialGym · two rooms | Offline OPD | 0.21 | 0.2700 | 0.31 | 0.3253 |
| | Offline SFT | 0.18 | 0.2413 | 0.27 | 0.2834 |
| | Online OPD | 0.2878 | 0.2998 | 0.3051† | 0.3051 |

† SpatialGym K16 and full-task online OPD refer to the same 16-room run. Offline K1/K16 values are collaborator-reported, with SpatialGym values shown at the supplied precision.

## Independent Navigation panel

This additional collaborator-reported panel is separate from the original main-table evaluation. Each model has 240 evaluated episodes.

| Model / method | Successes | Success |
|---|---:|---:|
| Original 4B | 110/240 | 45.83% |
| Teacher 27B | 205/240 | 85.42% |
| Complete K4 offline OPD | 184/240 | 76.67% |
| Representativeness-only offline OPD | 170/240 | 70.83% |
| Same-task K4 online OPD | 191/240 | 79.58% |

Offline OPD retains **91.36% of the same-task online gain**: `(184 − 110) / (191 − 110)`. It closes **77.89% of the teacher gap**: `(184 − 110) / (205 − 110)`. These ratios use different denominators.

## Supplementary training repetitions

Mean ± sample standard deviation across **all four** collaborator-reported training repetitions. Each run first averages its evaluation repeats. The original main-table run is separate from this summary.

| K4 OPD selection | Navigation standard | Navigation long horizon |
|---|---:|---:|
| Complete | 81.98 ± 2.73% | 23.75 ± 2.93% |
| Representativeness only | 80.73 ± 1.20% | 23.54 ± 4.20% |
| Fixed random K4 | 76.88 ± 2.86% | 21.67 ± 4.56% |

Complete selection has the highest mean on both panels. [The follow-up snapshot](collaborator_followup_results.json) contains the per-run success counts.

## Additional selection ablations

The complete reference is the original K4 main-table result. Simplified variants below are one collaborator-reported run per condition, separate from the original selection comparison and the training repetitions above.

| K4 OPD selector | Navigation standard | Navigation long horizon | Sokoban |
|---|---:|---:|---:|
| Complete reference | 82.08% | 22.08% | 85.55% |
| Representativeness only | 77.92% | 17.92% | 80.47% |
| Coverage only | 79.17% | 19.17% | 82.42% |
| Unordered matching | 81.25% | 21.25% | 85.55%† |

† Uses the same selected Sokoban set and result as the complete method. The [4B snapshot](paper_results_4b.json) separately retains the original three random subsets and component controls.

## Evaluation and monitoring

Navigation/Sokoban report mean binary success; SpatialGym reports continuous `cogmap_score`. Final evaluation repeats tasks four times: 240 episodes per Navigation panel, 256 Sokoban episodes, 16 held-out single-room SpatialGym episodes, and 80 two-room episodes. Training-room and all-single-room mapping scores remain available in the snapshots.

Learning curves report monitoring at updates 5, 10, …, 100. Navigation monitoring uses 16 of the 60 standard final tasks; final evaluations execute separately. SpatialGym monitoring reuses its four held-out development rooms. Final scores use the fixed update-100 weights.

Core 4B/9B endpoints are single training runs. Supplementary training repetitions, independent-panel evaluations, and additional ablations are reported separately, with data sources identified in the JSON snapshots.

```bash
python scripts/plot_results.py --output outputs/main-results.pdf
```

This command plots the original 4B core snapshot. The current paper's [online/offline budget figure](../assets/figures/data_budget.png) and [README main table](../README.md#results) also expose the newer results.
