# Formal experimental results

`main.json` contains the three-environment primary results from the **100-update, batch-64, fixed final-update-100** protocol, including original student, teacher, K4 OPD/SFT, full-pool OPD/SFT, and online OPD. `learning_curves.csv` contains the measured five-update monitoring points. They are development curves, not repeated use of final test panels. No unmeasured point is interpolated into the data.

Student: Qwen3.5-4B; teacher: Qwen3.5-27B. There is one training seed. Evaluation repeats each task four times. Navigation/Sokoban report binary success; SpatialGym reports the native continuous cognitive-map score, separated by room split. Equal updates/responses are not equal token or compute budgets.

The candidate pools contain 120 Navigation, 128 Sokoban and 32 SpatialGym trajectories. K4 is the retained training data size, not the total candidate-collection cost. The compact release includes the four Sokoban trajectories; it does not redistribute the full candidate corpora or model weights.

```bash
python scripts/plot_results.py --output outputs/main-results.pdf
```

This export is limited to locally audited core results and does not silently mix in old 50-update runs or collaborator-only ablations. The manuscript remains the source for the broader experimental tables.
