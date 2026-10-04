# Post-hoc adaptive-width diagnostic

This analysis uses the completed generation-5 adaptive finalist and its identity,
arithmetic, and native FocusGA controls. It reuses three development trials per
condition (seeds 4001–4003; 20 generations per phase) and five formerly reserved
trials (5001–5005; 80 generations per phase). Both have four alternating phases,
population 64, three training evaluations, and episode cap 500. It does not
change the earlier selection or repeat training. Exact original profiles,
candidate source hashes, and all accessed evidence hashes are recorded in
summary.json and checksums.json; the original frozen validation protocol remains
in reports/adaptive-validation-complete-20261003/frozen/protocol.md.

The logged sigma is the post-tell value for the next generation. The used width
is reconstructed by prepending its initial value 0.5 and removing the final
logged value. Adaptive trials also verify this reconstructed sequence against
their saved host-observation receipt. Returns are those of the population mean
(centroid). Dense active and inactive evaluations use the native recorded keys;
fresh previous-task checkpoint evaluations use independent episodes. The
inactive task in the first phase has not yet been trained. It must not be
interpreted as forgotten performance. The frozen objective is the equal-weight
mean of normalized dense active return and fresh previous-task endpoint return.
Scores are independently recomputed and checked against published trial scores.

The figure displays every old reserved trial without smoothing or seed selection.
Its dots show the three fresh previous-task endpoint means for each trial. The
summary includes every development and old reserved trial, phase-level widths,
training return means, scores, and paired differences with sample standard
deviations. Traces retain each generation's training-fitness mean as well.

This is exploratory mechanism analysis after seeing outcomes. Similarity or
association of widths and returns is not evidence of a causal mechanism. The
old reserved seeds can now motivate a hypothesis, but cannot be reused as an
untouched confirmatory test for it. Development-versus-reserved differences
combine changed seeds and phase duration. FocusGA changes parent selection as
well as mutation width and is therefore an algorithmic comparator rather than
a width-only ablation. No new training or model request is performed.

Reproduce from the repository root into fresh output paths:

```sh
MPLCONFIGDIR=/tmp/shinka-crl-mpl .venv/bin/python scripts/analyze_adaptive_mechanism.py --output /tmp/adaptive-mechanism-recheck --figure /tmp/adaptive-mechanism-recheck
```
