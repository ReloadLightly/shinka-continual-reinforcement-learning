"""Reanalyse closed adaptive-width experiments; no training or candidate selection.

Only checksum-verified published evidence is used. The old reserved partition is
now observed: this diagnostic can motivate hypotheses, not confirm new ones.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("selected", "identity", "arithmetic", "focus")
LABELS = ("Selected program (generation 5)", "Identity (fixed width)",
          "Arithmetic feedback", "Native FocusGA")
COLORS = ("#2263A5", "#C65F31", "#188577", "#8765A3", "#B08A27")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "n": len(values)}


class Evidence:
    def __init__(self, root, inputs):
        self.root, self.inputs = ROOT / root, inputs
        path = self.root / "checksums.json"
        self.published = json.loads(path.read_text())["published_sha256"]
        self.inputs[str(path.relative_to(ROOT))] = digest(path)

    def path(self, relative):
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root.resolve()) and not path.is_symlink(),
                f"Invalid evidence path: {relative}")
        require(relative in self.published and digest(path) == self.published[relative],
                f"Published evidence changed: {path}")
        self.inputs[str(path.relative_to(ROOT))] = self.published[relative]
        return path

    def read(self, relative):
        return json.loads(self.path(relative).read_text())


def trial_data(evidence, prefix, row, partition, condition, profile, candidate):
    records = evidence.read(prefix + "/training/training_metrics.json")
    config = evidence.read(prefix + "/training/config.json")
    evaluation = evidence.read(prefix + "/analysis/evaluation.json")
    manifest = evidence.read(prefix + "/training/manifest.json")
    interval = profile["ne"]["task_interval"]
    require(len(records) == 4 * interval == config["num_generations"], "Curve length differs")
    require(config["seed"] == manifest["seed"] == row["seed"]
            and config["trial"] == evaluation["trial"] == row["trial"]
            and evaluation["eval_seed"] == row["eval_seed"]
            and config["task_interval"] == interval and config["sigma"] == .5,
            "Trial identity or budget differs")
    require(manifest.get("program") == candidate.get("program"), "Program identity differs")
    next_widths = [record["sigma"] for record in records]
    used = [.5, *next_widths[:-1]]
    if condition != "focus":
        adapter = evidence.read(prefix + "/training/adaptive-manifest.json")
        validation = adapter["width_validation"]
        require(validation["raw_sigma_timing"] == "after_tell_next_generation"
                and validation["matched_host_observations"]
                and validation["sigma_used_sha256"] == hashlib.sha256(
                    json.dumps(used, separators=(",", ":")).encode()).hexdigest(),
                "Width timing or host-observation receipt differs")
        require(adapter["program_raw_sha256"] == candidate["source_sha256"],
                "Executed candidate source differs")
    curves = []
    for generation, (record, width) in enumerate(zip(records, used, strict=True)):
        task = (generation // interval) % 2
        require(record["generation"] == generation and record["task"] == task,
                "Unexpected phase schedule")
        active, inactive = record[f"centroid_task{task}"], record[f"centroid_task{1-task}"]
        require(all(math.isfinite(value) and 0 <= value <= 500 for value in (active, inactive)),
                "Invalid centroid return")
        require(math.isfinite(width) and .001 <= width <= 2, "Invalid mutation width")
        curves.append({"partition": partition, "condition": condition, "seed": row["seed"],
                       "generation": generation, "phase": generation // interval + 1,
                       "task": task, "sigma_used": width, "sigma_next": next_widths[generation],
                       "active_return": active, "inactive_return": inactive,
                       "train_fitness_mean": record["train_fitness_mean"]})
    checkpoints = [entry for entry in evaluation["per_task"] if entry["source"] == "centroid"]
    require([entry["task_idx"] for entry in checkpoints] == list(range(4)), "Missing checkpoints")
    previous = [statistics.mean(entry["prev_returns"]) for entry in checkpoints[1:]]
    active_score = statistics.mean(curve["active_return"] for curve in curves) / 500
    previous_score = statistics.mean(previous) / 500
    scores = {"active_score": active_score, "previous_score": previous_score,
              "combined_score": (active_score + previous_score) / 2}
    require(all(math.isclose(value, row["score"][key], abs_tol=1e-12)
                for key, value in scores.items()), "Recomputed objective differs")
    phases = []
    for phase in range(4):
        segment = curves[phase * interval:(phase + 1) * interval]
        phases.append({"phase": phase + 1,
                       "sigma_initial": segment[0]["sigma_used"],
                       "sigma_mean": statistics.mean(x["sigma_used"] for x in segment),
                       "sigma_median": statistics.median(x["sigma_used"] for x in segment),
                       "sigma_first_ten_max": max(x["sigma_used"] for x in segment[:10]),
                       "active_mean_return": statistics.mean(x["active_return"] for x in segment),
                       "inactive_mean_return": statistics.mean(x["inactive_return"] for x in segment),
                       "fresh_previous_mean": previous[phase - 1] if phase else None})
    return {"partition": partition, "condition": condition, "seed": row["seed"],
            "trial": row["trial"], "eval_seed": row["eval_seed"],
            "source_prefix": str((evidence.root / prefix).relative_to(ROOT)),
            "program_sha256": candidate.get("source_sha256"),
            "scores": scores, "sigma_median": statistics.median(used),
            "sigma_min": min(used), "sigma_max": max(used), "phases": phases,
            "reference_metrics": row["score"]["reference_metrics"],
            "curves": curves}


def load_trials(inputs):
    validation = Evidence("reports/adaptive-validation-complete-20261003", inputs)
    plan = validation.read("frozen/plan.json")
    summary = validation.read("summary.json")
    validation.path("frozen/protocol.md")
    require(plan["provenance"]["selected_generation"] == 5 and summary["status"] == "complete",
            "Expected closed generation-5 reserved evaluation")
    candidates = {candidate["id"]: candidate for candidate in plan["conditions"]}
    for condition in CONDITIONS:
        if candidates[condition]["source"]:
            source = validation.path("frozen/" + candidates[condition]["source"])
            require(digest(source) == candidates[condition]["source_sha256"], "Frozen program differs")
    trials = []
    for condition in CONDITIONS:
        for row in summary["conditions"][condition]["trials"]:
            trials.append(trial_data(validation, f"raw/trials/{row['index']:03d}/attempt_0001",
                                     row, "old_reserved", condition, plan["profile"], candidates[condition]))
    endpoint = Evidence("reports/adaptive-endpoint-complete-20261003", inputs)
    search = endpoint.read("summary.json")
    endpoint.read("endpoint-plan.json")
    endpoint.read("raw/evaluation/plan.json")
    selected = next(program for program in search["programs"] if program["generation"] == 5)
    require(selected["program_sha256"] == candidates["selected"]["source_sha256"],
            "Development and reserved program differ")
    controls = Evidence("reports/adaptive-controls-20261003", inputs)
    controls_summary = controls.read("summary.json")
    controls.read("raw/plan.json")
    requests = {request["request_id"]: request for request in controls_summary["requests"]
                if request["status"] == "complete"}
    for condition in CONDITIONS:
        evidence = endpoint if condition == "selected" else controls
        request = selected["request"] if condition == "selected" else requests[condition]
        prefix = ("raw/evaluation/" if condition == "selected" else "raw/") + request["cache_origin"]
        cache = evidence.read(prefix + "/summary.json")
        require(cache["status"] == "complete" and cache["candidate"]["source_sha256"]
                == candidates[condition]["source_sha256"], "Development recipe differs")
        for row in cache["trials"]:
            trials.append(trial_data(evidence, prefix + f"/seed_{row['seed']}", row,
                                     "development", condition, search["evaluation_profile"], candidates[condition]))
    require(len(trials) == 32, "Expected 12 development and 20 old reserved trials")
    return trials, {"development": search["evaluation_profile"], "old_reserved": plan["profile"]}


def plot(trials, output):
    figure, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey="row")
    for column, (condition, label) in enumerate(zip(CONDITIONS, LABELS, strict=True)):
        selected = sorted((trial for trial in trials if trial["partition"] == "old_reserved"
                           and trial["condition"] == condition), key=lambda trial: trial["seed"])
        axes[0, column].set_title(label, fontsize=10)
        for trial, color in zip(selected, COLORS, strict=True):
            curves = trial["curves"]
            x = [row["generation"] + 1 for row in curves]
            for axis, key in zip(axes[:, column], ("sigma_used", "active_return", "inactive_return"), strict=True):
                axis.plot(x, [row[key] for row in curves], color=color, lw=.9, alpha=.85,
                          label=str(trial["seed"]))
            # Fresh evaluations use independent episodes from the dense curves.
            axes[2, column].scatter([160, 240, 320],
                                    [phase["fresh_previous_mean"] for phase in trial["phases"][1:]],
                                    color=color, edgecolor="white", linewidth=.35, s=22, zorder=5)
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylim(.0008, .65)
        for axis in axes[:, column]:
            for switch in (80.5, 160.5, 240.5):
                axis.axvline(switch, color="0.65", ls="--", lw=.6)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=.18)
            axis.set_xlim(1, 323)
        for axis in axes[1:, column]:
            axis.set_ylim(0, 520)
        axes[2, column].set_xlabel("Completed generations")
    for axis, label in zip(axes[:, 0], ("Mutation width used (log scale)", "Active-task return", "Inactive-task return"), strict=True):
        axis.set_ylabel(label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, title="Old reserved seed", loc="upper center", ncol=5,
                  bbox_to_anchor=(.5, .985), frameon=False)
    figure.suptitle("Adaptive-width diagnostic on completed reserved trials", y=1.025, fontsize=14)
    figure.text(.5, -.025, "Lines: unsmoothed centroid evaluations. Dots: fresh previous-task checkpoint evaluations. "
                "Dashed lines: task switches.\nObserved data support hypothesis generation; width trajectories are not causal interventions.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, .94))
    for extension in ("svg", "pdf"):
        figure.savefig(output.with_suffix("." + extension), bbox_inches="tight")
    plt.close(figure)


def analyze(output, figure):
    require(not output.exists() and not figure.with_suffix(".svg").exists()
            and not figure.with_suffix(".pdf").exists(), "Use fresh output paths")
    inputs = {str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__)}
    for relative in ("src/shinka_crl/adaptive.py", "scripts/adaptive_train.py", "upstream.lock.json"):
        inputs[relative] = digest(ROOT / relative)
    # The unchanged native loop records sigma after tell for FocusGA as well.
    for relative in ("source/runners/train_nes.py", "source/algorithms/ne/ga.py"):
        path = ROOT / ".upstream/continual_neuroevolution" / relative
        inputs[str(path.relative_to(ROOT))] = digest(path)
    trials, profiles = load_trials(inputs)
    aggregate, paired = {}, {}
    for partition in ("development", "old_reserved"):
        aggregate[partition] = {}
        by_condition = {}
        for condition in CONDITIONS:
            by_condition[condition] = sorted((trial for trial in trials if trial["partition"] == partition
                                             and trial["condition"] == condition), key=lambda trial: trial["seed"])
            aggregate[partition][condition] = {
                key: describe([trial["scores"][key] for trial in by_condition[condition]])
                for key in ("combined_score", "active_score", "previous_score")}
        paired[partition] = {}
        for control in CONDITIONS[1:]:
            require([trial["seed"] for trial in by_condition["selected"]]
                    == [trial["seed"] for trial in by_condition[control]], "Unpaired trials")
            paired[partition]["selected_minus_" + control] = {
                key: describe([a["scores"][key] - b["scores"][key]
                               for a, b in zip(by_condition["selected"], by_condition[control], strict=True)])
                for key in ("combined_score", "active_score", "previous_score")}
    output.mkdir(parents=True)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plot(trials, figure)
    with (output / "traces.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trials[0]["curves"][0].keys())
        writer.writeheader()
        for trial in trials:
            writer.writerows(trial.pop("curves"))
    summary = {"analysis_version": "adaptive-mechanism-v1", "kind": "posthoc_diagnostic",
               "new_training_trials": 0, "new_model_requests": 0,
               "selection_status": "All original selections and outcomes remain closed.",
               "profiles": profiles, "conditions": list(CONDITIONS), "trials": trials,
               "aggregate": aggregate, "paired_differences": paired,
               "definitions": {
                   "sigma_used": "Initial 0.5 followed by all but the final post-tell logged sigma.",
                   "active_score": "Arithmetic mean of dense active-task centroid returns divided by 500.",
                   "previous_score": "Mean of three fresh previous-task endpoint means divided by 500.",
                   "combined_score": "Equal-weight mean of active_score and previous_score.",
                   "inactive_curve": "The other recurring task, including before it has first been trained; not all samples measure forgetting.",
                   "uncertainty": "Sample SD across paired task draws; no confidence interval or significance claim.",
               },
               "interpretation": [
                   "The selected formula approaches target width 0.008 when its normalized mean and best fitness both reach one.",
                   "Similar narrow selected widths coexist with both poor and strong previous-task retention. Width alone does not identify the cause of forgetting.",
                   "Arithmetic feedback can retain performance at even smaller widths; simple width collapse is not a sufficient explanation of the selected rule's negative result.",
                   "FocusGA also alters parent selection and evaluates the centroid inside its population budget; it is not a width-only causal control.",
                   "Development and reserved experiments have different seeds and phase durations, so their outcome difference does not isolate either factor.",
                   "Fitness statistics, not inactive-task evaluations or phase labels, drive candidate updates. Any switch response follows observations on the new task.",
                   "Test whether feedback-guided program search improves over equally budgeted simple adaptive or open-loop controls using independent outer searches and new untouched test draws.",
               ]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    protocol = """# Post-hoc adaptive-width diagnostic

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
"""
    (output / "protocol.md").write_text(protocol)
    published = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): digest(path)
                 for path in [output / "summary.json", output / "traces.csv", output / "protocol.md",
                              figure.with_suffix(".svg"), figure.with_suffix(".pdf")]}
    (output / "checksums.json").write_text(json.dumps({"input_sha256": inputs,
                                                       "output_sha256": published}, indent=2) + "\n")
    print(json.dumps({"trials": len(trials), "verified_inputs": len(inputs),
                      "old_reserved_selected_minus_focus": paired["old_reserved"]["selected_minus_focus"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/adaptive-mechanism-20261004")
    parser.add_argument("--figure", type=Path, default=ROOT / "reports/figures/adaptive-mechanism-20261004")
    args = parser.parse_args()
    analyze(args.output.resolve(), args.figure.resolve())
