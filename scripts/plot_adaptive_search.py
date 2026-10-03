"""Plot verified adaptive search scores and the selected rule's applied mutation widths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label):
    require(type(actual) in (float, int) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
            f"Score differs from raw evidence: {label}")


class Evidence:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.checksums = json.loads((self.root / "checksums.json").read_text())["published_sha256"]
        self.inputs = {"checksums.json": sha256(self.root / "checksums.json")}

    def read(self, name: str | Path):
        name = str(name)
        path = (self.root / name).resolve()
        require(path.is_relative_to(self.root), "Evidence path escapes published report")
        require(name in self.checksums and sha256(path) == self.checksums[name],
                f"Published evidence changed: {name}")
        self.inputs[name] = self.checksums[name]
        return json.loads(path.read_text())


def score_candidate(evidence: Evidence, base: Path, request: dict, profile: dict) -> dict:
    """Independently derive every displayed score from training and post-hoc episodes."""
    cache = evidence.read(base / "summary.json")
    require(cache["status"] == "complete" and cache["aggregate"] == request["aggregate"],
            "Request and cache aggregate disagree")
    require(cache["context_sha256"] == request["context_sha256"], "Evaluation contexts differ")
    seeds = profile["seeds"]
    require([trial["seed"] for trial in cache["trials"]] == seeds and len(seeds) == 3,
            "Expected the three frozen development seeds")
    count, interval = profile["ne"]["num_generations"], profile["ne"]["task_interval"]
    require(count == 80 and interval == 20 and profile["episode_length"] == 500,
            "Figure expects the frozen 80-generation adaptive development profile")
    scores, traces = [], {}
    for trial in cache["trials"]:
        seed = trial["seed"]
        root = base / f"seed_{seed}"
        records = evidence.read(root / "training/training_metrics.json")
        evaluation = evidence.read(root / "analysis/evaluation.json")
        require(len(records) == count and evaluation["episodes"] == 10,
                "Incomplete training curve or changed post-hoc episode count")
        require(evaluation["trial"] == seed + 1 and evaluation["eval_seed"] == seed + 900000,
                "Post-hoc evaluation seed identity changed")
        active = []
        for generation, record in enumerate(records):
            task = generation // interval % 2
            require(record["generation"] == generation and record["task"] == task,
                    "Training task sequence changed")
            value = record[f"centroid_task{task}"]
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 500,
                    "Invalid active-task centroid return")
            active.append(value)
        previous = [entry for entry in evaluation["per_task"]
                    if entry["source"] == "centroid" and entry["task_idx"] > 0]
        require([entry["task_idx"] for entry in previous] == [1, 2, 3],
                "Expected all three consecutive-switch evaluations")
        previous_returns = []
        for entry in previous:
            values = entry["prev_returns"]
            require(len(values) == 10 and all(type(value) in (int, float)
                    and math.isfinite(value) and 0 <= value <= 500 for value in values),
                    "Invalid previous-task episode returns")
            previous_returns.append(statistics.mean(values))
        row = {"active_score": statistics.mean(active) / 500,
               "previous_score": statistics.mean(previous_returns) / 500}
        row["combined_score"] = (row["active_score"] + row["previous_score"]) / 2
        for name, value in row.items():
            close(trial["score"][name], value, f"seed {seed} {name}")
        scores.append(row)
        traces[seed] = records
    for name in ("combined_score", "active_score", "previous_score"):
        values = [row[name] for row in scores]
        aggregate = cache["aggregate"]["scores"][name]
        require(aggregate["n"] == 3, "Aggregate seed count changed")
        close(aggregate["mean"], statistics.mean(values), f"{name} mean")
        close(aggregate["sample_sd"], statistics.stdev(values), f"{name} sample SD")
    return {"seeds": seeds, "values": [row["combined_score"] for row in scores],
            "mean": statistics.mean(row["combined_score"] for row in scores),
            "sample_sd": statistics.stdev(row["combined_score"] for row in scores),
            "records": traces}


def completed_stage(summary: dict, plan: dict) -> int:
    """Bind the plotted archive to a complete target declared before the search."""
    target = summary["programs_evaluated"]
    require(type(target) is int and target in (5, 13, 25) and target in plan["stages"],
            "Figure requires a declared 5-, 13-, or 25-slot stage")
    require(summary["status"] == "complete" and summary["slots_consumed"] == target,
            "Figure requires a complete search stage")
    sessions = summary["sessions"]
    require(bool(sessions) and all(session["status"] == "complete" for session in sessions)
            and sessions[-1]["target"] == target,
            "Completed search session does not match the plotted stage")
    require([row["generation"] for row in summary["programs"]] == list(range(target)),
            "Native generation sequence is incomplete")
    return target


def plot_search(report_dir: Path, controls_dir: Path, output: Path) -> dict:
    output = output.resolve()
    require(output.suffix == ".svg", "Output must be an SVG filename")
    destinations = [output, output.with_suffix(".pdf"), output.with_suffix(".json")]
    require(not any(path.exists() for path in destinations), "Refusing to overwrite figure artifacts")
    evidence, controls = Evidence(report_dir), Evidence(controls_dir)
    summary = evidence.read("summary.json")
    profile = evidence.read("raw/evaluation/plan.json")["profile"]
    target = completed_stage(summary, evidence.read("raw/search/plan.json"))
    rows = []
    for program in summary["programs"]:
        request = program["request"]
        base = Path("raw/evaluation") / request["cache_origin"]
        row = score_candidate(evidence, base, request, profile)
        rows.append({**row, "generation": program["generation"], "base": base,
                     "program_sha256": program["program_sha256"]})
    control_summary = controls.read("summary.json")
    require(controls.read("raw/plan.json")["profile"] == profile,
            "FocusGA control uses a different evaluation profile")
    focus_request = control_summary["control_requests"]["focus"]
    require(focus_request["aggregate"] == summary["controls"]["focus"]["aggregate"],
            "FocusGA reference differs from frozen search control")
    focus = score_candidate(controls, Path("raw") / focus_request["cache_origin"],
                            focus_request, profile)
    best = max(rows, key=lambda row: (row["mean"], -row["generation"]))
    widths = {}
    for seed in profile["seeds"]:
        receipt = evidence.read(best["base"] / f"seed_{seed}/training/adaptive-manifest.json")
        validation = receipt["width_validation"]
        raw = [record["sigma"] for record in best["records"][seed]]
        used = [0.5, *raw[:-1]]
        require(receipt["status"] == "complete" and receipt["invalid_update"] is False,
                "Adaptive launcher did not validate the complete rule")
        require(validation["raw_sigma_timing"] == "after_tell_next_generation"
                and validation["matched_host_observations"] is True
                and validation["final_update_validated"] is True,
                "Unknown mutation-width observation timing")
        require(hashlib.sha256(json.dumps(used, separators=(",", ":")).encode()).hexdigest()
                == validation["sigma_used_sha256"], "Applied sigma trace differs from receipt")
        require(all(type(value) in (int, float) and math.isfinite(value)
                    and 0.001 <= value <= 2 for value in used), "Invalid applied mutation width")
        widths[seed] = used

    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9.5,
             "axes.titlesize": 10.5, "axes.linewidth": 0.65, "axes.edgecolor": "#87909A",
             "axes.labelcolor": "#25313B", "xtick.color": "#55616C", "ytick.color": "#55616C",
             "xtick.labelsize": 8, "ytick.labelsize": 8, "grid.color": "#E1E5E8",
             "grid.linewidth": 0.55, "svg.fonttype": "none", "pdf.fonttype": 42,
             "svg.hashsalt": f"shinka-adaptive-stage{target}-v1", "figure.facecolor": "white"}
    # Preserve room for the width traces as the cumulative archive grows.
    figure_width, score_width = {5: (10.2, 1), 13: (12.2, 1.4), 25: (15.2, 2)}[target]
    with plt.rc_context(style):
        fig, (left, right) = plt.subplots(1, 2, figsize=(figure_width, 3.65),
                                         gridspec_kw={"width_ratios": [score_width, 1]})
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.80, wspace=0.30)
        for ax in (left, right):
            ax.set_axisbelow(True)
            ax.grid(axis="y")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        stage_label = {5: "Five", 13: "Thirteen", 25: "Twenty-five"}[target]
        left.set_title(f"(a)  {stage_label} evaluated programs", loc="left", pad=20)
        x = [row["generation"] for row in rows]
        means = [row["mean"] for row in rows]
        deviations = [row["sample_sd"] for row in rows]
        left.errorbar(x, means, yerr=deviations, color="#2263A5", fmt="o", markersize=5,
                      linewidth=1.1, capsize=4, zorder=3)
        for row in rows:
            left.scatter(np.asarray([row["generation"]] * 3) + [-0.12, 0, 0.12],
                         row["values"], s=19, color="#2263A5", alpha=0.32, zorder=2)
        left.scatter([best["generation"]], [best["mean"]], facecolors="none", edgecolors="#188577",
                     linewidths=1.3, s=112, zorder=4)
        left.axhline(focus["mean"], color="#636F79", linestyle=(0, (4, 3)), linewidth=1)
        left.set(xlim=(-0.45, target - 0.55), xticks=x,
                 xticklabels=["0\nidentity", *[str(generation) for generation in x[1:]]],
                 xlabel="Outer program generation", ylabel="Combined development score")
        left.set_ylim(min(0, min(m - s for m, s in zip(means, deviations)) - 0.035),
                      max(1, max(m + s for m, s in zip(means, deviations)) + 0.035))
        legend_layout = ({"ncol": 2, "bbox_to_anchor": (0, 1.10), "borderaxespad": 0}
                         if target > 5 else {})
        left.legend(handles=[Line2D([0], [0], marker="o", linestyle="none", color="#2263A5",
                                    markersize=4, label="Mean ± sample SD (3 seeds)"),
                             Line2D([0], [0], color="#636F79", linestyle=(0, (4, 3)),
                                    label="Native FocusGA control mean")],
                    loc="upper left", frameon=False, fontsize=7.4, handlelength=2.3,
                    **legend_layout)
        right.set_title(f"(b)  Applied widths · selected generation {best['generation']}",
                        loc="left", pad=20)
        for phase in range(4):
            if phase % 2:
                right.axvspan(phase * 20, (phase + 1) * 20, color="#EBEEF0", alpha=0.7,
                              linewidth=0, zorder=0)
            if phase:
                right.axvline(phase * 20, color="#A7AFB7", linestyle=(0, (2, 3)), linewidth=0.7)
            right.text(phase * 20 + 10, 1.025, "AB"[phase % 2],
                       transform=right.get_xaxis_transform(), ha="center", color="#6B7580", fontsize=8)
        for (seed, values), color in zip(widths.items(), ("#2263A5", "#188577", "#C65F31")):
            right.step(np.arange(81), [*values, values[-1]], where="post", color=color,
                       linewidth=1.4, label=str(seed), alpha=0.9)
        right.set(yscale="log", xlim=(0, 80), ylim=(0.0008, 2.5), xticks=[0, 20, 40, 60, 80],
                  xlabel="Inner GA generations", ylabel="Applied mutation width σ (log scale)")
        right.set_yticks([0.001, 0.01, 0.1, 0.5, 2], labels=["0.001", "0.01", "0.1", "0.5", "2"])
        right.legend(loc="lower left", frameon=False, fontsize=7.4, ncol=3,
                     title="Training seed", title_fontsize=7.4, handlelength=1.7, columnspacing=1.2)
        fig.text(0.07, 0.025, "Development-selected; no held-out evaluation · faint points: individual "
                 "seeds · error bars show trial dispersion, not confidence intervals", fontsize=7.5,
                 color="#65717D")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".adaptive-figure-", dir=output.parent) as temporary:
            staged = [Path(temporary) / path.name for path in destinations]
            title = f"Adaptive program search: {target}-slot development stage"
            fig.savefig(staged[0], metadata={"Date": None, "Title": title})
            fig.savefig(staged[1], metadata={"CreationDate": None, "ModDate": None,
                                            "Title": title})
            provenance = {
                "schema_version": 1, "kind": "adaptive_development_search",
                "stage_target": target,
                "plot_script": "scripts/plot_adaptive_search.py", "plot_script_sha256": sha256(Path(__file__)),
                "inputs": {evidence.root.name: evidence.inputs, controls.root.name: controls.inputs},
                "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
                "selection": "Highest combined development mean; ties choose earlier generation",
                "selected_generation": best["generation"], "selected_program_sha256": best["program_sha256"],
                "selected_seed_widths": widths,
                "displayed_scores": [{key: row[key] for key in
                                      ("generation", "seeds", "values", "mean", "sample_sd")} for row in rows],
                "focus_control": {key: focus[key] for key in ("seeds", "values", "mean", "sample_sd")},
                "sigma_timing": "Applied generation0 sigma=0.5; later widths are previous raw row sigma",
                "objective": "0.5 * mean_active_return/500 + 0.5 * mean_previous_task_return/500",
                "held_out": False, "smoothing": None,
                "outputs_sha256": {path.name: sha256(path) for path in staged[:2]},
            }
            staged[2].write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
            require(not any(path.exists() for path in destinations), "Refusing to overwrite figure artifacts")
            for source, destination in zip(staged, destinations):
                source.rename(destination)
        plt.close(fig)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--controls-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        provenance = plot_search(args.report_dir, args.controls_report, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Adaptive figure rejected: {exc}\n")
    print(json.dumps({"selected_generation": provenance["selected_generation"],
                      "outputs": list(provenance["outputs_sha256"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
