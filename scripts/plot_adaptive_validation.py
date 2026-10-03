"""Plot a complete, verified five-seed reserved adaptive comparison."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CONDITIONS = ("selected", "identity", "arithmetic", "focus", "static_shinka11", "static_random24")
SEEDS = (5001, 5002, 5003, 5004, 5005)
SCORES = ("combined_score", "active_score", "previous_score")
SEED_COLORS = ("#2263A5", "#C65F31", "#188577", "#8765A3", "#B08A27")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def describe(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "n": len(values)}


def load_data(report_dir):
    """Recompute plotted values from exported curves and fresh checkpoint returns."""
    report_dir = Path(report_dir).resolve()
    require(report_dir.is_dir(), "A complete reserved export is required")
    checksum_path = report_dir / "checksums.json"
    require(not checksum_path.is_symlink(), "Report checksum must be a regular file")
    checksums = json.loads(checksum_path.read_text())
    published = checksums["published_sha256"]
    inputs = {"checksums.json": digest(checksum_path)}
    actual = set()
    for path in report_dir.rglob("*"):
        require(not path.is_symlink(), "Published artifacts must not be symlinks")
        if path.is_file() and path != checksum_path:
            actual.add(str(path.relative_to(report_dir)))
    require(actual == set(published), "Published report file set differs from checksums")
    for relative, expected in published.items():
        path = (report_dir / relative).resolve()
        require(path.is_relative_to(report_dir) and digest(path) == expected,
                f"Published artifact changed: {relative}")

    def read(relative):
        require(relative in published, f"Unpublished input: {relative}")
        inputs[relative] = published[relative]
        return json.loads((report_dir / relative).read_text())

    summary, plan = read("summary.json"), read("frozen/plan.json")
    require(summary["kind"] == plan["kind"] == "reserved" and summary["status"] == "complete"
            and summary["protocol_version"] == plan["protocol_version"] == "adaptive-reserved-validation-v1",
            "Only a complete reserved comparison may be plotted")
    require(summary["planned_trials"] == summary["scored_trials"] == plan["planned_trials"] == 30,
            "All thirty reserved trials must be scored")
    require(plan["profile"]["name"] == "adaptive-validation" and plan["profile"]["seeds"] == list(SEEDS)
            and plan["profile"]["ne"] == {"num_generations": 320, "task_interval": 80,
                                           "pop_size": 64, "num_evals": 3}
            and plan["profile"]["num_phases"] == 4 and plan["profile"]["num_tasks"] == 2
            and plan["profile"]["episode_length"] == 500,
            "Reserved profile or seed partition changed")
    require(plan["objective_weights"] == {"active": .5, "previous": .5}
            and plan["objective_version"] == "adaptive-active-previous-v1"
            and plan["posthoc_episodes"] == 10,
            "Adaptive objective or checkpoint episode allocation changed")
    require(plan["trial_steps_nominal"] == 30720000
            and plan["planned_nominal_training_steps"] == 921600000
            and plan["planned_fresh_evaluation_episodes"] == 9000
            and summary["realized_fresh_evaluation_episodes"] == 9000,
            "Reserved trial or fresh evaluation budget differs")
    frozen_receipt = read("frozen/receipt.json")
    frozen_files = {name.removeprefix("frozen/"): value for name, value in published.items()
                    if name.startswith("frozen/") and name != "frozen/receipt.json"}
    require(frozen_files == frozen_receipt, "Frozen handoff receipt differs")
    require(published["frozen/protocol.md"] == plan["protocol_sha256"], "Frozen protocol changed")
    require(plan["provenance"]["closure"]["status"] == "closed"
            and plan["provenance"]["closure"]["slots_consumed"] == 25
            and plan["provenance"]["closure"]["remaining_generations"] == [],
            "Selection lacks a closed development allocation")
    conditions = plan["conditions"]
    require(tuple(c["id"] for c in conditions) == CONDITIONS
            and set(summary["conditions"]) == set(CONDITIONS), "Reserved condition membership changed")
    for candidate in conditions:
        source = candidate["source"]
        if source:
            require(published.get("frozen/" + source) == candidate["source_sha256"],
                    "Frozen candidate source changed")
            if candidate["variant"] == "ga_adaptive":
                parsed = ast.parse((report_dir / "frozen" / source).read_text())
                metadata = {"source_sha256": candidate["source_sha256"],
                            "canonical_ast_sha256": hashlib.sha256(ast.dump(
                                parsed, annotate_fields=True, include_attributes=False).encode()).hexdigest(),
                            "ast_nodes": sum(1 for _ in ast.walk(parsed)), "grammar_version": "adaptive-width-v1"}
                require(metadata == candidate["program"], "Frozen candidate AST changed")
    ranked = plan["provenance"]["ranked_candidates"]
    require(len(ranked) == 21 and len({r["generation"] for r in ranked}) == 21
            and any(r["generation"] == 0 for r in ranked)
            and all(math.isfinite(r["development_score"]) for r in ranked)
            and ranked == sorted(ranked, key=lambda r: (-r["development_score"], r["generation"])),
            "Development selection ranking changed")
    selected_generation = plan["provenance"]["selected_generation"]
    require(selected_generation == ranked[0]["generation"]
            and conditions[0]["program"] == ranked[0]["program"], "Selected source differs from ranking")
    recipes = plan["unique_recipes"]
    require(len(recipes) == 6 and len({r["recipe_key"] for r in recipes}) == 6
            and [r["candidate"] for r in recipes] == conditions
            and [r["memberships"] for r in recipes] == [[c] for c in CONDITIONS]
            and all(json_digest(r["recipe"]) == r["recipe_key"] for r in recipes),
            "Full recipe identity or deduplication changed")
    expected_order = [{"index": 6 * i + j, "recipe_key": recipe["recipe_key"],
                       "memberships": recipe["memberships"], "seed": seed,
                       "trial": seed + 1, "eval_seed": seed + 900000}
                      for i, seed in enumerate(SEEDS) for j, recipe in enumerate(recipes)]
    require(plan["trial_order"] == expected_order, "Reserved seed/task/evaluation mapping changed")
    state, state_receipt = read("raw/state.json"), read("raw/state-receipt.json")
    require(state_receipt == {"sha256": published["raw/state.json"]}
            and state["context_sha256"] == json_digest(plan)
            and state["sessions"] == summary["sessions"], "Reserved state binding changed")
    completed = {}
    for attempt in state["attempts"]:
        prefix = "raw/" + attempt["path"]
        require(published.get(prefix + "/receipt.json") == attempt["receipt_sha256"],
                "Trial receipt identity changed")
        receipt = read(prefix + "/receipt.json")
        # Binary checkpoints stay local; every published trial artifact must
        # still match the original receipt preserved by the compact exporter.
        exported = {name.removeprefix(prefix + "/"): value for name, value in published.items()
                    if name.startswith(prefix + "/") and name != prefix + "/receipt.json"}
        require(all(receipt.get(name) == value for name, value in exported.items()),
                "Published trial evidence differs from its original receipt")
        saved = read(prefix + "/summary.json")
        row = expected_order[attempt["trial_index"]]
        require(saved["context_sha256"] == json_digest(plan) and saved["trial"] == row,
                "Trial belongs to another condition, seed or context")
        if saved["status"] != "complete":
            require(saved["status"] in ("failed", "interrupted"), "Unresolved trial attempt")
            continue
        require(row["index"] not in completed, "Repeated completed trial")
        records = read(prefix + "/training/training_metrics.json")
        require(len(records) == 320, "Incomplete reserved training curve")
        active = []
        for index, record in enumerate(records):
            task = (index // 80) % 2
            require(record["generation"] == index and record["task"] == task,
                    "Training curve schedule changed")
            value = record[f"centroid_task{task}"]
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 500,
                    "Invalid active-task centroid return")
            active.append(value)
        evaluation = read(prefix + "/analysis/evaluation.json")
        require(evaluation["trial"] == row["trial"] and evaluation["eval_seed"] == row["eval_seed"]
                and evaluation["episodes"] == 10, "Fresh checkpoint evaluation identity changed")
        centroid = [entry for entry in evaluation["per_task"] if entry["source"] == "centroid"]
        require([r["task_idx"] for r in centroid] == [0, 1, 2, 3], "Incomplete centroid checkpoints")
        previous = []
        for entry in centroid[1:]:
            values = entry["prev_returns"]
            require(len(values) == 10 and all(type(value) in (int, float) and math.isfinite(value)
                                            and 0 <= value <= 500 for value in values),
                    "Invalid previous-task checkpoint returns")
            previous.append(sum(values) / 10)
        active_score = (sum(active) / 320) / 500
        previous_score = (sum(previous) / 3) / 500
        scores = {"combined_score": .5 * active_score + .5 * previous_score,
                  "active_score": active_score, "previous_score": previous_score}
        result = saved["result"]
        require({key: result[key] for key in row} == row
                and all(result["score"][key] == value for key, value in scores.items()),
                "Stored trial score differs from raw training and checkpoint evidence")
        require(result["fresh_evaluation_episodes"] == 300
                and sum(len(v) for e in evaluation["per_task"] for k, v in e.items()
                        if k.endswith("returns")) == 300, "Fresh episode accounting changed")
        completed[row["index"]] = result
    require(sorted(completed) == list(range(30)), "Incomplete thirty-trial comparison")
    plotted = []
    labels = (f"Selected\n(gen. {selected_generation})", "Identity", "Arithmetic", "FocusGA",
              "Static\nShinka 11", "Static\nRandom 24")
    for identity, label in zip(CONDITIONS, labels, strict=True):
        rows = summary["conditions"][identity]["trials"]
        require([row["seed"] for row in rows] == list(SEEDS)
                and all(row == completed[row["index"]] and row["memberships"] == [identity] for row in rows),
                "Summary condition trials differ from verified raw trials")
        aggregated = {key: describe([row["score"][key] for row in rows]) for key in SCORES}
        require(aggregated == summary["conditions"][identity]["aggregate"]["scores"],
                "Summary mean or sample standard deviation differs")
        plotted.append({"condition": identity, "label": label, "scores": aggregated,
                        "seeds": [{"seed": row["seed"], **{key: row["score"][key] for key in SCORES}}
                                  for row in rows]})
    selected = plotted[0]["seeds"]
    require(set(summary["comparisons"]) == set(CONDITIONS[1:]), "Missing paired comparisons")
    for control in plotted[1:]:
        paired = [{"seed": left["seed"], "difference": left["combined_score"] - right["combined_score"]}
                  for left, right in zip(selected, control["seeds"], strict=True)]
        comparison = summary["comparisons"][control["condition"]]
        require(comparison["paired_differences"] == paired
                and comparison["aggregate"] == describe([row["difference"] for row in paired])
                and comparison["primary"] is (control["condition"] == "focus")
                and comparison["complete_five_seed_comparison"] is True, "Paired arithmetic changed")
    return {"profile": plan["profile"], "objective": plan["objective_version"],
            "objective_weights": plan["objective_weights"], "selected_generation": selected_generation,
            "conditions": plotted, "primary_comparison": summary["comparisons"]["focus"],
            "verified_completed_trials": 30, "verified_published_artifacts": len(published),
            "interpretation": "Reserved transfer validation; descriptive paired comparison, separate from paper reporting"}, inputs


def plot_adaptive_validation(report_dir, output):
    report_dir, output = Path(report_dir).resolve(), Path(output).resolve()
    targets = [output, output.with_suffix(".pdf"), output.with_suffix(".json")]
    require(output.suffix == ".svg" and not any(path.exists() for path in targets)
            and not output.is_relative_to(report_dir),
            "Use fresh SVG, PDF and JSON paths outside the immutable report")
    data, inputs = load_data(report_dir)
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
             "axes.linewidth": .65, "axes.edgecolor": "#87909A", "axes.labelcolor": "#25313B",
             "xtick.color": "#55616C", "ytick.color": "#55616C", "xtick.labelsize": 8,
             "ytick.labelsize": 8, "grid.color": "#E1E5E8", "grid.linewidth": .55,
             "svg.fonttype": "none", "svg.hashsalt": "shinka-crl-adaptive-validation-v1",
             "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), gridspec_kw={"width_ratios": [1.18, 1.]})
        fig.subplots_adjust(left=.065, right=.985, bottom=.30, top=.80, wspace=.25)
        for ax in axes:
            ax.set_axisbelow(True)
            ax.grid(axis="y")
            for edge in ("top", "right"):
                ax.spines[edge].set_visible(False)
        aggregates = [row["scores"]["combined_score"] for row in data["conditions"]]
        axes[0].set(xticks=range(6), xticklabels=[row["label"] for row in data["conditions"]],
                    xlim=(-.45, 5.45), ylabel="Combined score J",
                    ylim=(min(0., min(a["mean"] - a["sample_sd"] for a in aggregates)) - .035,
                          max(1., max(a["mean"] + a["sample_sd"] for a in aggregates)) + .035))
        axes[0].set_title("A  Reserved scores by condition", loc="left", pad=26, fontsize=10,
                          fontweight="semibold", color="#25313B")
        for index, condition in enumerate(data["conditions"]):
            for seed_index, row in enumerate(condition["seeds"]):
                axes[0].scatter(index + (seed_index - 2) * .075, row["combined_score"],
                                s=24, color=SEED_COLORS[seed_index], alpha=.88, linewidths=0, zorder=3)
            aggregate = condition["scores"]["combined_score"]
            axes[0].errorbar(index, aggregate["mean"], yerr=aggregate["sample_sd"], fmt="D",
                             markersize=4.5, capsize=3, linewidth=1.1, color="#25313B", zorder=2)
        primary = data["primary_comparison"]
        differences = primary["paired_differences"]
        aggregate = primary["aggregate"]
        bound = max(.10, *(abs(row["difference"]) for row in differences),
                    abs(aggregate["mean"] - aggregate["sample_sd"]),
                    abs(aggregate["mean"] + aggregate["sample_sd"])) * 1.14
        axes[1].axhline(0, color="#64717D", linewidth=.9, linestyle=(0, (4, 3)), zorder=1)
        for index, row in enumerate(differences):
            axes[1].scatter(index, row["difference"], s=32, color=SEED_COLORS[index], linewidths=0, zorder=3)
        axes[1].errorbar(5.45, aggregate["mean"], yerr=aggregate["sample_sd"], fmt="D",
                         markersize=5, capsize=4, linewidth=1.2, color="#25313B")
        axes[1].set(xticks=[*range(5), 5.45], xticklabels=[*[str(seed) for seed in SEEDS], "Mean\n± SD"],
                    xlim=(-.45, 6.), ylim=(-bound, bound), ylabel="J selected − J FocusGA")
        axes[1].set_title("B  Primary paired difference", loc="left", pad=26, fontsize=10,
                          fontweight="semibold", color="#25313B")
        axes[0].text(0, 1.04, "Five seeds per condition; larger J is better", transform=axes[0].transAxes,
                     fontsize=7.5, color="#65717D")
        axes[1].text(0, 1.04, "Positive differences favor the selected rule", transform=axes[1].transAxes,
                     fontsize=7.5, color="#65717D")
        handles = [Line2D([], [], color=color, marker="o", linestyle="none", markersize=4,
                          label=str(seed)) for seed, color in zip(SEEDS, SEED_COLORS, strict=True)]
        handles.append(Line2D([], [], color="#25313B", marker="D", markersize=4, linewidth=1.,
                              label="Mean ± sample SD"))
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.059, .125), frameon=False,
                   ncol=6, fontsize=7.5, handlelength=1.4, columnspacing=1.8)
        fig.text(.065, .077, "J = 0.5 active-task return + 0.5 fresh previous-task return, each divided by 500. "
                 "Error bars: sample SD across five seeds.", fontsize=7.5, color="#65717D")
        fig.text(.065, .031, "Paired comparisons are descriptive. Reserved transfer validation remains "
                 "separate from final paper-reporting trials.", fontsize=7.5, color="#65717D")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".adaptive-validation-figure-", dir=output.parent) as temporary:
            stage = Path(temporary)
            svg, pdf = stage / output.name, stage / output.with_suffix(".pdf").name
            title = "Adaptive reserved validation: five paired seeds"
            fig.savefig(svg, metadata={"Date": None, "Title": title})
            fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None, "Title": title})
            provenance = {"schema_version": 1, "kind": "adaptive_reserved_validation_comparison",
                          "input_report_name": report_dir.name, "input_sha256": inputs,
                          "plot_script_sha256": digest(__file__), "python_version": platform.python_version(),
                          "matplotlib_version": matplotlib.__version__, **data,
                          "output_sha256": {path.name: digest(path) for path in (svg, pdf)}}
            sidecar = stage / output.with_suffix(".json").name
            sidecar.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
            for source, target in zip((svg, pdf, sidecar), targets, strict=True):
                source.rename(target)
        plt.close(fig)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot_adaptive_validation(args.report_dir, args.output)
