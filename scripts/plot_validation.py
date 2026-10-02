"""Plot paired finalist scores and individual signed switch losses from a validated export."""

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

COLORS = {"default": "#65717D", "shinka": "#2263A5", "random": "#C65F31", "shared": "#188577"}
SWITCH_LABELS = ("1: A → B", "2: B → A", "3: A → B")
SWITCH_MARKERS = ("o", "s", "^")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def finite_return(value, cap: int) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= cap,
            "Invalid checkpoint return")
    return float(value)


def candidate_style(candidate: dict) -> tuple[str, str]:
    memberships = candidate["memberships"]
    require(bool(memberships), "Finalist lacks its selection membership")
    if any(member["arm"] == "default" for member in memberships):
        return "Default", COLORS["default"]
    labels, arms = [], set()
    for member in memberships:
        arm = member["arm"]
        require(arm in {"shinka", "random"} and type(member["index"]) is int,
                "Unexpected finalist membership")
        label = f"{'Shinka' if arm == 'shinka' else 'Random'} {member['index']}"
        if label not in labels:
            labels.append(label)
        arms.add(arm)
    return " / ".join(labels), COLORS[next(iter(arms))] if len(arms) == 1 else COLORS["shared"]


def load_plot_data(report_dir: Path) -> tuple[dict, dict]:
    """Reconstruct plotted values from receipt-checked curves and episode returns."""
    report_dir = report_dir.resolve()
    summary, checksums = (read_json(report_dir / name) for name in ("summary.json", "checksums.json"))
    require(summary.get("status") == "complete" and summary.get("selection_complete") is True,
            "Finish the complete reserved validation comparison before plotting")
    require(all(summary.get("validation", {}).get(key) is True for key in
                ("scores_rederived", "posthoc_metrics_rederived", "switch_differences_rederived",
                 "artifact_receipts_match", "task_vectors_match")), "Require a validated evidence export")
    inputs = {name: digest(report_dir / name) for name in ("summary.json", "checksums.json")}

    def artifact(relative: str) -> Path:
        require(not Path(relative).is_absolute(), "Artifact paths must be relative")
        path = (report_dir / relative).resolve()
        require(path.is_relative_to(report_dir), "Artifact escapes report directory")
        receipt = checksums.get(relative, {})
        value = digest(path)
        require(receipt.get("exported") is True and value == receipt.get("exported_sha256"),
                f"Exported artifact changed: {relative}")
        inputs[relative] = value
        return path

    plan = read_json(artifact("raw/plan.json"))
    frozen_path = artifact("raw/finalists/manifest.json")
    frozen = read_json(frozen_path)
    require(checksums["raw/finalists/manifest.json"]["original_sha256"]
            == plan["finalist_manifest_sha256"] == summary["finalist_manifest_sha256"],
            "Finalist selection identity differs")
    profile = summary["profile"]
    require(profile == plan["profile"] == frozen["profile"], "Do not mix validation protocols")
    require(profile["name"] == "cartpole-validation" and profile["seeds"] == list(range(2001, 2006))
            and profile["num_phases"] == 4 and profile["num_tasks"] == 2
            and profile["episode_length"] == 500 and profile["eval_episodes"] == 10
            and profile["ne"] == {"num_generations": 320, "task_interval": 80,
                                  "pop_size": 64, "num_evals": 3},
            "Unexpected reserved validation protocol")
    candidates, seeds = frozen["candidates"], profile["seeds"]
    ids = [candidate["id"] for candidate in candidates]
    require(2 <= len(candidates) <= 5 and len(set(ids)) == len(ids)
            and ids == plan["candidate_ids"], "Finalist order or identity differs")
    expected = [(identifier, seed) for seed in seeds for identifier in ids]
    rows = summary["rows"]
    require([(row["candidate_id"], row["seed"]) for row in rows] == expected
            and summary["completed_trials"] == summary["planned_trials"] == len(expected),
            "Missing, duplicate or reordered validation trials")
    require([group["candidate_id"] for group in summary["groups"]] == ids,
            "Missing or reordered finalist aggregates")
    plotted = []
    for candidate, group in zip(candidates, summary["groups"]):
        label, color = candidate_style(candidate)
        program_path = "raw/finalists/" + candidate["program_path"]
        require(digest(artifact(program_path)) == candidate["program_sha256"],
                "Frozen finalist program changed")
        require(group["settings"] == candidate["settings"]
                and group["program_sha256"] == candidate["program_sha256"]
                and group["memberships"] == candidate["memberships"]
                and group["completed_seeds"] == len(seeds), "Finalist aggregate identity differs")
        trials = []
        for row in [item for item in rows if item["candidate_id"] == candidate["id"]]:
            seed = row["seed"]
            require(row["trial"] == seed + 1 and row["eval_seed"] == 900000 + seed,
                    "Validation trial or evaluation seed differs")
            training_root, analysis_root = ("raw/" + row[key] for key in ("training_path", "analysis_path"))
            manifest = read_json(artifact(training_root + "/manifest.json"))
            require(manifest["profile"] == profile and manifest["method"] == "ga"
                    and manifest["seed"] == seed and manifest["trial"] == seed + 1
                    and manifest["ga_settings"] == candidate["settings"], "Training identity differs")
            records = read_json(artifact(training_root + "/training_metrics.json"))
            require(len(records) == 320, "Unexpected number of training checkpoints")
            active = []
            for index, record in enumerate(records):
                task = index // 80 % 2
                require(record["generation"] == index and record["task"] == task,
                        "Unexpected generation or task schedule")
                active.append(finite_return(record[f"centroid_task{task}"], 500))
            score = sum(active) / len(active) / 500
            require(score == row["normalized_score"], "Active score differs from raw training curve")
            evaluation = read_json(artifact(analysis_root + "/evaluation.json"))
            require(all(evaluation.get(key) == value for key, value in {
                "method": "ga", "env": "CartPole-v1", "pop_size": 64,
                "trial": seed + 1, "num_tasks": 4, "episodes": 10,
                "eval_seed": 900000 + seed}.items()), "Fresh evaluation identity differs")
            entries = [entry for entry in evaluation["per_task"] if entry["source"] == "centroid"]
            require(len(entries) == 4 and {e["task_idx"] for e in entries} == set(range(4)),
                    "Missing or duplicated centroid phase evaluations")
            by_phase = {entry["task_idx"]: entry for entry in entries}

            def mean_returns(phase, key):
                values = by_phase[phase].get(key)
                require(isinstance(values, list) and len(values) == 10,
                        "Expected ten fresh checkpoint episodes")
                return sum(finite_return(value, 500) for value in values) / 10

            own = [mean_returns(phase, "returns") for phase in range(4)]
            previous = [None] + [mean_returns(phase, "prev_returns") for phase in range(1, 4)]
            phases = row["analysis"]["phase_returns"]
            require(len(phases) == 4 and all(
                p["phase"] == i and p["task"] == i % 2 and p["own_mean"] == own[i]
                and p["previous_mean"] == previous[i] for i, p in enumerate(phases)),
                "Phase return summary differs from raw episodes")
            switches = [{"from_phase": i, "to_phase": i + 1, "task": i % 2,
                         "before": own[i], "after": previous[i + 1],
                         "forgetting": own[i] - previous[i + 1]} for i in range(3)]
            require(switches == row["switches"], "Switch differences differ from raw episodes")
            forgetting = sum(switch["forgetting"] for switch in switches) / 3
            require(forgetting == row["analysis"]["metrics"]["forgetting"],
                    "Mean forgetting differs from the individual switches")
            trials.append({"seed": seed, "trial": seed + 1, "active_score": score,
                           "switches": switches, "mean_forgetting": forgetting})
        scores = [trial["active_score"] for trial in trials]
        mean, sd = statistics.mean(scores), statistics.stdev(scores)
        require(group["score"] == {"mean": mean, "sample_sd": sd, "n": 5},
                "Five-seed active-score aggregation differs")
        forgetting = [trial["mean_forgetting"] for trial in trials]
        require(group["metrics"]["forgetting"] == {
            "mean": statistics.mean(forgetting), "sample_sd": statistics.stdev(forgetting), "n": 5},
            "Five-seed forgetting aggregation differs")
        plotted.append({"candidate_id": candidate["id"], "label": label, "color": color,
                        "memberships": candidate["memberships"], "settings": candidate["settings"],
                        "program_sha256": candidate["program_sha256"], "trials": trials,
                        "mean_active_score": mean, "sample_sd_active_score": sd})
    return {"profile": profile, "seeds": seeds, "candidates": plotted,
            "synthetic": summary.get("synthetic", False) is True}, inputs


def plot_validation(report_dir: Path, output: Path) -> dict:
    report_dir, output = report_dir.resolve(), output.resolve()
    require(output.suffix == ".svg", "Output must have an .svg extension")
    outputs = [output, output.with_suffix(".pdf"), output.with_suffix(".json")]
    require(not any(path.exists() for path in outputs), "Refusing to overwrite figure artifacts")
    data, inputs = load_plot_data(report_dir)
    candidates = data["candidates"]
    count = len(candidates)
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
             "axes.linewidth": .65, "axes.edgecolor": "#87909A", "axes.labelcolor": "#25313B",
             "xtick.color": "#55616C", "ytick.color": "#55616C", "xtick.labelsize": 8,
             "ytick.labelsize": 8, "grid.color": "#E1E5E8", "grid.linewidth": .55,
             "svg.fonttype": "none", "svg.hashsalt": "shinka-crl-validation-v1", "pdf.fonttype": 42,
             "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(style):
        fig, (active_ax, forgetting_ax) = plt.subplots(
            1, 2, figsize=(11.8, 4.5), gridspec_kw={"width_ratios": [1, 1.55]})
        fig.subplots_adjust(left=.065, right=.985, bottom=.25, top=.75, wspace=.28)
        for ax in (active_ax, forgetting_ax):
            ax.set_axisbelow(True)
            ax.grid(axis="y")
            for edge in ("top", "right"):
                ax.spines[edge].set_visible(False)
            ax.set_xticks(range(count))
            ax.set_xticklabels([candidate["label"].replace(" / ", "\n/ ").replace(" ", "\n", 1)
                                for candidate in candidates])
            ax.set_xlim(-.48, count - .52)
        active_ax.set_title("A  Paired validation scores", loc="left", pad=34, fontsize=10,
                            fontweight="semibold", color="#25313B")
        forgetting_ax.set_title("B  Signed loss at every task switch", loc="left", pad=34,
                                fontsize=10, fontweight="semibold", color="#25313B")
        offsets = [(index - 2) * .025 for index in range(5)]
        for index, offset in enumerate(offsets):
            values = [candidate["trials"][index]["active_score"] for candidate in candidates]
            active_ax.plot([x + offset for x in range(count)], values, color="#9AA5AD",
                           linewidth=.75, alpha=.48, zorder=2)
        for x, candidate in enumerate(candidates):
            values = [trial["active_score"] for trial in candidate["trials"]]
            active_ax.scatter([x + offset for offset in offsets], values, s=25,
                              color=candidate["color"], alpha=.7, edgecolors="white",
                              linewidths=.35, zorder=3)
            active_ax.scatter([x], [candidate["mean_active_score"]], s=57, marker="D",
                              color=candidate["color"], edgecolors="white", linewidths=.7, zorder=4)
            for trial_index, trial in enumerate(candidate["trials"]):
                xs = [x + (switch - 1) * .235 + offsets[trial_index] for switch in range(3)]
                losses = [switch["forgetting"] for switch in trial["switches"]]
                forgetting_ax.plot(xs, losses, color="#A8B0B7", linewidth=.55, alpha=.45, zorder=2)
                for switch, (point_x, loss) in enumerate(zip(xs, losses)):
                    forgetting_ax.scatter([point_x], [loss], s=23, marker=SWITCH_MARKERS[switch],
                                          color=candidate["color"], alpha=.7,
                                          edgecolors="white", linewidths=.3, zorder=3)
        active_ax.set(ylim=(0, 1.025), ylabel="Active-task score, J")
        active_ax.set_yticks([0, .2, .4, .6, .8, 1])
        forgetting_ax.axhline(0, color="#65717D", linewidth=1.0, zorder=1)
        forgetting_ax.set(ylim=(-525, 525), ylabel="Forgetting: return before − return after")
        forgetting_ax.set_yticks([-500, -250, 0, 250, 500])
        forgetting_ax.legend(handles=[Line2D([0], [0], marker=marker, linestyle="none",
                                             color="#65717D", markersize=5, label=label)
                                      for marker, label in zip(SWITCH_MARKERS, SWITCH_LABELS)],
                              loc="lower center", bbox_to_anchor=(.5, 1.01), frameon=False,
                              ncol=3, handletextpad=.35, columnspacing=1.3, fontsize=7.5)
        fig.text(.065, .073, "Left: points are five validation seeds; diamonds are means; thin lines "
                 "connect the same seed across configurations.", fontsize=7.5, color="#65717D")
        fig.text(.065, .03, "Right: positive = loss; negative = improvement. Each seed contributes "
                 "three switches; their mean can conceal individual losses.", fontsize=7.5, color="#65717D")
        if data["synthetic"]:
            fig.text(.065, .95, "SYNTHETIC TEST FIXTURE — NOT EXPERIMENTAL RESULTS",
                     fontsize=8, color="#C65F31")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".validation-figure-", dir=output.parent) as temporary:
            staging = Path(temporary)
            svg, pdf = staging / output.name, staging / output.with_suffix(".pdf").name
            fig.savefig(svg, metadata={"Date": None, "Title": "Static finalist validation"})
            fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None,
                                      "Title": "Static finalist validation"})
            provenance = {"schema_version": 1, "kind": "paired_static_finalist_validation",
                          "plot_script": "scripts/plot_validation.py",
                          "plot_script_sha256": digest(Path(__file__)),
                          "input_report": report_dir.name, "input_sha256": inputs,
                          "matplotlib_version": matplotlib.__version__, **data,
                          "aggregation": "Arithmetic mean of five paired training seeds; no confidence interval or smoothing",
                          "forgetting_definition": "own return at phase i minus previous-task return at phase i+1, each from ten fresh centroid episodes",
                          "pairing": "Seed labels pair task draws and initialization seeds; they do not imply identical post-training policies",
                          "outputs_sha256": {svg.name: digest(svg), pdf.name: digest(pdf)}}
            sidecar = staging / output.with_suffix(".json").name
            sidecar.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
            require(not any(path.exists() for path in outputs), "Refusing to overwrite figure artifacts")
            for source, destination in zip((svg, pdf, sidecar), outputs):
                source.rename(destination)
        plt.close(fig)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = plot_validation(args.report_dir, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Validation plot rejected: {exc}\n")
    print(json.dumps({"outputs": list(result["outputs_sha256"]),
                      "configurations": len(result["candidates"]), "synthetic": result["synthetic"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
