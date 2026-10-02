"""Draw publication-ready learning curves from validated exported pilot evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {"ga": "#2263A5", "es": "#188577", "ppo": "#C65F31"}
METHODS = ("ga", "es", "ppo")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def plot_pilot(report_dir: Path, output: Path) -> dict:
    report_dir, output = report_dir.resolve(), output.resolve()
    require(output.suffix == ".svg", "Output must have an .svg extension")
    outputs = [output, output.with_suffix(".pdf"), output.with_suffix(".json")]
    require(not any(path.exists() for path in outputs), "Refusing to overwrite figure artifacts")
    summary_path, plan_path = report_dir / "summary.json", report_dir / "raw/plan.json"
    summary, plan = read_json(summary_path), read_json(plan_path)
    checksums = read_json(report_dir / "checksums.json")
    require(digest(plan_path) == checksums["raw/plan.json"]["exported_sha256"],
            "Exported protocol changed")
    require(summary.get("status") == "complete", "Publication figure requires a complete pilot")
    require(summary.get("validation", {}).get("all_planned_trials") is True,
            "Pilot report does not contain all planned trials")
    inputs = {"summary.json": digest(summary_path), "raw/plan.json": digest(plan_path),
              "checksums.json": digest(report_dir / "checksums.json")}
    groups = defaultdict(list)
    profiles = {}
    for row in summary["rows"]:
        profile = plan["profiles"][row["profile"]]
        condition, method = row["condition"], row["method"]
        require(condition in ("stationary", "switching") and method in METHODS,
                "Unexpected pilot condition or method")
        if condition in profiles:
            require(profiles[condition] == profile, "Do not mix budgets within a panel")
        profiles[condition] = profile
        relative = Path("raw") / row["training_path"] / "training_metrics.json"
        path = (report_dir / relative).resolve()
        require(path.is_relative_to(report_dir), "Training path escapes report directory")
        require(digest(path) == checksums[str(relative)]["exported_sha256"],
                f"Exported training metrics changed: {relative}")
        inputs[str(relative)] = digest(path)
        records = read_json(path)
        budget = profile["ppo"] if method == "ppo" else profile["ne"]
        count = budget["num_updates" if method == "ppo" else "num_generations"]
        per_update = (budget["num_envs"] * budget["num_steps"] if method == "ppo" else
                      budget["pop_size"] * budget["num_evals"] * profile["episode_length"])
        require(len(records) == count, "Unexpected number of training records")
        rewards = []
        for index, record in enumerate(records):
            task = (index // budget["task_interval"]) % profile["num_tasks"]
            require(record["generation"] == index and record["task"] == task,
                    "Unexpected generation or task sequence")
            rewards.append(record[f"centroid_task{task}"])
        y = np.asarray(rewards, dtype=float)
        require(bool(np.all(np.isfinite(y))) and bool(np.all((0 <= y) & (y <= 500))),
                "Invalid return curve")
        x = np.arange(1, count + 1, dtype=float) * per_update / 1e6
        groups[(condition, method)].append((row["seed"], x, y))
    require(set(profiles) == {"stationary", "switching"}, "Both pilot conditions are required")
    require(all(len(groups[(condition, method)]) == 3
                for condition in profiles for method in METHODS),
            "Expected three seeds for every method and condition")

    style = {
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 10.5, "axes.titleweight": "medium", "axes.linewidth": 0.65,
        "axes.edgecolor": "#87909A", "axes.labelcolor": "#25313B",
        "xtick.color": "#55616C", "ytick.color": "#55616C", "xtick.labelsize": 8,
        "ytick.labelsize": 8, "xtick.major.size": 3, "ytick.major.size": 3,
        "grid.color": "#E1E5E8", "grid.linewidth": 0.55, "svg.fonttype": "none",
        "svg.hashsalt": "shinka-crl-pilot-v1", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.55), sharey=True)
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.79, wspace=0.15)
        for ax, condition, title in zip(
                axes, ("stationary", "switching"),
                ("(a)  Stationary control", "(b)  Alternating observation offsets")):
            profile = profiles[condition]
            ax.set_title(title, loc="left", pad=24)
            ax.set_axisbelow(True)
            ax.grid(axis="y")
            for edge in ("top", "right"):
                ax.spines[edge].set_visible(False)
            total_steps = max(curve[1][-1] for method in METHODS
                              for curve in groups[(condition, method)])
            if condition == "switching":
                interval = total_steps / profile["num_phases"]
                for phase in range(profile["num_phases"]):
                    left, right = phase * interval, (phase + 1) * interval
                    if phase % 2:
                        ax.axvspan(left, right, color="#EBEEF0", alpha=0.7, linewidth=0, zorder=0)
                    if phase:
                        ax.axvline(left, color="#A7AFB7", linestyle=(0, (2, 3)),
                                   linewidth=0.7, zorder=1)
                    ax.text((left + right) / 2, 1.025, "AB"[phase % 2],
                            transform=ax.get_xaxis_transform(), color="#6B7580",
                            ha="center", va="bottom", fontsize=8)
            for method in METHODS:
                trials = sorted(groups[(condition, method)], key=lambda trial: trial[0])
                x = trials[0][1]
                require(all(np.array_equal(trial[1], x) for trial in trials),
                        "Trial clocks within a method differ")
                for _, _, y in trials:
                    ax.plot(x, y, color=COLORS[method], linewidth=0.65, alpha=0.26, zorder=2)
                mean = np.mean(np.stack([trial[2] for trial in trials]), axis=0)
                ax.plot(x, mean, color=COLORS[method], linewidth=1.85, zorder=3,
                        solid_capstyle="round")
            ax.set(xlim=(0, total_steps), ylim=(0, 515),
                   xlabel="Nominal training steps (millions)")
            ax.set_yticks(np.arange(0, 501, 100))
            ax.set_xticks(np.linspace(0, total_steps, 5))
            ax.set_xticklabels([f"{value:.2f}".rstrip("0").rstrip(".")
                                for value in np.linspace(0, total_steps, 5)])
        axes[0].set_ylabel("Active-task centroid return")
        handles = [Line2D([0], [0], color=COLORS[method], linewidth=2.3,
                          label=method.upper()) for method in METHODS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.50, 1.0),
                   frameon=False, ncol=3, handlelength=2.2, columnspacing=2.6)
        fig.text(0.075, 0.028, "Development pilot · three seeds per method · thin: individual trials; "
                 "bold: arithmetic mean · evaluation cost excluded", fontsize=7.6, color="#65717D")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".pilot-figure-", dir=output.parent) as temporary:
            staging = Path(temporary)
            svg, pdf = staging / output.name, staging / output.with_suffix(".pdf").name
            fig.savefig(svg, format="svg", metadata={"Date": None, "Title": "CartPole pilot"})
            fig.savefig(pdf, format="pdf", metadata={"CreationDate": None, "ModDate": None,
                                                    "Title": "CartPole pilot"})
            provenance = {
                "schema_version": 1, "kind": "development_pilot_learning_curves",
                "plot_script": "scripts/plot_pilot.py", "plot_script_sha256": digest(Path(__file__)),
                "input_report": report_dir.name, "input_sha256": inputs,
                "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
                "methods": list(METHODS), "seeds_per_method_condition": 3,
                "curve_clock": "(zero_based_generation + 1) * nominal_steps_per_update",
                "aggregation": "Arithmetic mean across three training seeds; no confidence bands",
                "smoothing": None, "outputs_sha256": {svg.name: digest(svg), pdf.name: digest(pdf)},
            }
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
        provenance = plot_pilot(args.report_dir, args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Pilot plot rejected: {exc}\n")
    print(json.dumps({"outputs": list(provenance["outputs_sha256"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
