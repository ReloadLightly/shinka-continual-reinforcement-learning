"""Plot the complete paper-budget development reference and observed phase times."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def plot_reference(report_dir: Path, output: Path) -> dict:
    report_dir, output = report_dir.resolve(), output.resolve()
    outputs = [output, output.with_suffix(".pdf"), output.with_suffix(".json")]
    require(output.suffix == ".svg" and not any(p.exists() for p in outputs),
            "Choose a fresh .svg output path")
    checksums = read(report_dir / "checksums.json")["published_sha256"]
    for relative, expected in checksums.items():
        path = (report_dir / relative).resolve()
        require(path.is_relative_to(report_dir) and digest(path) == expected,
                f"Changed published evidence: {relative}")
    summary, plan = read(report_dir / "summary.json"), read(report_dir / "raw/plan.json")
    profile = plan["profile"]
    require(summary["status"] == "complete" and profile["name"] == "paper-cartpole-timing",
            "Requires a complete paper-budget development reference")
    require(profile["num_phases"] == 20 and profile["num_tasks"] == 2
            and profile["ne"]["task_interval"] == 200
            and profile["ne"]["num_generations"] == 4000
            and profile["ne"]["pop_size"] == 512 and profile["ne"]["num_evals"] == 3
            and profile["episode_length"] == 500 and profile["eval_episodes"] == 10
            and summary["seed"] == 1001 and summary["trial"] == 1002,
            "Unexpected reference protocol")
    records = read(report_dir / "raw/training/training_metrics.json")
    require(len(records) == 4000, "Incomplete trajectory")
    for i, row in enumerate(records):
        require(row["generation"] == i and row["task"] == (i // 200) % 2,
                "Unexpected task or generation")
    curves = {f"task_{task}": [row[f"centroid_task{task}"] for row in records]
              for task in (0, 1)}
    values = np.asarray(list(curves.values()), dtype=float)
    require(np.all(np.isfinite(values)) and np.all((values >= 0) & (values <= 500)),
            "Invalid centroid return")
    active = values[np.array([r["task"] for r in records]), np.arange(4000)]
    require(abs(float(active.mean() / 500) - summary["active_score"]["normalized_score"]) < 1e-10,
            "Active score differs from trajectory")
    times = summary["training"]["phase_timings"]
    events = [json.loads(line) for line in
              (report_dir / "raw/training/phase-events.jsonl").read_text().splitlines()]
    require(len(times) == len(events) == 20, "Incomplete phase timing")
    previous = 0
    for i, (phase, event) in enumerate(zip(times, events)):
        require(phase["phase"] == i and event["completed_generations"] == (i + 1) * 200
                and phase["boundary_elapsed_seconds"] == event["elapsed_seconds"]
                and phase["observed_wall_seconds"] == event["elapsed_seconds"] - previous,
                "Phase intervals differ from timing journal")
        previous = event["elapsed_seconds"]
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
             "axes.titlesize": 10.5, "axes.titleweight": "medium", "axes.linewidth": .65,
             "axes.edgecolor": "#87909A", "axes.labelcolor": "#25313B",
             "xtick.color": "#55616C", "ytick.color": "#55616C", "xtick.labelsize": 8,
             "ytick.labelsize": 8, "grid.color": "#E1E5E8", "grid.linewidth": .55,
             "svg.fonttype": "none", "svg.hashsalt": "shinka-crl-reference-v1",
             "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.9), gridspec_kw={"width_ratios": [2.25, 1]})
        fig.subplots_adjust(left=.06, right=.985, bottom=.23, top=.79, wspace=.26)
        for ax in axes:
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            for edge in ("top", "right"):
                ax.spines[edge].set_visible(False)
        ax = axes[0]
        for phase in range(20):
            if phase % 2:
                ax.axvspan(phase, phase + 1, color="#EBEEF0", alpha=.75, linewidth=0)
            ax.text(phase + .5, 1.025, "AB"[phase % 2], transform=ax.get_xaxis_transform(),
                    color="#65717D", ha="center", fontsize=7)
        x = np.arange(1, 4001) / 200
        for task, color in enumerate(("#2263A5", "#C65F31")):
            ax.plot(x, values[task], color=color, linewidth=.65, alpha=.82,
                    label=f"Centroid on task {'AB'[task]}")
        ax.set(xlim=(0, 20), ylim=(0, 515), xlabel="Completed training phases",
               ylabel="Return on each task")
        ax.set_xticks(np.arange(0, 21, 2))
        ax.set_title("A   Learning across 20 task phases", loc="left", pad=26)
        axes[1].bar(np.arange(1, 21), [t["observed_wall_seconds"] for t in times],
                    color=["#2263A5"] + ["#9AABB9"] * 19, width=.72)
        axes[1].set(xlim=(.25, 20.75), xlabel="Training phase", ylabel="Observed wall time (s)")
        axes[1].set_xticks([1, 5, 10, 15, 20])
        axes[1].set_title("B   Phase timing", loc="left", pad=26)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(.38, 1), ncol=2, frameon=False)
        fig.text(.06, .055, "One development seed · default GA · population 512 · 200 generations/phase · unsmoothed curves",
                 fontsize=7.6, color="#65717D")
        fig.text(.06, .012, "Task labels show the active training task; both tasks are evaluated. Phase 1 includes startup/JIT; intervals include checkpoint I/O.",
                 fontsize=7.1, color="#65717D")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".reference-figure-", dir=output.parent) as temp:
            svg, pdf, sidecar = [Path(temp) / p.name for p in outputs]
            fig.savefig(svg, metadata={"Date": None, "Title": "Paper-budget GA development reference"})
            fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None,
                                       "Title": "Paper-budget GA development reference"})
            provenance = {"schema_version": 1, "kind": "development_reference",
                          "plot_script_sha256": digest(Path(__file__)),
                          "input_report": report_dir.name,
                          "input_sha256": {**checksums, "checksums.json": digest(report_dir / "checksums.json")},
                          "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
                          "curves": curves, "phase_timings": times, "smoothing": None,
                          "outputs_sha256": {p.name: digest(p) for p in (svg, pdf)}}
            sidecar.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
            require(not any(p.exists() for p in outputs), "Refusing to overwrite artifacts")
            for source, destination in zip((svg, pdf, sidecar), outputs):
                source.rename(destination)
        plt.close(fig)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plot_reference(args.report_dir, args.output)
    print(json.dumps({"outputs": list(result["outputs_sha256"])}))


if __name__ == "__main__":
    main()
