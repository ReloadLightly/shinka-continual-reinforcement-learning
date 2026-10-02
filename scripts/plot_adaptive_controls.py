"""Plot observed fixed-control adaptive objective scores from a verified export."""

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

CONTROLS = ("identity", "arithmetic", "focus", "static_shinka11", "static_random24")
LABELS = ("Identity", "Arithmetic", "FocusGA", "Static\nShinka 11", "Static\nRandom 24")
SCORES = ("combined_score", "active_score", "previous_score")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_data(report_dir):
    checksums = json.loads((report_dir / "checksums.json").read_text())
    inputs = {"checksums.json": digest(report_dir / "checksums.json")}

    def read(relative):
        path = (report_dir / relative).resolve()
        require(path.is_relative_to(report_dir), "Artifact escapes the report directory")
        value = digest(path)
        require(checksums["published_sha256"].get(relative) == value,
                f"Published artifact changed: {relative}")
        inputs[relative] = value
        return json.loads(path.read_text())

    summary, plan = read("summary.json"), read("raw/plan.json")
    require(summary["status"] == "complete" and summary["profile"] == plan["profile"],
            "Require a complete exported control comparison")
    require(plan["objective_weights"] == {"active": .5, "previous": .5}
            and plan["objective_version"] == "adaptive-active-previous-v1",
            "Unexpected adaptive selection objective")
    seeds = plan["profile"]["seeds"]
    require(seeds == [4001, 4002, 4003], "Unexpected development seed partition")
    plotted = []
    for identity, label in zip(CONTROLS, LABELS, strict=True):
        request = summary.get("control_requests", {}).get(identity)
        require(request is not None and request["candidate"]["id"] == identity
                and request["status"] == "complete", f"Missing or incomplete control: {identity}")
        cached = read(f"raw/{request['cache_origin']}/summary.json")
        require(cached["status"] == "complete" and cached["candidate"] == request["candidate"]
                and cached["aggregate"] == request["aggregate"], "Cached control identity differs")
        trials = cached["trials"]
        require([row["seed"] for row in trials] == seeds, "Incomplete or reordered control trials")
        scores = {}
        for key in SCORES:
            values = [row["score"][key] for row in trials]
            require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                        for value in values), "Invalid adaptive objective component")
            scores[key] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "n": 3}
            require(scores[key] == cached["aggregate"]["scores"][key], "Score aggregation differs")
        for row in trials:
            value = row["score"]
            require(math.isclose(value["combined_score"], .5 * value["active_score"]
                                 + .5 * value["previous_score"], rel_tol=0, abs_tol=1e-12),
                    "Combined objective differs from its components")
        plotted.append({"control": identity, "label": label, "scores": scores,
                        "seeds": [{"seed": row["seed"], **{key: row["score"][key] for key in SCORES}}
                                  for row in trials]})
    return {"profile": plan["profile"], "objective": plan["objective_version"],
            "objective_weights": plan["objective_weights"], "controls": plotted}, inputs


def plot_adaptive_controls(report_dir, output):
    report_dir, output = Path(report_dir).resolve(), Path(output).resolve()
    targets = [output, output.with_suffix(".pdf"), output.with_suffix(".json")]
    require(output.suffix == ".svg" and not any(path.exists() for path in targets),
            "Use a fresh .svg output with fresh PDF and JSON sidecars")
    data, inputs = load_data(report_dir)
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
             "axes.linewidth": .65, "axes.edgecolor": "#87909A", "axes.labelcolor": "#25313B",
             "xtick.color": "#55616C", "ytick.color": "#55616C", "xtick.labelsize": 8,
             "ytick.labelsize": 8, "grid.color": "#E1E5E8", "grid.linewidth": .55,
             "svg.fonttype": "none", "svg.hashsalt": "shinka-crl-adaptive-controls-v1",
             "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2), sharey=True)
        fig.subplots_adjust(left=.065, right=.98, bottom=.29, top=.77, wspace=.16)
        upper = max(1., max(s["mean"] + s["sample_sd"] for row in data["controls"]
                            for s in row["scores"].values())) + .025
        lower = min(0., min(s["mean"] - s["sample_sd"] for row in data["controls"]
                            for s in row["scores"].values())) - .025
        for ax in axes:
            ax.set_axisbelow(True)
            ax.grid(axis="y")
            for edge in ("top", "right"):
                ax.spines[edge].set_visible(False)
            ax.set(xticks=range(5), xticklabels=LABELS, xlim=(-.45, 4.45), ylim=(lower, upper))
        axes[0].set_ylabel("Normalized development return")
        axes[0].set_title("A  Combined selection objective", loc="left", pad=27, fontsize=10,
                          fontweight="semibold", color="#25313B")
        axes[1].set_title("B  Active and previous-task components", loc="left", pad=27, fontsize=10,
                          fontweight="semibold", color="#25313B")
        for index, row in enumerate(data["controls"]):
            combined = row["scores"]["combined_score"]
            axes[0].errorbar(index, combined["mean"], yerr=combined["sample_sd"], fmt="D",
                             markersize=5, capsize=3, linewidth=1.1, color="#188577")
            paired = [row["scores"][key]["mean"] for key in ("active_score", "previous_score")]
            axes[1].plot([index - .12, index + .12], paired, color="#A8B0B7", linewidth=.8)
            for key, offset, color, marker, label in (
                ("active_score", -.12, "#2263A5", "o", "Active task"),
                ("previous_score", .12, "#C65F31", "s", "Previous task"),
            ):
                component = row["scores"][key]
                axes[1].errorbar(index + offset, component["mean"], yerr=component["sample_sd"],
                                 fmt=marker, markersize=4.5, capsize=2.5, linewidth=1., color=color,
                                 label=label if index == 0 else None)
        axes[1].legend(loc="lower left", bbox_to_anchor=(-.02, 1.01), frameon=False, ncol=2,
                       fontsize=7.5, handlelength=1.5, columnspacing=1.7)
        axes[0].text(0, 1.04, "Equal weight on both objective components", transform=axes[0].transAxes,
                     fontsize=7.5, color="#65717D")
        fig.text(.065, .105, "Points: means over three shared development seeds · error bars: sample "
                 "standard deviation · five fixed controls", fontsize=7.5, color="#65717D")
        fig.text(.065, .045, "Previous-task return comes from fresh phase-checkpoint episodes. "
                 "This comparison contains no evolved adaptive programs.", fontsize=7.5, color="#65717D")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".adaptive-control-figure-", dir=output.parent) as temporary:
            stage = Path(temporary)
            svg, pdf = stage / output.name, stage / output.with_suffix(".pdf").name
            fig.savefig(svg, metadata={"Date": None, "Title": "Adaptive objective fixed controls"})
            fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None,
                                       "Title": "Adaptive objective fixed controls"})
            provenance = {"schema_version": 1, "kind": "adaptive_fixed_control_comparison",
                          "input_sha256": inputs, "plot_script_sha256": digest(__file__), **data,
                          "output_sha256": {path.name: digest(path) for path in (svg, pdf)}}
            sidecar = stage / output.with_suffix(".json").name
            sidecar.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
            for source, target in zip((svg, pdf, sidecar), targets, strict=True):
                source.rename(target)
        plt.close(fig)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/adaptive-controls-20261003"))
    parser.add_argument("--output", type=Path, default=Path("figures/adaptive-controls-20261003.svg"))
    arguments = parser.parse_args()
    plot_adaptive_controls(arguments.report_dir, arguments.output)
