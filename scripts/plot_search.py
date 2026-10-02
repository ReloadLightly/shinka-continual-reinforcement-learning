"""Plot observed search trajectories from a validated, matched evidence export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = {"shinka": "#2263A5", "random": "#C65F31"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def plot_search(report_dir: Path, output: Path) -> dict:
    report_dir, output = report_dir.resolve(), output.resolve()
    require(output.suffix == ".svg", "Output must have an .svg extension")
    outputs = [output, output.with_suffix(".pdf"), output.with_suffix(".json")]
    require(not any(path.exists() for path in outputs), "Refusing to overwrite figure artifacts")
    summary = read_json(report_dir / "summary.json")
    checksums = read_json(report_dir / "checksums.json")
    require(summary.get("status") == "complete", "Finish the search stage before plotting")
    require(summary.get("validation", {}).get("scores_rederived_from_raw_curves") is True,
            "Require an independently validated score export")
    inputs = {name: digest(report_dir / name) for name in ("summary.json", "checksums.json")}

    def artifact(relative: str) -> Path:
        path = (report_dir / relative).resolve()
        require(path.is_relative_to(report_dir), "Artifact escapes report directory")
        value = digest(path)
        require(value == checksums[relative]["exported_sha256"], f"Artifact changed: {relative}")
        inputs[relative] = value
        return path

    plan = read_json(artifact("raw/plan.json"))
    require(summary["profile"] == plan["profile"], "Protocol mismatch")
    profile = summary["profile"]
    initial, arms = [], {"shinka": [], "random": []}
    for row in sorted(summary["rows"], key=lambda item: (item["arm"], item["index"])):
        require(row["arm"] in arms, "Unknown search arm")
        if not row["correct"]:
            continue
        program = "raw/" + row["program_path"]
        artifact(program)
        require(checksums[program]["original_sha256"] == row["program_sha256"],
                "Program identity mismatch")
        root = (f"raw/shinka/gen_{row['index']}/results" if row["arm"] == "shinka"
                else f"raw/random/candidate_{row['index']:03d}")
        metrics = read_json(artifact(root + "/metrics.json"))
        require(read_json(artifact(root + "/correct.json"))["correct"] is True,
                "Do not plot a failed evaluation as a return")
        seeds = metrics["private"]["seed_results"]
        require([seed["seed"] for seed in seeds] == profile["seeds"], "Seed partition mismatch")
        scores = [seed["normalized_score"] for seed in seeds]
        require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                    for value in scores), "Invalid development score")
        score = statistics.mean(scores)
        require(score == row["combined_score"] == metrics["combined_score"], "Score mismatch")
        settings = {key: metrics["public"][key] for key in ("sigma", "elite_ratio")}
        key = [struct.unpack("!f", struct.pack("!f", settings["sigma"]))[0],
               max(1, int(profile["ne"]["pop_size"] * settings["elite_ratio"]))]
        require(settings == row["settings"] and key == row["effective_key"],
                "Effective configuration mismatch")
        point = {"index": row["index"], "score": score, "effective_key": key,
                 "program_sha256": row["program_sha256"]}
        if row["arm"] == "shinka" and row["index"] == 0:
            initial.append(point)
        else:
            arms[row["arm"]].append(point)
    require(len(initial) == 1, "Require one shared initial configuration")
    default = initial[0]
    seen = {tuple(default["effective_key"])}
    distinct = []
    for point in arms["shinka"]:
        key = tuple(point["effective_key"])
        if key not in seen:
            distinct.append(point)
            seen.add(key)
    count = len(distinct)
    require(count > 0, "No distinct completed mutations to plot")
    comparison = summary["comparison"]
    require(comparison["matched_random_prefix_available"] is True
            and comparison["matched_random_prefix_count"] == count,
            "Require the matching frozen random prefix")
    require([point["index"] for point in arms["random"][:count]] == list(range(1, count + 1)),
            "Random controls are not the preregistered prefix")
    paths = {"shinka": [default, *distinct], "random": [default, *arms["random"][:count]]}
    curves = {}
    for arm, points in paths.items():
        best, incumbent = default["score"], []
        for point in points:
            best = max(best, point["score"])
            incumbent.append(best)
        curves[arm] = {"x": list(range(count + 1)), "candidate_scores": [p["score"] for p in points],
                       "incumbent_scores": incumbent, "programs": points}
    require(curves["shinka"]["incumbent_scores"][-1] == comparison["shinka_best_including_default"]
            and curves["random"]["incumbent_scores"][-1]
            == comparison["random_best_matched_including_default"], "Incumbent summary mismatch")

    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
             "axes.linewidth": 0.65, "axes.edgecolor": "#87909A", "axes.labelcolor": "#25313B",
             "xtick.color": "#55616C", "ytick.color": "#55616C", "xtick.labelsize": 8,
             "ytick.labelsize": 8, "grid.color": "#E1E5E8", "grid.linewidth": 0.55,
             "svg.fonttype": "none", "svg.hashsalt": "shinka-crl-search-v1", "pdf.fonttype": 42,
             "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(9.0, 3.7))
        fig.subplots_adjust(left=0.085, right=0.98, bottom=0.23, top=0.80)
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
        ax.axhline(default["score"], color="#87909A", linewidth=0.9, linestyle=(0, (3, 3)))
        for arm, marker in (("shinka", "o"), ("random", "s")):
            curve = curves[arm]
            ax.step(curve["x"], curve["incumbent_scores"], where="post", color=COLORS[arm],
                    linewidth=2.0, zorder=3)
            ax.scatter(curve["x"][1:], curve["candidate_scores"][1:], color=COLORS[arm],
                       s=24, alpha=0.38, marker=marker, edgecolors="none", zorder=2)
            ax.scatter([count], [curve["incumbent_scores"][-1]], color=COLORS[arm],
                       s=27, marker=marker, edgecolors="white", linewidth=0.5, zorder=4)
        ax.scatter([0], [default["score"]], color="#55616C", s=20, zorder=5)
        ax.set(xlim=(-0.15, count + 0.25), ylim=(0, 1.0), ylabel="Development score, J",
               xlabel="Distinct configurations evaluated after the shared default")
        tick_step = max(1, math.ceil(count / 12))
        ax.set_xticks(sorted(set([0, count, *range(0, count + 1, tick_step)])))
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        handles = [Line2D([0], [0], color=COLORS[arm], linewidth=2.2, label=label)
                   for arm, label in (("shinka", "ShinkaEvolve"), ("random", "Random search"))]
        handles.append(Line2D([0], [0], color="#87909A", linewidth=1, linestyle="--",
                              label="Shared default"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.98),
                   frameon=False, ncol=3, handlelength=2.2, columnspacing=2.6)
        fig.text(0.085, 0.042, "Steps: best observed score · faint points: individual configurations · "
                 "three development seeds · one search per arm", fontsize=7.5, color="#65717D")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".search-figure-", dir=output.parent) as temporary:
            staging = Path(temporary)
            svg, pdf = staging / output.name, staging / output.with_suffix(".pdf").name
            fig.savefig(svg, metadata={"Date": None, "Title": "Development configuration search"})
            fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None,
                                      "Title": "Development configuration search"})
            provenance = {"schema_version": 1, "kind": "development_search_trajectories",
                          "plot_script": "scripts/plot_search.py",
                          "plot_script_sha256": digest(Path(__file__)),
                          "input_report": report_dir.name, "input_sha256": inputs,
                          "matplotlib_version": matplotlib.__version__, "curves": curves,
                          "matched_distinct_mutants": count,
                          "duplicate_evaluations_charged_separately":
                              comparison["shinka_duplicate_completed_evaluations"],
                          "random_candidates_not_in_matched_prefix": len(arms["random"]) - count,
                          "interpretation": "Observed development selection paths, no confidence "
                                            "bands or smoothing. Axis counts distinct configurations; "
                                            "repeated and failed training work remains in the report.",
                          "outputs_sha256": {svg.name: digest(svg), pdf.name: digest(pdf)}}
            sidecar = staging / output.with_suffix(".json").name
            sidecar.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
            for source, destination in zip((svg, pdf, sidecar), outputs):
                require(not destination.exists(), "Refusing to overwrite figure artifacts")
                source.rename(destination)
        plt.close(fig)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plot_search(args.report_dir, args.output)
    print(json.dumps({"outputs": list(result["outputs_sha256"]),
                      "matched_distinct_mutants": result["matched_distinct_mutants"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
