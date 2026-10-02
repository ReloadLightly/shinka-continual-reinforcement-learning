"""Plot verified real adaptive-gate widths and persistent diagnostic memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plot_adaptive_gate(report_dir, output):
    report_dir, output = Path(report_dir).resolve(), Path(output).resolve()
    require(output.suffix == ".svg", "Output must have an .svg extension")
    targets = [output, output.with_suffix(".pdf"), output.with_suffix(".json")]
    require(not any(path.exists() for path in targets), "Refusing to replace figure artifacts")
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
    verification = summary.get("verification", {})
    require(summary.get("status") == "complete" and verification.get("status") == "passed"
            and all(check.get("passed") is True for check in verification.get("checks", [])),
            "Require a completed gate with independently verified numerical traces")
    require(len(plan["trials"]) == 7 and summary["seed"] == 3001, "Unexpected gate protocol")
    profiles = {row["id"]: row["profile"] for row in plan["trials"]}
    series, checks = {}, verification["checks"]
    for name in ("identity", "halving", "arithmetic"):
        profile = profiles[f"{name}_switching"]
        require(profile["ne"]["num_generations"] == 6 and profile["ne"]["task_interval"] == 3
                and profile["num_phases"] == 2 and profile["num_tasks"] == 2,
                "Unexpected diagnostic generation or task schedule")
        trace = read(f"raw/{name}_switching/training/adapter-trace/manifest.json")
        states = trace["states"]
        require(trace["status"] == "complete" and trace["algorithm_variant"] == "ga_adaptive"
                and [row["completed_generations"] for row in states] == list(range(7)),
                "Incomplete adaptive state trace")
        sigma = [row["sigma_next"] for row in states]
        memory = [row["memory"] for row in states]
        require(all(type(x) in (int, float) and math.isfinite(x) and .001 <= x <= 2 for x in sigma)
                and all(len(row) == 4 and all(math.isfinite(x) for x in row) for row in memory),
                "Invalid observed width or memory")
        series[name] = {"generations": list(range(6)), "sigma_used": sigma[:-1],
                        "sigma_next": sigma[1:], "completed_generations": list(range(7)),
                        "memory": memory}
        if name != "identity":
            matched = [check for check in checks if check["name"].startswith(name + ":")]
            require(len(matched) == 1 and matched[0]["sigma_used"] == sigma[:-1]
                    and matched[0]["sigma_next"] == sigma[1:], "Width verification differs from trace")
        else:
            require(sigma == [.5] * 7 and any(check["name"].startswith("identity_switching: exact")
                                            for check in checks), "Identity parity was not verified")
    counter = series["halving"]["memory"]
    require(all(row == [generation] * 4 for generation, row in enumerate(counter)),
            "The diagnostic counter did not persist across the task switch")
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
             "axes.linewidth": .65, "axes.edgecolor": "#87909A", "axes.labelcolor": "#25313B",
             "xtick.color": "#55616C", "ytick.color": "#55616C", "xtick.labelsize": 8,
             "ytick.labelsize": 8, "grid.color": "#E1E5E8", "grid.linewidth": .55,
             "svg.fonttype": "none", "svg.hashsalt": "shinka-crl-adaptive-gate-v1",
             "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white"}
    with plt.rc_context(style):
        fig, (width_ax, memory_ax) = plt.subplots(1, 2, figsize=(10.5, 3.9))
        fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.79, wspace=.27)
        for ax in (width_ax, memory_ax):
            ax.set_axisbelow(True)
            ax.grid(axis="y")
            for edge in ("top", "right"):
                ax.spines[edge].set_visible(False)
            ax.axvline(3, color="#9AA5AD", linewidth=.9, linestyle=(0, (3, 3)))
            ax.text(3.08, .95, "Task switch", transform=ax.get_xaxis_transform(),
                    va="top", fontsize=7.5, color="#65717D")
        colors = {"identity": "#65717D", "halving": "#2263A5", "arithmetic": "#C65F31"}
        for name, values in series.items():
            width_ax.plot(values["generations"], values["sigma_used"], marker="o", markersize=4,
                          linewidth=1.7, color=colors[name], label=name.capitalize())
        upper = 1.18 * max(max(values["sigma_used"]) for values in series.values())
        width_ax.set(xlabel="Generation g (zero-based)", ylabel="Mutation width used, σ",
                     xticks=range(6), xlim=(-.12, 5.3), ylim=(0, upper))
        width_ax.set_title("A  Width used to generate offspring", loc="left", pad=28,
                           fontsize=10, fontweight="semibold", color="#25313B")
        width_ax.legend(loc="lower left", bbox_to_anchor=(-.02, 1.01), ncol=3, frameon=False,
                        fontsize=7.5, handlelength=1.5, columnspacing=1.4)
        memory_ax.plot(range(7), [row[0] for row in counter], marker="o", markersize=4,
                       linewidth=1.7, color=colors["halving"])
        memory_ax.set(xlabel="Completed generations", ylabel="Diagnostic memory counter",
                      xticks=range(7), yticks=range(7), xlim=(-.12, 6.2), ylim=(-.15, 6.5))
        memory_ax.set_title("B  Memory persists across the switch", loc="left", pad=28,
                            fontsize=10, fontweight="semibold", color="#25313B")
        memory_ax.text(0, 1.03, "All four memory values coincide", transform=memory_ax.transAxes,
                       va="bottom", fontsize=7.5, color="#65717D")
        fig.text(.075, .095, "Observed diagnostic traces · one development seed · six generations · "
                 "identity matches native GA exactly", fontsize=7.5, color="#65717D")
        fig.text(.075, .035, "Raw σ is logged after each update; panel A shifts it to the generation "
                 "that uses it. These checks establish adapter behavior.", fontsize=7.5, color="#65717D")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".adaptive-gate-figure-", dir=output.parent) as temporary:
            stage = Path(temporary)
            svg, pdf = stage / output.name, stage / output.with_suffix(".pdf").name
            fig.savefig(svg, metadata={"Date": None, "Title": "Adaptive mutation adapter gate"})
            fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None,
                                       "Title": "Adaptive mutation adapter gate"})
            data = {"schema_version": 1, "kind": "observed_adaptive_adapter_gate",
                    "input_sha256": inputs, "plot_script_sha256": digest(__file__),
                    "seed": summary["seed"], "task_switch_generation": 3, "series": series,
                    "output_sha256": {path.name: digest(path) for path in (svg, pdf)}}
            sidecar = stage / output.with_suffix(".json").name
            sidecar.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
            for source, target in zip((svg, pdf, sidecar), targets, strict=True):
                source.rename(target)
        plt.close(fig)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/adaptive-gate-20261003"))
    parser.add_argument("--output", type=Path, default=Path("figures/adaptive-gate-20261003.svg"))
    arguments = parser.parse_args()
    plot_adaptive_gate(arguments.report_dir, arguments.output)
