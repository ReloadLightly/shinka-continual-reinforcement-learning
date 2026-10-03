"""Plot the verified four-phase GA/ES/PPO development comparison after PPO completes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from shinka_crl import analysis
from shinka_crl.baseline_contract import validate_baseline_config
from shinka_crl.experiment import REPO_ROOT, UPSTREAM_COMMIT, load_profile, score_curve

NE_REPORT = REPO_ROOT / "reports/reference-development-prefixes-20261003"
METHODS = ("ga", "es", "ppo")
COLORS = {"ga": "#2263A5", "es": "#188577", "ppo": "#C65F31"}
TOTAL_STEPS = 614_400_000
PHASE_STEPS = 153_600_000
MEASURES = (("learning_accuracy", "Learning accuracy", "Return"),
            ("forgetting", "Signed forgetting", "Return"),
            ("learning_minus_forgetting", "LA − F", "Return"),
            ("zero_shot_transfer", "Transfer", "Return"),
            ("normalized_curve_average", "Cumulative performance", "Cumulative / (steps × 500)"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def published_hashes(root):
    """Verify the entire compact PPO export before selecting any measurements."""
    root = Path(root).resolve()
    receipt = root / "checksums.json"
    require(receipt.is_file() and not receipt.is_symlink(), "Missing published PPO checksums")
    hashes = read(receipt)["published_sha256"]
    paths = list(root.rglob("*"))
    require(not any(path.is_symlink() for path in paths), "Published symlinks are forbidden")
    require({str(path.relative_to(root)) for path in paths if path.is_file() and path != receipt}
            == set(hashes), "Published file set differs from checksums")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and digest(path) == expected,
                f"Changed published evidence: {name}")
    return {**hashes, "checksums.json": digest(receipt)}


def validate_prefix_profile(profile):
    expected = load_profile("paper-cartpole")
    expected.update(num_phases=4, seeds=[1001])
    expected["ne"]["num_generations"] = 800
    expected["ppo"]["num_updates"] = 6000
    require({key: value for key, value in profile.items() if key != "purpose"}
            == {key: value for key, value in expected.items() if key != "purpose"},
            "Figure requires the exact four-phase development prefix protocol")


def curve(records, measured):
    method = measured["method"]
    count, step_size = (6000, 102400) if method == "ppo" else (800, 768000)
    require(measured["num_phases"] == 4 and measured["phase_sequence"] == [0, 1, 0, 1]
            and measured["steps_per_update"] == step_size
            and measured["nominal_training_steps"] == TOTAL_STEPS and len(records) == count,
            "Curve horizon or native step clock differs")
    require((measured["seed"], measured["trial"], measured["eval_seed"]) == (1001, 1002, 901001),
            "Development trial identities differ")
    require([record["generation"] for record in records] == list(range(count)),
            "Curve records must be contiguous native updates")
    interval = count // 4
    require([record["task"] for record in records] == [(i // interval) % 2 for i in range(count)],
            "Curve task schedule differs")
    x = np.arange(1, count + 1, dtype=np.int64) * step_size
    y = np.asarray([record[f"centroid_task{record['task']}"] for record in records], dtype=float)
    require(np.isfinite(y).all() and ((0 <= y) & (y <= 500)).all(), "Invalid active-task returns")
    return {"analysis": measured, "x": x, "y": y}


def load_ne_data():
    # The checked-in derivation already verifies both complete source archives,
    # historical code identities, exact baselines, and fresh episode arithmetic.
    namespace = runpy.run_path(str(NE_REPORT / "derivation.py"))
    derived = namespace["derive"]()
    require(derived == read(NE_REPORT / "summary.json"), "GA/ES derived summary differs")
    grouped, vectors = {}, None
    inputs = {"ne/" + name: digest(NE_REPORT / name)
              for name in ("summary.json", "protocol.json", "derivation.py")}
    for row in derived["rows"]:
        method = row["method"]
        require(method in {"ga", "es"} and method not in grouped and row["new_trial"] is False,
                "Expected one reused development trajectory per NE method")
        records = read(NE_REPORT / row["evidence"]["training_records"])[:800]
        grouped[method] = curve(records, row["analysis"])
        require(vectors is None or vectors == row["noise_vectors"], "NE task draws differ")
        vectors = row["noise_vectors"]
        report = NE_REPORT / row["evidence"]["report"]
        for name, value in read(report / "checksums.json")["published_sha256"].items():
            inputs[f"ne-source/{method}/{name}"] = value
        inputs[f"ne-source/{method}/checksums.json"] = digest(report / "checksums.json")
    require(set(grouped) == {"ga", "es"}, "Both GA and ES prefixes are required")
    return grouped, vectors, inputs


def load_data(ppo_report):
    root = Path(ppo_report).resolve()
    hashes = published_hashes(root)

    def published(name):
        require(name in hashes, f"Unpublished PPO input: {name}")
        return read(root / name)

    summary = published("summary.json")
    require(summary.get("status") == "complete" and summary.get("mode") == "development"
            and summary.get("experiment_kind") == "resource_limited_development_prefix",
            "Figure requires a completed PPO development prefix; diagnostic and reporting data are excluded")
    require(summary.get("completed_trials") == summary.get("planned_trials") == 1
            and summary.get("completed_reporting_trials") == 0 and len(summary["rows"]) == 1,
            "Expected exactly one development PPO prefix")
    row = summary["rows"][0]
    require((row["method"], row["seed"], row["trial"], row["eval_seed"])
            == ("ppo", 1001, 1002, 901001), "PPO development trial identity differs")
    require(row["training_path"] == "training" and row["analysis_path"] == "analysis",
            "Unexpected native prefix artifact paths")
    manifest = published("raw/training/manifest.json")
    profile = manifest["profile"]
    validate_prefix_profile(profile)
    require(summary["profiles"] == {"paper-cartpole": profile}
            and manifest["upstream_commit"] == UPSTREAM_COMMIT
            and manifest.get("ga_settings") is None
            and manifest["derivation"]["additional_training_updates"] == 0,
            "Prefix provenance differs from its original trained trajectory")
    finalizer = published("raw/training/process-measurement.json")
    before = published("raw/training/checkpoint-probe.json")
    after = published("validation/artifacts-probe.json")
    require(finalizer["status"] == "complete" and finalizer["returncode"] == 0
            and finalizer["start_update"] == 6000 and finalizer["phase_events"] == []
            and before["step"] == before["record_count"] == after["record_count"] == 6000
            and before["records_sha256"] == after["records_sha256"]
            and before["phase_agents"] == after["phase_agents"]
            and before["phase_tasks"] == [0, 1, 0, 1],
            "Native finalization changed the saved records, phase policies, or endpoint")
    results, records = (published("raw/training/" + name)
                        for name in ("results.json", "training_metrics.json"))
    validate_baseline_config(results["config"], "ppo")
    training_score = score_curve(records, profile=profile, method="ppo")
    require(published("raw/training/summary.json") == training_score, "Native curve summary differs")
    posthoc = published("raw/analysis/manifest.json")
    require(posthoc["analysis_source_sha256"] == digest(analysis.__file__)
            and posthoc["upstream_commit"] == UPSTREAM_COMMIT, "PPO analysis implementation differs")
    measured = analysis.summarize_trial(
        manifest=manifest, results=results, records=records,
        evaluation=published("raw/analysis/evaluation.json"),
        checkpoint_metadata=published("raw/analysis/checkpoint-metadata.json"),
        episodes=10, eval_seed=901001)
    require(measured == published("raw/analysis/summary.json")
            and all(row[key] == measured[key]
                    for key in ("metrics", "phase_returns", "phase_training_returns"))
            and row["training_env_steps_nominal"] == measured["nominal_training_steps"],
            "PPO plotted measurements differ from raw episodes or training curve")
    grouped, vectors, inputs = load_ne_data()
    require(results["noise_vectors"] == vectors, "PPO task draws differ from GA and ES")
    grouped["ppo"] = curve(records, measured)
    inputs.update({"ppo/" + name: value for name, value in hashes.items()})
    return grouped, inputs


def format_axis(ax):
    ax.grid(axis="y", linewidth=.5, color="#E1E5E8")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_prefixes(ppo_report, output):
    output = Path(output).resolve()
    targets = [output, output.with_suffix(".pdf"), output.with_name(output.stem + "-metrics.svg"),
               output.with_name(output.stem + "-metrics.pdf"), output.with_suffix(".json")]
    require(output.suffix == ".svg" and not any(path.exists() for path in targets),
            "Choose a fresh .svg output path")
    grouped, inputs = load_data(ppo_report)
    label = "Four-phase development comparison · n = 1 per method"
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
             "axes.titlesize": 10, "axes.linewidth": .65, "axes.edgecolor": "#87909A",
             "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.fonttype": "none",
             "svg.hashsalt": "shinka-crl-development-prefix-v1", "pdf.fonttype": 42,
             "figure.facecolor": "white", "axes.facecolor": "white", "path.simplify": False}
    with plt.rc_context(style):
        curves, axes = plt.subplots(2, 3, figsize=(11.7, 6.5))
        curves.subplots_adjust(left=.07, right=.985, bottom=.16, top=.83, hspace=.45, wspace=.25)
        curves.suptitle("CartPole learning and retention\n" + label, fontsize=11)
        for column, method in enumerate(METHODS):
            data, color = grouped[method], COLORS[method]
            for ax in axes[:, column]:
                format_axis(ax)
                ax.set_ylim(0, 520)
            ax = axes[0, column]
            for phase in range(4):
                left, right = phase * PHASE_STEPS / 1e6, (phase + 1) * PHASE_STEPS / 1e6
                if phase % 2:
                    ax.axvspan(left, right, color="#EBEEF0", alpha=.7, linewidth=0)
                if phase:
                    ax.axvline(left, color="#A7AFB7", linestyle=(0, (2, 3)), linewidth=.6)
                ax.text((left + right) / 2, 1.015, "AB"[phase % 2],
                        transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
            ax.plot(data["x"] / 1e6, data["y"], color=color, linewidth=.8, rasterized=True)
            ax.set(xlim=(0, TOTAL_STEPS / 1e6), xticks=np.arange(5) * PHASE_STEPS / 1e6,
                   xlabel="Nominal training steps (millions)", ylabel="Active-task return")
            ax.set_title(method.upper(), pad=24)
            ax = axes[1, column]
            phases = data["analysis"]["phase_returns"]
            for field, linestyle, marker, start in (("own_mean", "-", "o", 0),
                                                    ("previous_mean", "--", "s", 1)):
                ax.plot(np.arange(start + 1, 5), [phase[field] for phase in phases[start:]],
                        color=color, linewidth=1.4, linestyle=linestyle, marker=marker,
                        markersize=4, markerfacecolor="white" if start else color)
            ax.set(xlim=(.75, 4.25), xticks=[1, 2, 3, 4], xlabel="Completed phase",
                   ylabel="Fresh checkpoint return")
        curves.legend(handles=[Line2D([], [], color="#465562", marker="o", label="Own task"),
                               Line2D([], [], color="#465562", marker="s", markerfacecolor="white",
                                      linestyle="--", label="Previous task")],
                      loc="lower center", bbox_to_anchor=(.5, .047), ncol=2, frameon=False)
        curves.text(.07, .022, "Native update clocks; no smoothing. A/B: alternating observation tasks.\n"
                    "Checkpoint means use ten fresh episodes per target. One development trajectory per method; no uncertainty intervals.", fontsize=7)

        metrics, panels = plt.subplots(1, 5, figsize=(12.5, 3.6))
        metrics.subplots_adjust(left=.06, right=.985, bottom=.23, top=.72, wspace=.6)
        metrics.suptitle("CartPole outcomes over four phases\n" + label, fontsize=11)
        for ax, (key, title, units) in zip(panels, MEASURES):
            format_axis(ax)
            values = [grouped[method]["analysis"]["metrics"][key] for method in METHODS]
            for index, (method, value) in enumerate(zip(METHODS, values)):
                ax.scatter(index, value, color=COLORS[method], s=34, zorder=3)
                ax.annotate(f"{value:.3f}" if key == "normalized_curve_average" else f"{value:.1f}",
                            (index, value), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=7)
            if key in {"forgetting", "learning_minus_forgetting"}:
                ax.axhline(0, color="#697680", linewidth=.6, linestyle=":")
                low, high = min(0., min(values)), max(0., max(values))
                span = max(50., high - low)
                ax.set_ylim(low - .08 * span, high + .23 * span)
            else:
                ax.set_ylim(0, 1.08 if key == "normalized_curve_average" else 540)
            ax.set(title=title, ylabel=units, xticks=range(3), xticklabels=[m.upper() for m in METHODS],
                   xlim=(-.5, 2.5))
        metrics.text(.06, .08, "Points are single development trials; no between-trial uncertainty is estimable.\n"
                     "Signed forgetting averages three switches; positive values indicate loss. Transfer includes recurring tasks.\n"
                     "Cumulative performance uses the pinned integration grid and division by nominal steps × 500.", fontsize=7)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".reference-prefix-", dir=output.parent) as temporary:
            staged = [Path(temporary) / path.name for path in targets]
            for figure, paths in ((curves, staged[:2]), (metrics, staged[2:4])):
                figure.savefig(paths[0], dpi=300, metadata={"Date": None, "Title": label})
                figure.savefig(paths[1], dpi=300,
                               metadata={"CreationDate": None, "ModDate": None, "Title": label})
            provenance = {
                "schema_version": 1, "mode": "development", "label": label,
                "n_per_method": 1, "completed_reporting_trials": 0,
                "ne_report": NE_REPORT.name, "ppo_report": Path(ppo_report).name,
                "input_sha256": inputs, "plot_script_sha256": digest(__file__),
                "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
                "smoothing": None, "dispersion": None,
                "curve_rendering": "Unsmoothed native samples rasterized at 300 dpi; axes remain vector",
                "curve_clock": "(zero-based update + 1) × nominal steps per update; no cross-method interpolation",
                "native_clocks": {method: {"samples": len(grouped[method]["x"]),
                                           "first_step": int(grouped[method]["x"][0]),
                                           "final_step": int(grouped[method]["x"][-1])} for method in METHODS},
                "measurements": {method: grouped[method]["analysis"] for method in METHODS},
                "outputs_sha256": {path.name: digest(path) for path in staged[:4]},
            }
            staged[-1].write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
            require(not any(path.exists() for path in targets), "Refusing to overwrite figures")
            for source, target in zip(staged, targets):
                source.rename(target)
        plt.close(curves)
        plt.close(metrics)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plot_prefixes(args.ppo_report, args.output)
    print(json.dumps({"label": result["label"], "outputs": list(result["outputs_sha256"])}, indent=2))
