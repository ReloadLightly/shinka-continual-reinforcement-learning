"""Plot verified full-budget CartPole reference curves, retention, and trial distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from shinka_crl import analysis
from shinka_crl.baseline_contract import validate_baseline_config
from shinka_crl.experiment import load_profile, score_curve

METHODS = ("ga", "es", "ppo")
COLORS = {"ga": "#2263A5", "es": "#188577", "ppo": "#C65F31"}
MEASURES = (("learning_accuracy", "Learning accuracy (return)"),
            ("forgetting", "Signed forgetting (return)"),
            ("learning_minus_forgetting", "LA − F (return)"),
            ("zero_shot_transfer", "Zero-shot transfer (return)"),
            ("normalized_curve_average", "Cumulative / (steps × 500)"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    return {"n": len(values), "mean": statistics.mean(values), "values": values,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def identity(row):
    return tuple(row[k] for k in ("profile", "method", "seed", "trial", "eval_seed"))


def load_data(report_dir, *, allow_development=False, allow_partial=False):
    root = Path(report_dir).resolve()
    checksum_path = root / "checksums.json"
    require(not checksum_path.is_symlink(), "Checksum must be a regular file")
    hashes = json.loads(checksum_path.read_text())["published_sha256"]
    paths = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "Published symlinks are forbidden")
    require({str(p.relative_to(root)) for p in paths if p.is_file() and p != checksum_path} == set(hashes),
            "Published file set differs from checksums")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and digest(path) == expected, f"Changed published evidence: {name}")

    def read(name):
        require(name in hashes, f"Unpublished input: {name}")
        return json.loads((root / name).read_text())

    summary, plan = read("summary.json"), read("raw/plan.json")
    mode, status = summary["mode"], summary["status"]
    require(mode == plan["mode"] and mode in {"development", "reporting"}, "Diagnostic data cannot be a scientific plot")
    require(mode != "development" or allow_development, "Development requires --allow-development")
    require(status in {"complete", "partial", "failed", "interrupted"}, "Cannot plot an active suite")
    require(status == "complete" or allow_partial, "Partial comparison requires --allow-partial")
    profile = plan["profiles"]["paper-cartpole"]
    expected = load_profile("paper-cartpole")
    if mode == "development":
        expected["seeds"] = [1001]
    require({k: v for k, v in profile.items() if k != "purpose"}
            == {k: v for k, v in expected.items() if k != "purpose"}, "Full-budget profile changed")
    require(plan["eval_episodes"] == 10 and summary["profiles"] == plan["profiles"], "Evaluation protocol changed")
    methods = plan["methods"]
    require(methods and len(methods) == len(set(methods)) and set(methods) <= set(METHODS), "Unknown reference methods")
    require(mode != "reporting" or methods in (list(METHODS), ["ga", "es"]),
            "Reporting requires GA, ES, PPO or the amended GA, ES scope")
    jobs = [{"profile": "paper-cartpole", "method": method, "seed": seed,
             "trial": i if mode == "reporting" else seed + 1, "eval_seed": seed + 900000}
            for i, seed in enumerate(profile["seeds"], 1) for method in methods]
    require(plan["jobs"] == jobs and summary["planned_trials"] == len(jobs), "Declared seed/task mapping changed")
    rows, keys = summary["rows"], [identity(row) for row in summary["rows"]]
    require(rows and len(keys) == len(set(keys)) and set(keys) <= {identity(j) for j in jobs}
            and summary["completed_trials"] == len(rows), "Invalid completed trial set")
    complete = len(rows) == len(jobs)
    require(summary["validation"]["all_planned_trials"] == complete, "Completeness claim differs from trials")
    require(status != "complete" or complete, "Incomplete suite marked complete")
    require(mode != "reporting" or allow_partial or complete, "Final figure requires all declared trials")
    original_complete = mode == "reporting" and methods == list(METHODS) and complete and status == "complete"
    if mode == "reporting" and methods == ["ga", "es"]:
        require(summary.get("declared_methods") == methods
                and summary.get("original_three_method_target_complete") is False,
                "Amended reporting must explicitly retain the incomplete original target")
    if "declared_methods" in summary:
        require(summary["declared_methods"] == methods, "Reported method scope differs from plan")
    if "original_three_method_target_complete" in summary:
        require(summary["original_three_method_target_complete"] is original_complete,
                "Original target completeness differs from trials")
    require(plan["source_sha256"]["src/shinka_crl/analysis.py"] == digest(analysis.__file__), "Analysis source differs from frozen protocol")
    grouped, noise = {}, {}
    for row in rows:
        method = row["method"]
        train, posthoc = "raw/" + row["training_path"], "raw/" + row["analysis_path"]
        manifest, results, records = (read(train + "/" + name) for name in
                                      ("manifest.json", "results.json", "training_metrics.json"))
        require(manifest["profile"] == profile, "Training profile differs from declared comparison")
        validate_baseline_config(results["config"], method)
        training_score = score_curve(records, profile=profile, method=method)
        saved_score = read(train + "/summary.json")
        require(all(saved_score.get(k) == v for k, v in training_score.items()), "Training score differs from raw curve")
        measured = analysis.summarize_trial(manifest=manifest, results=results, records=records,
            evaluation=read(posthoc + "/evaluation.json"), checkpoint_metadata=read(posthoc + "/checkpoint-metadata.json"),
            episodes=10, eval_seed=row["eval_seed"])
        require(measured == read(posthoc + "/summary.json") and identity(measured) == identity(row), "Raw checkpoint analysis differs")
        require(all(row[k] == measured[k] for k in ("metrics", "phase_returns", "phase_training_returns")), "Plotted measurements differ from raw evidence")
        require(row["training_env_steps_nominal"] == measured["nominal_training_steps"] == 3072000000, "Training budget differs")
        require(row["seed"] not in noise or noise[row["seed"]] == results["noise_vectors"], "Task draws differ by method")
        noise[row["seed"]] = results["noise_vectors"]
        y = np.asarray([record[f"centroid_task{record['task']}"] for record in records], dtype=float)
        x = np.arange(1, len(y) + 1, dtype=np.int64) * measured["steps_per_update"]
        grouped.setdefault(method, []).append({"row": row, "x": x, "y": y})
    require({g["method"] for g in summary["groups"]} == set(grouped), "Reported aggregate methods differ")
    for method, trials in grouped.items():
        trials.sort(key=lambda t: t["row"]["seed"])
        group = next(g for g in summary["groups"] if g["method"] == method)
        require(group["n"] == len(trials) and group["seeds"] == [t["row"]["seed"] for t in trials], "Aggregate trial membership differs")
        require(all(np.array_equal(t["x"], trials[0]["x"]) for t in trials), "Clocks within a method differ")
        for metric in trials[0]["row"]["metrics"]:
            require(group["metrics"][metric] == describe([t["row"]["metrics"][metric] for t in trials]), "Aggregate metric differs from trials")
    return summary, grouped, {**hashes, "checksums.json": digest(checksum_path)}


def format_axis(ax):
    ax.grid(axis="y", linewidth=.5, color="#E1E5E8")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_reference_comparison(report_dir, output, *, allow_development=False, allow_partial=False):
    output = Path(output).resolve()
    targets = [output, output.with_suffix(".pdf"), output.with_name(output.stem + "-metrics.svg"),
               output.with_name(output.stem + "-metrics.pdf"), output.with_suffix(".json")]
    require(output.suffix == ".svg" and not any(p.exists() for p in targets), "Choose a fresh .svg output path")
    summary, grouped, inputs = load_data(report_dir, allow_development=allow_development, allow_partial=allow_partial)
    methods = [m for m in METHODS if m in grouped]
    amended = summary["mode"] == "reporting" and summary.get("declared_methods") == ["ga", "es"]
    label = ("Development (not reporting)" if summary["mode"] == "development" else
             ("GA/ES reporting; PPO deferred" if summary["status"] == "complete"
              else "Partial GA/ES reporting; PPO deferred") if amended else
             "Final reporting" if summary["status"] == "complete" else "Partial reporting")
    label += f" · {summary['completed_trials']}/{summary['planned_trials']} trials"
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
             "axes.titlesize": 10, "axes.linewidth": .65, "axes.edgecolor": "#87909A",
             "xtick.labelsize": 8, "ytick.labelsize": 8, "svg.fonttype": "none",
             "svg.hashsalt": "shinka-crl-reference-comparison-v1", "pdf.fonttype": 42,
             "figure.facecolor": "white", "axes.facecolor": "white", "path.simplify": False}
    with plt.rc_context(style):
        curves, axes = plt.subplots(2, len(methods), figsize=(max(6, 4 * len(methods)), 6.6), squeeze=False)
        curves.subplots_adjust(left=.13 if len(methods) == 1 else .075,
                               right=.985, bottom=.16, top=.85, hspace=.45, wspace=.25)
        curves.suptitle("CartPole learning and retention\n" + label, fontsize=11)
        for column, method in enumerate(methods):
            trials, color = grouped[method], COLORS[method]
            for ax in axes[:, column]:
                format_axis(ax)
                ax.set_ylim(0, 515)
            ax = axes[0, column]
            for phase in range(20):
                left, right = phase * .1536, (phase + 1) * .1536
                if phase % 2:
                    ax.axvspan(left, right, color="#EBEEF0", alpha=.7, linewidth=0)
                if phase:
                    ax.axvline(left, color="#A7AFB7", linestyle=(0, (2, 3)), linewidth=.5)
                ax.text((left + right) / 2, 1.015, "AB"[phase % 2], transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5)
            for trial in trials:
                ax.plot(trial["x"] / 1e9, trial["y"], color=color, alpha=.22, linewidth=.55, rasterized=True)
            ax.plot(trials[0]["x"] / 1e9, np.mean([t["y"] for t in trials], axis=0), color=color, linewidth=1.45, rasterized=True)
            ax.set(xlim=(0, 3.072), xlabel="Nominal training steps (billions)", ylabel="Active-task return")
            ax.set_title(f"{method.upper()} · n = {len(trials)}", pad=24)
            ax = axes[1, column]
            for field, linestyle, start in (("own_mean", "-", 0), ("previous_mean", "--", 1)):
                values = np.asarray([[p[field] for p in t["row"]["phase_returns"][start:]] for t in trials])
                phases = np.arange(start + 1, 21)
                for values_trial in values:
                    ax.plot(phases, values_trial, color=color, alpha=.2, linewidth=.65, linestyle=linestyle)
                ax.plot(phases, values.mean(axis=0), color=color, linewidth=1.5, linestyle=linestyle, marker="o", markersize=2.5)
            ax.set(xlim=(.5, 20.5), xticks=[1, 5, 10, 15, 20], xlabel="Completed phase", ylabel="Fresh checkpoint return")
        curves.legend(handles=[Line2D([], [], color="#465562", label="Own task", linewidth=1.5),
                               Line2D([], [], color="#465562", label="Previous task", linestyle="--", linewidth=1.5)],
                      loc="lower center", bbox_to_anchor=(.5, .04), ncol=2, frameon=False)
        curves.text(.075, .017, "Thin: individual trials; bold: arithmetic mean. Native update clocks; no smoothing.\n"
                    "Checkpoint agents use ten fresh episodes per target; previous-task return starts at phase 2.", fontsize=7)
        metrics, panels = plt.subplots(2, 3, figsize=(11, 6.4))
        metrics.subplots_adjust(left=.08, right=.98, bottom=.13, top=.84, wspace=.35, hspace=.42)
        metrics.suptitle("CartPole outcomes across trials\n" + label, fontsize=11)
        for ax, (key, title) in zip(panels.flat, MEASURES):
            format_axis(ax)
            ax.set_title(title, loc="left")
            for i, method in enumerate(methods):
                values = [t["row"]["metrics"][key] for t in grouped[method]]
                jitter = np.linspace(-.14, .14, len(values)) if len(values) > 1 else np.array([0.])
                ax.scatter(i + jitter, values, s=24, color=COLORS[method], alpha=.6, edgecolors="none", zorder=3)
                ax.errorbar(i, statistics.mean(values), yerr=statistics.stdev(values) if len(values) > 1 else 0,
                            color="#182630", marker="_", markersize=13, capsize=4, linewidth=1.3, zorder=4)
            if key in {"forgetting", "learning_minus_forgetting"}:
                ax.axhline(0, color="#697680", linewidth=.6, linestyle=":")
            ax.set(xticks=range(len(methods)), xticklabels=[m.upper() for m in methods], xlim=(-.5, len(methods) - .5))
        ax = panels.flat[-1]
        format_axis(ax)
        for method in methods:
            la = [t["row"]["metrics"]["learning_accuracy"] for t in grouped[method]]
            forgetting = [t["row"]["metrics"]["forgetting"] for t in grouped[method]]
            ax.scatter(la, forgetting, s=22, color=COLORS[method], alpha=.45)
            ax.scatter(statistics.mean(la), statistics.mean(forgetting), marker="D", color=COLORS[method], s=44, edgecolors="white", label=method.upper())
        ax.set(title="Learning–retention trade-off", xlabel="Learning accuracy (return)", ylabel="Signed forgetting (return)")
        ax.legend(frameon=False, fontsize=7)
        metrics.text(.08, .035, "Points: trials. Black bars: mean ± sample SD where n > 1; no confidence intervals.\n"
                     "Positive forgetting denotes lost return; negative values denote improvement. Diamonds: method means.", fontsize=7.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".reference-comparison-", dir=output.parent) as temp:
            staged = [Path(temp) / p.name for p in targets]
            for fig, paths in ((curves, staged[:2]), (metrics, staged[2:4])):
                fig.savefig(paths[0], dpi=300, metadata={"Date": None, "Title": label})
                fig.savefig(paths[1], dpi=300, metadata={"CreationDate": None, "ModDate": None, "Title": label})
            provenance = {"schema_version": 1, "mode": summary["mode"], "status": summary["status"],
                "label": label, "completed_trials": summary["completed_trials"], "input_report": Path(report_dir).name,
                "declared_methods": summary.get("declared_methods", methods),
                "original_three_method_target_complete": summary["mode"] == "reporting"
                    and methods == list(METHODS) and summary["status"] == "complete",
                "input_sha256": inputs, "plot_script_sha256": digest(__file__),
                "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
                "curve_clock": "(zero-based update + 1) * nominal steps per update; no cross-method interpolation",
                "smoothing": None, "curve_rendering": "Unsmoothed native samples rasterized at 300 dpi; axes remain vector",
                "dispersion": "sample standard deviation across trials; no confidence intervals",
                "groups": summary["groups"], "phase_returns": [{k: r[k] for k in ("method", "seed", "trial", "phase_returns")} for r in summary["rows"]],
                "native_clocks": {m: {"samples": len(grouped[m][0]["x"]), "first_step": int(grouped[m][0]["x"][0]),
                                     "final_step": int(grouped[m][0]["x"][-1])} for m in methods},
                "outputs_sha256": {p.name: digest(p) for p in staged[:4]}}
            staged[-1].write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
            require(not any(p.exists() for p in targets), "Refusing to overwrite figures")
            for source, target in zip(staged, targets):
                source.rename(target)
        plt.close(curves)
        plt.close(metrics)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-development", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = plot_reference_comparison(args.report_dir, args.output,
        allow_development=args.allow_development, allow_partial=args.allow_partial)
    print(json.dumps({"outputs": list(result["outputs_sha256"]), "label": result["label"]}, indent=2))


if __name__ == "__main__":
    main()
