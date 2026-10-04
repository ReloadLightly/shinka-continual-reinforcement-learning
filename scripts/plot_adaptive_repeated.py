"""Plot a complete repeated-search comparison, rederived from published evidence.

Partial experiments are refused. Evaluation seeds are paired task draws within
an outer-search repetition; the primary aggregate has two observations, not ten.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shinka_crl.adaptive_objective import score_adaptive_trial  # noqa: E402
from shinka_crl.analysis import summarize_trial  # noqa: E402

SEEDS = (7001, 7002, 7003, 7004, 7005)
OUTER_SEEDS = (202610041, 202610042)
SCORES = ("combined_score", "active_score", "previous_score")
CONTROLS = ("identity", "arithmetic", "focus")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def describe(values):
    require(len(values) >= 2, "At least two observations are required for sample SD")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "n": len(values)}


def primary_comparison(conditions, outer_seeds=OUTER_SEEDS, seeds=SEEDS):
    """Average matched task differences first; aggregate outer searches second."""
    require(tuple(outer_seeds) == OUTER_SEEDS, "Require both prespecified outer-search pairs")
    repetitions = []
    for outer_seed in outer_seeds:
        left, right = (conditions[f"{arm}_{outer_seed}"]["trials"]
                       for arm in ("evolutionary", "independent"))
        require([row["seed"] for row in left] == list(seeds)
                and [row["seed"] for row in right] == list(seeds),
                "Every finalist requires all five paired evaluation seeds")
        paired = [{"seed": a["seed"], "difference": a["score"]["combined_score"] - b["score"]["combined_score"]}
                  for a, b in zip(left, right, strict=True)]
        repetitions.append({"outer_seed": outer_seed, "paired_seed_differences": paired,
                            "aggregate": describe([row["difference"] for row in paired]),
                            "complete_five_seed_comparison": True})
    return repetitions, describe([row["aggregate"]["mean"] for row in repetitions])


def recompute_trial(read, prefix, plan, row):
    """Pure scientific reconstruction; never loads binary policies or runs agents."""
    manifest = read(prefix + "/training/manifest.json")
    records = read(prefix + "/training/training_metrics.json")
    results = read(prefix + "/training/results.json")
    evaluation = read(prefix + "/analysis/evaluation.json")
    require(manifest["profile"] == plan["profile"] and manifest["seed"] == row["seed"]
            and manifest["trial"] == row["trial"]
            and manifest["upstream_commit"] == plan["upstream_commit"], "Trial provenance differs")
    analysis = summarize_trial(manifest=manifest, results=results, records=records,
                               evaluation=evaluation,
                               checkpoint_metadata=read(prefix + "/analysis/checkpoint-metadata.json"),
                               episodes=plan["posthoc_episodes"], eval_seed=row["eval_seed"])
    require(analysis == read(prefix + "/analysis/summary.json"), "Saved reference metrics differ from raw evidence")
    score = score_adaptive_trial(profile=plan["profile"], records=records, analysis=analysis,
                                 evaluation=evaluation, method=manifest["method"], seed=row["seed"],
                                 trial=row["trial"], eval_seed=row["eval_seed"], episodes=plan["posthoc_episodes"])
    episodes = sum(len(values) for entry in evaluation["per_task"] for key, values in entry.items()
                   if key.endswith("returns"))
    require(episodes == 300, "Unexpected fresh evaluation allocation")
    return score, analysis, [record[f"centroid_task{record['task']}"] for record in records]


def load_data(report_dir):
    report = Path(report_dir).resolve()
    checksum = report / "checksums.json"
    require(checksum.is_file() and not checksum.is_symlink(), "Require an exported evidence checksum")
    published = json.loads(checksum.read_text())["published_sha256"]
    inputs = {"checksums.json": digest(checksum)}
    actual = set()
    for path in report.rglob("*"):
        require(not path.is_symlink(), "Published evidence must not contain symbolic links")
        if path.is_file() and path != checksum:
            actual.add(str(path.relative_to(report)))
    require(actual == set(published), "Published file set differs from its checksum receipt")
    for name, expected in published.items():
        path = (report / name).resolve()
        require(path.is_relative_to(report) and digest(path) == expected, f"Published artifact changed: {name}")

    def read(relative):
        require(relative in published, f"Missing published input: {relative}")
        inputs[relative] = published[relative]
        return json.loads((report / relative).read_text())

    summary, plan = read("summary.json"), read("frozen/plan.json")
    require(summary["kind"] == plan["kind"] == "repeated_reserved"
            and summary["protocol_version"] == plan["protocol_version"] == "adaptive-repeated-validation-v1",
            "Require the repeated adaptive-search protocol")
    require(summary["status"] == "complete", "Partial comparison: no final plot; preserve missing outcomes as missing")
    profile = plan["profile"]
    require(profile["seeds"] == list(SEEDS) and profile["ne"] == {
        "num_generations": 320, "task_interval": 80, "pop_size": 64, "num_evals": 3}
        and profile["num_phases"] == 4 and profile["num_tasks"] == 2
        and profile["episode_length"] == 500 and profile["env"] == "CartPole-v1_sigma0.5"
        and plan["objective_weights"] == {"active": .5, "previous": .5}
        and plan["objective_version"] == "adaptive-active-previous-v1" and plan["posthoc_episodes"] == 10,
        "Declared learner or objective protocol differs")
    for relative in ("src/shinka_crl/analysis.py", "src/shinka_crl/adaptive_objective.py"):
        require(digest(ROOT / relative) == plan["source_sha256"][relative], "Scientific metric implementation differs from the freeze")
    frozen = {name.removeprefix("frozen/"): value for name, value in published.items()
              if name.startswith("frozen/") and name != "frozen/receipt.json"}
    require(read("frozen/receipt.json") == frozen
            and published["frozen/protocol.md"] == plan["protocol_sha256"], "Frozen handoff changed")
    selections = plan["provenance"]["selected_searches"]
    expected = {(seed, arm) for seed in OUTER_SEEDS for arm in ("evolutionary", "independent")}
    require(len(selections) == 4 and {(row["outer_seed"], row["arm"]) for row in selections} == expected,
            "Require exactly the two prespecified paired outer searches")
    candidates = {candidate["id"]: candidate for candidate in plan["conditions"]}
    condition_ids = [f"{arm}_{seed}" for seed in OUTER_SEEDS for arm in ("evolutionary", "independent")] + list(CONTROLS)
    require(len(plan["conditions"]) == 7 and set(candidates) == set(condition_ids)
            and set(summary["conditions"]) == set(condition_ids), "Finalist or fixed control membership changed")
    for selection in selections:
        ranking = selection["ranked_candidates"]
        require(len(ranking) == 5 and {r["generation"] for r in ranking} == set(range(5))
                and ranking == sorted(ranking, key=lambda r: (-r["development_score"], r["generation"]))
                and ranking[0]["generation"] == selection["generation"]
                and ranking[0]["development_score"] == selection["development_score"]
                and candidates[selection["id"]]["source_sha256"] == selection["source_sha256"],
                "Finalist differs from the frozen exact development ranking")
    for candidate in candidates.values():
        if candidate["source"]:
            name = "frozen/" + candidate["source"]
            require(published[name] == candidate["source_sha256"], "Frozen candidate source differs")
            tree = ast.parse((report / name).read_text())
            require(hashlib.sha256(ast.dump(tree, annotate_fields=True, include_attributes=False).encode()).hexdigest()
                    == candidate["program"]["canonical_ast_sha256"], "Frozen candidate AST differs")
    recipes = plan["unique_recipes"]
    require(len({recipe["recipe_key"] for recipe in recipes}) == len(recipes)
            and all(json_digest(recipe["recipe"]) == recipe["recipe_key"] for recipe in recipes)
            and sorted(member for recipe in recipes for member in recipe["memberships"]) == sorted(condition_ids),
            "Deduplicated recipe identity or memberships differ")
    for recipe in recipes:
        require(recipe["candidate"] in plan["conditions"] and recipe["candidate"]["id"] in recipe["memberships"],
                "Representative recipe candidate differs")
        for identity in recipe["memberships"]:
            candidate = candidates[identity]
            require(candidate["variant"] == recipe["recipe"]["variant"]
                    and candidate.get("program", {}).get("canonical_ast_sha256")
                    == recipe["recipe"]["program_identity"], "Reused memberships have different program identities")
    order = [{"index": i * len(recipes) + j, "recipe_key": recipe["recipe_key"],
              "memberships": recipe["memberships"], "seed": seed, "trial": seed + 1, "eval_seed": seed + 900000}
             for i, seed in enumerate(SEEDS) for j, recipe in enumerate(recipes)]
    require(plan["trial_order"] == order and plan["planned_trials"] == summary["scored_trials"]
            == summary["planned_trials"] == len(order), "Missing or altered trial allocation")
    require(plan["trial_steps_nominal"] == 30720000
            and plan["planned_nominal_training_steps"] == len(order) * 30720000
            and plan["planned_fresh_evaluation_episodes"] == summary["scored_fresh_evaluation_episodes"]
            == len(order) * 300, "Nominal or fresh episode allocation differs")
    state = read("raw/state.json")
    require(read("raw/state-receipt.json") == {"sha256": published["raw/state.json"]}
            and state["context_sha256"] == json_digest(plan), "State binding differs")
    completed, curves, phase_returns, task_vectors = {}, {}, {}, {}
    for attempt in state["attempts"]:
        prefix = "raw/" + attempt["path"]
        require(published[prefix + "/receipt.json"] == attempt["receipt_sha256"], "Trial receipt changed")
        receipt = read(prefix + "/receipt.json")
        require(all(receipt.get(name.removeprefix(prefix + "/")) == value
                    for name, value in published.items()
                    if name.startswith(prefix + "/") and name != prefix + "/receipt.json"),
                "Published trial differs from its original receipt")
        saved, row = read(prefix + "/summary.json"), order[attempt["trial_index"]]
        require(saved["context_sha256"] == json_digest(plan) and saved["trial"] == row, "Trial context differs")
        if saved["status"] != "complete":
            require(saved["status"] in ("failed", "interrupted"), "Unresolved trial attempt")
            continue
        require(row["index"] not in completed, "Duplicate completed trial")
        recipe = next(recipe for recipe in recipes if recipe["recipe_key"] == row["recipe_key"])
        candidate = recipe["candidate"]
        training = read(prefix + "/training/manifest.json")
        require(training.get("program") == candidate.get("program")
                and training["method"] == recipe["recipe"]["native_method"], "Executed program or learner differs")
        if candidate["source_sha256"]:
            require(published[prefix + "/program.py"] == candidate["source_sha256"], "Executed source differs")
        vectors = read(prefix + "/training/results.json")["noise_vectors"]
        require(task_vectors.setdefault(row["seed"], vectors) == vectors, "Paired trials have different task vectors")
        score, analysis, curve = recompute_trial(read, prefix, plan, row)
        result = saved["result"]
        require({key: result[key] for key in row} == row and result["score"] == score
                and result["fresh_evaluation_episodes"] == 300
                and result["phase_returns"] == analysis["phase_returns"], "Saved result differs from recomputed evidence")
        completed[row["index"]], curves[row["index"]], phase_returns[row["index"]] = result, curve, analysis["phase_returns"]
    require(set(completed) == set(range(len(order))), "Incomplete trial evidence")
    conditions = {}
    for identity in condition_ids:
        rows = summary["conditions"][identity]["trials"]
        require([row["seed"] for row in rows] == list(SEEDS)
                and all(row == completed[row["index"]] and identity in row["memberships"] for row in rows),
                "Missing five-seed condition or incorrect reuse membership")
        aggregates = {key: describe([row["score"][key] for row in rows]) for key in SCORES}
        reference = {key: describe([row["score"]["reference_metrics"][key] for row in rows])
                     for key in rows[0]["score"]["reference_metrics"]}
        require(aggregates == summary["conditions"][identity]["aggregate"]["scores"]
                and all(value == reference[key] for key, value in summary["conditions"][identity]["aggregate"]["reference_metrics"].items()),
                "Condition means or sample deviations differ")
        conditions[identity] = {"trials": rows, "scores": aggregates, "reference_metrics": reference,
                                "active_curves": [{"seed": row["seed"], "returns": curves[row["index"]]} for row in rows],
                                "phase_returns": [{"seed": row["seed"], "phases": phase_returns[row["index"]]} for row in rows]}
    repetitions, across = primary_comparison(conditions)
    require(repetitions == summary["repetition_comparisons"] and across == summary["across_search_repetitions"],
            "Primary outer-search aggregate differs from paired raw results")
    secondary = {}
    for identity in condition_ids[:4]:
        for control in CONTROLS:
            paired = [{"seed": left["seed"], "difference": left["score"]["combined_score"] - right["score"]["combined_score"]}
                      for left, right in zip(conditions[identity]["trials"], conditions[control]["trials"], strict=True)]
            secondary[f"{identity}_minus_{control}"] = {
                "paired_differences": paired, "aggregate": describe([row["difference"] for row in paired])}
    require(secondary == summary["comparisons"], "Secondary paired differences changed")
    return {"profile": profile, "conditions": conditions, "selected_searches": selections,
            "primary_repetitions": repetitions, "primary_across_searches": across, "secondary_comparisons": secondary,
            "verified_unique_trials": len(completed), "experimental_condition_seed_memberships": 35,
            "verified_published_artifacts": len(published),
            "interpretation": "Two paired outer-search repetitions on shared development and evaluation tasks; descriptive conditional evidence, not ten independent search repetitions."}, inputs


def plot_adaptive_repeated(report_dir, output):
    report, output = Path(report_dir).resolve(), Path(output).resolve()
    targets = [output, output.with_suffix(".pdf"), output.with_suffix(".json")]
    require(output.suffix == ".svg" and not any(path.exists() for path in targets)
            and not output.is_relative_to(report), "Use fresh SVG/PDF/JSON paths outside the immutable report")
    data, inputs = load_data(report)
    with plt.rc_context({"font.size": 9, "svg.fonttype": "none", "svg.hashsalt": "adaptive-repeated-v1",
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.subplots_adjust(left=.07, right=.98, bottom=.15, top=.94, hspace=.48, wspace=.25)
        ax = axes[0, 0]
        primary = data["primary_across_searches"]
        differences = [row["aggregate"]["mean"] for row in data["primary_repetitions"]]
        ax.axhline(0, color=".5", ls="--", lw=.8)
        ax.scatter([0, 1], differences, color=["#2263A5", "#C65F31"], s=42)
        ax.errorbar(2.2, primary["mean"], yerr=primary["sample_sd"], fmt="D", color="#25313B", capsize=4)
        ax.set(xticks=[0, 1, 2.2], xticklabels=["Outer search 1", "Outer search 2", "Mean ± SD\n(n = 2)"],
               xlim=(-.5, 2.8), ylabel="J evolutionary − J independent",
               title="A  Primary differences across outer searches")
        identities = list(data["conditions"])
        labels = ["E1", "I1", "E2", "I2", "Identity", "Arithmetic", "FocusGA"]
        ax = axes[0, 1]
        for offset, key, color, label in zip((-.22, 0, .22), SCORES,
                                             ("#25313B", "#2263A5", "#C65F31"),
                                             ("Combined", "Active", "Previous"), strict=True):
            means = [data["conditions"][identity]["scores"][key]["mean"] for identity in identities]
            deviations = [data["conditions"][identity]["scores"][key]["sample_sd"] for identity in identities]
            ax.errorbar([i + offset for i in range(7)], means, yerr=deviations, fmt="o", ms=3.5,
                        lw=.7, capsize=2, color=color, label=label)
        ax.set(xticks=range(7), xticklabels=labels, ylabel="Normalized score (mean ± seed SD)",
               title="B  Score components by finalist and control")
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower left")
        colors = ("#2263A5", "#2263A5", "#C65F31", "#C65F31", "#7F7F7F", "#188577", "#8765A3")
        styles = ("-", "--", "-", "--", ":", "-.", ":")
        for identity, label, color, style in zip(identities, labels, colors, styles, strict=True):
            condition = data["conditions"][identity]
            curve = [statistics.mean(row["returns"][generation] for row in condition["active_curves"])
                     for generation in range(320)]
            axes[1, 0].plot(range(1, 321), curve, color=color, ls=style, lw=1.1, label=label)
            previous = [statistics.mean(row["phases"][phase]["previous_mean"] for row in condition["phase_returns"])
                        for phase in (1, 2, 3)]
            axes[1, 1].plot([160, 240, 320], previous, color=color, ls=style, marker="o", ms=3, lw=1.1)
        axes[1, 0].set(xlabel="Completed generations", ylabel="Mean active-task return", ylim=(0, 515),
                       title="C  Learning curves (five paired task draws)")
        axes[1, 1].set(xlabel="Completed generations at checkpoint", ylabel="Mean previous-task return",
                       xticks=[160, 240, 320], ylim=(0, 515), title="D  Fresh retention at phase endpoints")
        for switch in (80.5, 160.5, 240.5):
            axes[1, 0].axvline(switch, color=".7", ls="--", lw=.6)
        for ax in axes.flat:
            ax.grid(axis="y", alpha=.2)
        handles, labels = axes[1, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=7, loc="lower center", bbox_to_anchor=(.5, .045), frameon=False)
        fig.text(.07, .025, "E/I: evolutionary/independent finalist. Bars: sample SD, not confidence intervals. "
                 "Task draws are shared; primary n = 2 outer repetitions.", fontsize=8)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".adaptive-repeated-", dir=output.parent) as temporary:
            stage = Path(temporary)
            svg, pdf = stage / output.name, stage / output.with_suffix(".pdf").name
            fig.savefig(svg, metadata={"Date": None})
            fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
            provenance = {"schema_version": 1, "kind": "adaptive_repeated_search_comparison",
                          "input_report_name": report.name, "input_sha256": inputs,
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
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_adaptive_repeated(args.report_dir, args.output)
