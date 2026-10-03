"""Reference figures preserve native clocks, signed forgetting, and evidence partitions."""

import importlib.util
import json
from pathlib import Path

import pytest

from shinka_crl import analysis
from shinka_crl.experiment import UPSTREAM_COMMIT, load_profile, nominal_training_steps, score_curve

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_reference_comparison", ROOT / "scripts/plot_reference_comparison.py")
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def seal(root):
    write(root / "checksums.json", {"published_sha256": {
        str(p.relative_to(root)): plotter.digest(p) for p in root.rglob("*")
        if p.is_file() and p.name != "checksums.json"}})


def alter(root, relative, change):
    path = root / relative
    value = json.loads(path.read_text())
    change(value)
    write(path, value)
    seal(root)


def config(method, profile, job):
    budget = profile["ppo"] if method == "ppo" else profile["ne"]
    cfg = {"env": "CartPole-v1", "method": method, "hidden_dims": [16, 16], "num_params": 386,
           "first_task_clean": True, "task_warmup": 0, "seed": job["seed"], "trial": job["trial"],
           "episode_length": 500, "num_tasks": 2, "task_sequence": [i % 2 for i in range(20)],
           "eval_episodes": 10, "task_interval": budget["task_interval"], "pop_size": 512,
           "num_generations": budget["num_updates" if method == "ppo" else "num_generations"]}
    if method in ("ga", "es"):
        cfg.update(objective="mean", obs_norm=False, num_evals=3)
    if method == "ga":
        cfg.update(sigma=.5, searcher_kwargs={"elite_ratio": .5, "init_around_mean": False},
                   searcher_resolved={"refresh": True, "num_elites": 256, "num_offspring": 256,
                                      "variation": "gaussian", "sigma": .5, "cross_over_rate": 0.})
    elif method == "es":
        cfg.update(sigma=.1, learning_rate=.05, optimizer="sgd", shaping="zscore", sigma_lr=0.,
                   searcher_kwargs={}, searcher_resolved={})
    else:
        cfg.update(learning_rate=.0003, optimizer="adam", adam_b1=.9, num_epochs=10,
                   gamma=.95, ent_coef=.01, clip_eps=.2, head="categorical", normalize_obs=False,
                   reward_scale=1., eval_interval=1, value_hidden_dims=[128, 128, 128],
                   num_envs=2048, num_steps=50, num_minibatches=32, num_timesteps=3072000000)
    return cfg


@pytest.fixture
def synthetic(tmp_path):
    def create(mode="development", methods=None, all_trials=False):
        root = tmp_path / mode
        profile = load_profile("paper-cartpole")
        methods = methods or (["es"] if mode == "development" else list(plotter.METHODS))
        if mode == "development":
            profile["seeds"] = [1001]
            profile["purpose"] = "Synthetic test evidence; never a trained result"
        jobs = [{"profile": "paper-cartpole", "method": method, "seed": seed,
                 "trial": i if mode == "reporting" else seed + 1, "eval_seed": seed + 900000}
                for i, seed in enumerate(profile["seeds"], 1) for method in methods]
        plan = {"mode": mode, "profiles": {"paper-cartpole": profile}, "methods": methods,
                "jobs": jobs, "eval_episodes": 10,
                "source_sha256": {"src/shinka_crl/analysis.py": plotter.digest(analysis.__file__)}}
        rows, groups = [], []
        for job in (jobs if all_trials else jobs[:len(methods)]):
            method = job["method"]
            cfg = config(method, profile, job)
            vectors = [[0., 0., 0., 0.], [.1, .2, .3, .4]]
            manifest = {**job, "profile": profile, "status": "complete", "upstream_commit": UPSTREAM_COMMIT}
            results = {"config": cfg, "noise_vectors": vectors, "env_steps": nominal_training_steps(profile, method)}
            records = [{"generation": i, "task": (i // cfg["task_interval"]) % 2,
                        f"centroid_task{(i // cfg['task_interval']) % 2}": float(200 + i % 25)}
                       for i in range(cfg["num_generations"])]
            source = "final" if method == "ppo" else "centroid"
            evaluation = {"method": method, "env": "CartPole-v1", "trial": job["trial"], "pop_size": 512,
                          "episodes": 10, "eval_seed": job["eval_seed"], "num_tasks": 20,
                          "agent_sources": [source], "per_task": []}
            for phase in range(20):
                row = {"task_idx": phase, "source": source, "returns": [100. + phase] * 10}
                if phase:
                    row["prev_returns"] = [104. + phase] * 10
                if phase < 19:
                    row["zero_shot_next_returns"] = [110. + phase] * 10
                evaluation["per_task"].append(row)
            metadata = {"sources": {source: [20, 386]}, "finite": True,
                        "noise_vectors": [vectors[i % 2] for i in range(20)]}
            measured = analysis.summarize_trial(manifest=manifest, results=results, records=records,
                evaluation=evaluation, checkpoint_metadata=metadata, episodes=10, eval_seed=job["eval_seed"])
            base = f"trials/{method}" + (f"/seed_{job['seed']}" if all_trials else "")
            train, posthoc = f"{base}/training/attempt_0001", f"{base}/analysis/attempt_0001"
            for name, value in (("manifest.json", manifest), ("results.json", results),
                                ("training_metrics.json", records),
                                ("summary.json", score_curve(records, profile=profile, method=method))):
                write(root / "raw" / train / name, value)
            for name, value in (("summary.json", measured), ("evaluation.json", evaluation),
                                ("checkpoint-metadata.json", metadata)):
                write(root / "raw" / posthoc / name, value)
            rows.append({**job, "metrics": measured["metrics"], "phase_returns": measured["phase_returns"],
                         "phase_training_returns": measured["phase_training_returns"],
                         "training_env_steps_nominal": measured["nominal_training_steps"],
                         "training_path": train, "analysis_path": posthoc})
        for method in methods:
            trials = [row for row in rows if row["method"] == method]
            groups.append({"method": method, "n": len(trials), "seeds": [r["seed"] for r in trials],
                           "metrics": {key: plotter.describe([r["metrics"][key] for r in trials])
                                       for key in trials[0]["metrics"]}})
        complete = len(rows) == len(jobs)
        write(root / "raw/plan.json", plan)
        write(root / "summary.json", {"mode": mode, "status": "complete" if complete else "partial",
              "completed_trials": len(rows), "planned_trials": len(jobs), "profiles": plan["profiles"],
              "declared_methods": methods,
              "original_three_method_target_complete": mode == "reporting" and complete and methods == list(plotter.METHODS),
              "rows": rows, "groups": groups, "validation": {"all_planned_trials": complete}})
        seal(root)
        return root
    return create


def test_real_diagnostic_is_rejected_even_with_all_exceptions():
    with pytest.raises(ValueError, match="Diagnostic"):
        plotter.load_data(ROOT / "reports/reference-comparison-diagnostic-20261003",
                          allow_development=True, allow_partial=True)


def test_development_requires_flag_and_uses_signed_raw_metrics(synthetic):
    root = synthetic()
    with pytest.raises(ValueError, match="allow-development"):
        plotter.load_data(root)
    summary, groups, _ = plotter.load_data(root, allow_development=True)
    assert summary["rows"][0]["metrics"]["forgetting"] == -5.
    assert summary["rows"][0]["metrics"]["learning_accuracy"] == 109.5
    assert groups["es"][0]["x"].tolist() == [768000 * i for i in range(1, 4001)]


def test_partial_reporting_requires_flag_and_preserves_ppo_clock(synthetic):
    root = synthetic("reporting")
    with pytest.raises(ValueError, match="allow-partial"):
        plotter.load_data(root)
    summary, groups, _ = plotter.load_data(root, allow_partial=True)
    assert summary["completed_trials"] == 3 and summary["planned_trials"] == 30
    assert len(groups["ppo"][0]["x"]) == 30000
    assert groups["ppo"][0]["x"][0] == 102400
    assert groups["ppo"][0]["x"][-1] == groups["ga"][0]["x"][-1] == 3072000000
    assert not summary["validation"]["all_planned_trials"]


def test_cannot_mark_partial_comparison_as_final(synthetic):
    root = synthetic("reporting")
    alter(root, "summary.json", lambda s: s.update(status="complete"))
    with pytest.raises(ValueError, match="Incomplete suite"):
        plotter.load_data(root, allow_partial=True)


def test_complete_amended_scope_has_twenty_trials_without_claiming_original_target(synthetic, tmp_path):
    root = synthetic("reporting", methods=["ga", "es"], all_trials=True)
    summary, groups, _ = plotter.load_data(root)
    assert summary["completed_trials"] == summary["planned_trials"] == 20
    assert all(len(trials) == 10 for trials in groups.values())
    assert not summary["original_three_method_target_complete"]
    metadata = plotter.plot_reference_comparison(root, tmp_path / "amended.svg")
    assert metadata["label"] == "GA/ES reporting; PPO deferred · 20/20 trials"
    assert metadata["declared_methods"] == ["ga", "es"]
    assert not metadata["original_three_method_target_complete"]


def test_amended_scope_requires_complete_trials_and_explicit_original_limitation(synthetic):
    root = synthetic("reporting", methods=["ga", "es"])
    with pytest.raises(ValueError, match="allow-partial"):
        plotter.load_data(root)
    summary, _, _ = plotter.load_data(root, allow_partial=True)
    assert summary["completed_trials"] == 2 and summary["planned_trials"] == 20
    alter(root, "summary.json", lambda s: s.update(original_three_method_target_complete=True))
    with pytest.raises(ValueError, match="incomplete original target"):
        plotter.load_data(root, allow_partial=True)


@pytest.mark.parametrize("relative,change,message", [
    ("summary.json", lambda s: s["rows"][0]["metrics"].update(forgetting=0), "measurements"),
    ("summary.json", lambda s: s["groups"][0]["metrics"]["forgetting"].update(mean=0), "Aggregate metric"),
    ("raw/plan.json", lambda p: p["jobs"][0].update(trial=1), "seed/task mapping"),
    ("raw/plan.json", lambda p: p["profiles"]["paper-cartpole"]["ne"].update(pop_size=64), "profile changed"),
    ("raw/trials/es/training/attempt_0001/training_metrics.json", lambda r: r[0].update(generation=1), "generation"),
    ("raw/trials/es/analysis/attempt_0001/evaluation.json", lambda e: e["per_task"][1].update(prev_returns=[500.] * 10), "analysis differs"),
])
def test_rederives_numbers_even_after_checksums_are_updated(synthetic, relative, change, message):
    root = synthetic()
    alter(root, relative, change)
    with pytest.raises(ValueError, match=message):
        plotter.load_data(root, allow_development=True)


def test_changed_or_unlisted_file_is_rejected(synthetic):
    root = synthetic()
    (root / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed published"):
        plotter.load_data(root, allow_development=True)
    (root / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="file set"):
        plotter.load_data(root, allow_development=True)


def test_actual_render_labels_development_and_saves_four_artifacts(synthetic, tmp_path):
    root = synthetic()
    output = tmp_path / "reference.svg"
    metadata = plotter.plot_reference_comparison(root, output, allow_development=True)
    assert metadata["label"] == "Development (not reporting) · 1/1 trials"
    assert metadata["smoothing"] is None
    assert metadata["native_clocks"]["es"] == {"samples": 4000, "first_step": 768000, "final_step": 3072000000}
    assert "Development (not reporting)" in output.read_text()
    assert len(metadata["outputs_sha256"]) == 4
    for name, expected in metadata["outputs_sha256"].items():
        assert plotter.digest(tmp_path / name) == expected
    saved = json.loads(output.with_suffix(".json").read_text())
    assert saved == metadata
    with pytest.raises(ValueError, match="fresh"):
        plotter.plot_reference_comparison(root, output, allow_development=True)
