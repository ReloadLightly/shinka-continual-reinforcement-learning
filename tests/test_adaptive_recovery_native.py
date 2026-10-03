"""Real pinned Shinka runner/DB/patcher/scheduler; no model or RL training calls."""

from pathlib import Path
import os
import random
import textwrap

import pytest

from shinka_crl import adaptive_recovery_native as recovery
from shinka_crl.pilot import read_json, sha256, write_json
from shinka_crl.reference_timing import artifact_hashes
from shinka_crl.search import database_snapshot


EVALUATOR = '''\
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--program_path')
parser.add_argument('--results_dir')
args = parser.parse_args()
program, result = Path(args.program_path), Path(args.results_dir)
generation = int(program.parent.name.split('_')[-1])
root = result / 'evaluation'
root.mkdir(parents=True)
def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\\n')
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'program.py').write_bytes(program.read_bytes())
correct = generation != int(os.environ.get('SYNTHETIC_FAIL_GENERATION', '-1'))
score = generation / 100
metrics = {'combined_score': score, 'public': {'synthetic': True},
           'private': {'request_file': 'request.json'}}
request = {'status': 'complete' if correct else 'failed', 'slot_consumed': True,
           'evaluator_model_calls': 0, 'candidate': {'source_sha256': digest(program)},
           'aggregate': {'scores': {'combined_score': {'mean': score}}}}
write(root / 'request.json', request)
write(root / 'correct.json', {'correct': correct, 'error': None if correct else 'synthetic'})
write(root / 'metrics.json', metrics)
write(root / 'receipt.json', {p.name: digest(p) for p in sorted(root.iterdir())})
for name in ('correct.json', 'metrics.json'):
    (result / name).write_bytes((root / name).read_bytes())
write(result / 'synthetic-process.json', {'generation': generation, 'pid': os.getpid()})
if os.environ.get('SYNTHETIC_TAMPER') == str(generation):
    (root / 'program.py').write_text('# corrupted after seal\\n')
'''


@pytest.fixture
def native(tmp_path, monkeypatch):
    pytest.importorskip("shinka")
    import numpy as np
    from shinka.database import DatabaseConfig, Program, ProgramDatabase
    from shinka.llm.llm import AsyncLLMClient
    from shinka.llm.providers import QueryResult

    # All provider traffic is replaced at its public async query boundary.
    monkeypatch.setenv("SHINKA_PRICING_MODE", "offline")
    monkeypatch.setenv("SHINKA_HEADLESS_COMMAND", "/bin/true")
    monkeypatch.delenv("SHINKA_RECOVERY_GENERATION", raising=False)
    work = tmp_path / "work"
    task, archive = work / "task", work / "shinka"
    task.mkdir(parents=True)
    archive.mkdir()
    code = ("# EVOLVE-BLOCK-START\n"
            "def update_sigma(sigma, stats, memory):\n    return sigma, memory\n"
            "# EVOLVE-BLOCK-END\n")
    (task / "initial.py").write_text(code)
    (task / "evaluate.py").write_text(EVALUATOR)
    (task / "shinka-subscription.yaml").write_text(textwrap.dedent("""\
        evo:
          task_sys_msg: 'Synthetic isolated native recovery integration fixture.'
          llm_models: ['headless/codex?effort=medium']
          llm_dynamic_selection: null
          llm_kwargs: {}
          embedding_model: null
          meta_rec_interval: null
          meta_llm_models: null
          novelty_llm_models: null
          evolve_prompts: false
          prompt_llm_models: null
          enable_wandb_logging: false
          max_novelty_attempts: 1
          max_patch_resamples: 1
          max_patch_attempts: 1
          patch_types: [full]
          patch_type_probs: [1.0]
          enable_controlled_oversubscription: false
        db:
          num_islands: 1
          archive_size: 20
          num_archive_inspirations: 1
          num_top_k_inspirations: 1
        job:
          time: '00:01:00'
          numeric_threads_per_job: 1
        max_evaluation_jobs: 1
        max_proposal_jobs: 1
        max_db_workers: 1
        verbose: false
        """))
    database = ProgramDatabase(DatabaseConfig(db_path=str(archive / "programs.sqlite"),
                                             num_islands=1), embedding_model=None)
    for generation in range(15):
        directory = archive / f"gen_{generation}"
        directory.mkdir()
        (directory / "main.py").write_text(code)
        database.add(Program(id=f"historical-{generation}", generation=generation,
                             code=code, correct=generation != 14,
                             combined_score=0.1 if generation != 14 else 0,
                             parent_id="historical-0" if generation else None))
    database.close()
    (archive / "gen_15").mkdir()
    (archive / "gen_15/.generation_lock").write_text("")
    rng = np.random.RandomState(1701).get_state()
    write_json(archive / "rng_state.json", {"python": random.Random(1701).getstate(),
                                            "numpy": [rng[0], rng[1].tolist(), *rng[2:]]})
    calls = []

    async def query(self, msg, system_msg, **kwargs):
        generation = int(os.environ["SHINKA_RECOVERY_GENERATION"])
        assert read_json(work / f"recovery_slots/gen_{generation}.json")["generation"] == generation
        # At every subsequent actual provider invocation, the preceding row and
        # sealed result must already exist; this exposes the original native race.
        if generation > 16:
            previous = database_snapshot(archive / "programs.sqlite")[-1]
            assert previous["generation"] == generation - 1 and previous["correct"] == 1
            assert (archive / f"gen_{generation - 1}/results/evaluation/receipt.json").is_file()
        calls.append({"generation": generation, "prompt": msg, "system": system_msg})
        if os.environ.get("SYNTHETIC_PROVIDER_FAIL") == str(generation):
            return None
        content = ("<NAME>synthetic</NAME><DESCRIPTION>Integration fixture</DESCRIPTION>\n"
                   "```python\n" + code.replace("return sigma, memory",
                                               f"return sigma * {generation}.0, memory") + "```\n")
        return QueryResult(content=content, msg=msg, system_msg=system_msg,
                           new_msg_history=[], model_name="headless/codex?effort=medium",
                           kwargs={}, input_tokens=1, output_tokens=1)

    monkeypatch.setattr(AsyncLLMClient, "query", query)
    return work, calls


def assert_no_live_evaluators(work):
    for path in (work / "shinka").glob("gen_*/results/synthetic-process.json"):
        with pytest.raises(ProcessLookupError):
            os.kill(read_json(path)["pid"], 0)


def test_native_recovery_reaches_25_without_reusing_failed_or_canceled_slots(native):
    work, calls = native
    before = database_snapshot(work / "shinka/programs.sqlite")
    old_artifacts = {generation: artifact_hashes(work / f"shinka/gen_{generation}")
                     for generation in range(16)}
    source = Path(__import__("shinka.core.async_runner", fromlist=["__file__"]).__file__)
    source_hash = sha256(source)
    old_rng = sha256(work / "shinka/rng_state.json")
    assert recovery.native_main(work, target=25) == 0
    rows = database_snapshot(work / "shinka/programs.sqlite")
    receipt = read_json(work / "native-recovery.json")
    assert rows[:15] == before
    assert old_artifacts == {generation: artifact_hashes(work / f"shinka/gen_{generation}")
                             for generation in range(16)}
    assert len(rows) == 24
    assert [row["generation"] for row in rows] == [*range(15), *range(16, 25)]
    assert [call["generation"] for call in calls] == list(range(16, 25))
    by_id = {row["id"]: row for row in rows}
    for row in rows[15:]:
        parent = by_id[row["parent_id"]]
        assert parent["correct"] == 1 and parent["generation"] < row["generation"]
    assert receipt["status"] == "complete"
    assert receipt["completed_generations"] == receipt["next_generation_to_submit"] == 25
    assert receipt["consumed_generations"] == list(range(25))
    assert receipt["verified_generations"] == list(range(16, 25))
    assert receipt["graceful_return"] and receipt["rng_saved"]
    assert receipt["initial_rng_sha256"] == old_rng
    assert receipt["final_rng_sha256"] == sha256(work / "shinka/rng_state.json") != old_rng
    assert sha256(source) == source_hash
    assert not (work / "shinka/gen_25").exists()
    assert list((work / "shinka/gen_15").iterdir()) == [work / "shinka/gen_15/.generation_lock"]
    assert "SHINKA_RECOVERY_GENERATION" not in os.environ
    assert_no_live_evaluators(work)


@pytest.mark.parametrize("failure", ["evaluation", "proposal", "receipt"])
def test_native_terminal_failure_stops_before_next_slot(native, monkeypatch, failure):
    work, calls = native
    environment = {"evaluation": "SYNTHETIC_FAIL_GENERATION",
                   "proposal": "SYNTHETIC_PROVIDER_FAIL", "receipt": "SYNTHETIC_TAMPER"}
    monkeypatch.setenv(environment[failure], "16")
    before = database_snapshot(work / "shinka/programs.sqlite")
    assert recovery.native_main(work, target=25) == 1
    rows = database_snapshot(work / "shinka/programs.sqlite")
    receipt = read_json(work / "native-recovery.json")
    assert rows[:15] == before
    assert [call["generation"] for call in calls] == [16]
    assert receipt["status"] == "failed" and receipt["terminal_reason"]
    assert receipt["attempted_generations"] == [16]
    assert receipt["consumed_generations"] == list(range(17))
    assert receipt["graceful_return"] and receipt["rng_saved"]
    assert not (work / "recovery_slots/gen_17.json").exists()
    assert not (work / "shinka/gen_17").exists()
    if failure == "proposal":
        assert len(rows) == 15
        assert (work / "shinka/gen_16/failure.json").exists()
        assert not (work / "shinka/gen_16/results/synthetic-process.json").exists()
    else:
        assert len(rows) == 16
        assert rows[-1]["correct"] == (0 if failure == "evaluation" else 1)
    assert_no_live_evaluators(work)
    # The same working copy cannot silently retry either a failed row or no-row slot.
    with pytest.raises(ValueError, match="receipt already exists"):
        recovery.native_main(work, target=25)
    assert len(calls) == 1


def test_preexisting_reservation_is_not_reused(native):
    work, calls = native
    recovery._reserve(work, 16)
    original = (work / "recovery_slots/gen_16.json").read_bytes()
    assert recovery.native_main(work, target=25) == 1
    assert calls == []
    assert (work / "recovery_slots/gen_16.json").read_bytes() == original
    assert not (work / "shinka/gen_16").exists()
    assert "FileExistsError" in read_json(work / "native-recovery.json")["terminal_reason"]
    assert read_json(work / "native-recovery.json")["consumed_generations"] == list(range(17))


def test_failure_at_target_is_not_reported_as_success(native, monkeypatch):
    work, calls = native
    monkeypatch.setenv("SYNTHETIC_FAIL_GENERATION", "16")
    assert recovery.native_main(work, target=17) == 1
    receipt = read_json(work / "native-recovery.json")
    assert receipt["completed_generations"] == 17
    assert receipt["status"] == "failed"
    assert receipt["verified_generations"] == []
    assert [call["generation"] for call in calls] == [16]


def test_wrong_starting_rows_rejected_before_native_or_provider(native):
    work, calls = native
    with pytest.raises(ValueError, match="starting native rows"):
        recovery.native_main(work, target=25, start=15, missing=(14,))
    assert calls == []
    assert not (work / "recovery_slots").exists()
