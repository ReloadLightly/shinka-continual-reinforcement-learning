"""Independent standard-library audit of the reserved freeze; no model/training calls."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import subprocess

ROOT = Path.cwd()
FROZEN = ROOT / 'reports/adaptive-validation-freeze-20261003'
CLOSURE = ROOT / 'reports/adaptive-endpoint-closure-20261003.json'
OUT = ROOT / 'reports/adaptive-validation-prepared-review-20261003.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def inventory(path):
    result = {}
    for item in sorted(path.rglob('*')):
        assert not item.is_symlink(), str(item)
        if item.is_file():
            result[str(item.relative_to(path))] = sha(item)
    return result


def verify_receipt(path):
    actual = inventory(path)
    actual.pop('receipt.json')
    assert actual == read(path / 'receipt.json'), str(path)


plan = read(FROZEN / 'plan.json')
closure = read(CLOSURE)
assert closure['status'] == 'closed' and closure['slots_consumed'] == closure['target_slots'] == 25
assert closure['remaining_generations'] == [] and closure['native_status'] == 'complete'
assert sha(ROOT / closure['independent_review']) == closure['independent_review_sha256']
assert plan['provenance']['closure'] == closure
assert plan['provenance']['closure_sha256'] == sha(CLOSURE)
verify_receipt(FROZEN)
assert sha(FROZEN / 'protocol.md') == plan['protocol_sha256']
assert (FROZEN / 'protocol.md').read_text().replace('This proposal makes no', 'This protocol makes no').replace('this proposed comparison does not supply', 'this comparison does not supply') == (ROOT / 'docs/adaptive-validation.md').read_text()
assert plan['kind'] == 'reserved' and plan['protocol_version'] == 'adaptive-reserved-validation-v1'
revision = plan['implementation_revision']
for name, expected in plan['source_sha256'].items():
    assert sha(ROOT / name) == expected, name
    committed = subprocess.check_output(['git', 'show', f'{revision}:{name}'], cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest() == expected, name
native = Path(plan['upstream'])
assert subprocess.check_output(['git', '-C', str(native), 'rev-parse', 'HEAD'], text=True).strip() == plan['upstream_commit']
for name, expected in plan['native_source_sha256'].items():
    assert sha(native / name) == expected, name
    committed = subprocess.check_output(['git', '-C', str(native), 'show', f'{plan["upstream_commit"]}:{name}'])
    assert hashlib.sha256(committed).hexdigest() == expected, name

report = ROOT / closure['endpoint_report']
assert sha(report / 'summary.json') == closure['endpoint_summary_sha256']
assert sha(report / 'checksums.json') == closure['endpoint_checksums_sha256']
endpoint_checks = read(report / 'checksums.json')
for name, expected in endpoint_checks['published_sha256'].items():
    assert sha(report / name) == expected, name
work = Path(closure['endpoint_work'])
original = read(work / 'plan.json')
study = Path(original['evaluation_study'])
development = read(study / 'plan.json')
assert digest(development) == plan['provenance']['development_plan_sha256']
assert development['profile']['seeds'] == [4001, 4002, 4003]
assert plan['runtime'] == development['runtime']
assert plan['cpu_affinity'] == development['cpu_affinity'] == [0, 1]
assert plan['thread_environment'] == development['thread_environment']
assert plan['python'] == development['python']
with sqlite3.connect(f'file:{work / "shinka/programs.sqlite"}?mode=ro', uri=True) as db:
    rows = db.execute('SELECT id,generation,code,correct,combined_score FROM programs ORDER BY generation').fetchall()
valid = [row for row in rows if row[3]]
assert len(rows) == 24 and len(valid) == 21
assert [r[1] for r in rows] == [g for g in range(25) if g != 15]
assert [r[1] for r in rows if not r[3]] == [14, 17, 20]
ranked = []
verified_caches = set()
trials_checked = 0
for native_id, generation, code, _, database_score in valid:
    request_dir = work / f'shinka/gen_{generation}/results/evaluation'
    verify_receipt(request_dir)
    request = read(request_dir / 'request.json')
    assert request['status'] == 'complete'
    assert code == (request_dir / 'program.py').read_text()
    metadata = request['candidate']['program']
    parsed = ast.parse(code)
    assert hashlib.sha256(ast.dump(parsed, annotate_fields=True, include_attributes=False).encode()).hexdigest() == metadata['canonical_ast_sha256']
    assert sha(request_dir / 'program.py') == metadata['source_sha256']
    assert sum(1 for _ in ast.walk(parsed)) == metadata['ast_nodes'] <= 512
    cache = study / request['cache_origin']
    if cache not in verified_caches:
        verify_receipt(cache)
        verified_caches.add(cache)
    assert sha(cache / 'receipt.json') == request['cache_receipt_sha256']
    cached = read(cache / 'summary.json')
    assert cached['status'] == 'complete'
    scores = []
    trial_records = []
    for seed in (4001, 4002, 4003):
        directory = cache / f'seed_{seed}'
        records = read(directory / 'training/training_metrics.json')
        assert len(records) == 80
        active = []
        for index, metric in enumerate(records):
            task = (index // 20) % 2
            assert metric['generation'] == index and metric['task'] == task
            value = metric[f'centroid_task{task}']
            assert math.isfinite(value) and 0 <= value <= 500
            active.append(value)
        active_score = (sum(active) / len(active)) / 500
        evaluation = read(directory / 'analysis/evaluation.json')
        assert evaluation['trial'] == seed + 1 and evaluation['eval_seed'] == seed + 900000
        assert evaluation['episodes'] == 10
        centroid = [r for r in evaluation['per_task'] if r['source'] == 'centroid']
        assert [r['task_idx'] for r in centroid] == list(range(4))
        previous = []
        for entry in centroid[1:]:
            assert len(entry['prev_returns']) == 10
            assert all(math.isfinite(v) and 0 <= v <= 500 for v in entry['prev_returns'])
            previous.append(sum(entry['prev_returns']) / 10)
        previous_score = (sum(previous) / 3) / 500
        combined = 0.5 * active_score + 0.5 * previous_score
        saved = next(r for r in cached['trials'] if r['seed'] == seed)
        assert saved['score']['active_score'] == active_score
        assert saved['score']['previous_score'] == previous_score
        assert saved['score']['combined_score'] == combined
        assert read(directory / 'training/config.json')['seed'] == seed
        episodes = sum(len(v) for e in evaluation['per_task'] for k, v in e.items() if k.endswith('returns'))
        assert episodes == 300
        trial_records.append({'seed': seed, 'combined_score': combined, 'active_score': active_score,
                              'previous_score': previous_score})
        scores.append(combined)
        trials_checked += 1
    mean = statistics.mean(scores)
    assert mean == request['aggregate']['scores']['combined_score']['mean'] == database_score
    ranked.append({'generation': generation, 'native_id': native_id, 'development_score': mean,
                   'development_seeds': [4001, 4002, 4003], 'source': str(request_dir / 'program.py'),
                   'program': metadata, 'request_receipt_sha256': sha(request_dir / 'receipt.json'),
                   'cache_receipt_sha256': request['cache_receipt_sha256'], 'trials': trial_records})
ranked.sort(key=lambda r: (-r['development_score'], r['generation']))
assert [{k: v for k, v in r.items() if k != 'trials'} for r in ranked] == plan['provenance']['ranked_candidates']
assert trials_checked == 63
winner = ranked[0]
assert winner['generation'] == plan['provenance']['selected_generation'] == 5
assert winner['native_id'] == plan['provenance']['selected_native_id']
assert winner['development_score'] == plan['provenance']['selected_development_score']
assert 0 in [r['generation'] for r in ranked]

conditions = plan['conditions']
assert [c['id'] for c in conditions] == ['selected', 'identity', 'arithmetic', 'focus', 'static_shinka11', 'static_random24']
for candidate, control in zip(conditions[1:], development['controls'], strict=True):
    assert {k: v for k, v in candidate.items() if k not in ('source', 'program')} == {k: v for k, v in control.items() if k not in ('source', 'program')}
for candidate in conditions:
    if candidate['source']:
        assert sha(FROZEN / candidate['source']) == candidate['source_sha256']
        if candidate['variant'] == 'ga_adaptive':
            tree = ast.parse((FROZEN / candidate['source']).read_text())
            assert candidate['program'] == {'source_sha256': candidate['source_sha256'], 'canonical_ast_sha256': hashlib.sha256(ast.dump(tree, annotate_fields=True, include_attributes=False).encode()).hexdigest(), 'ast_nodes': sum(1 for _ in ast.walk(tree)), 'grammar_version': 'adaptive-width-v1'}
assert (FROZEN / conditions[0]['source']).read_bytes() == Path(winner['source']).read_bytes()
assert conditions[0]['program'] == winner['program']
profile = read(ROOT / 'src/shinka_crl/profiles/adaptive-validation.json')
assert plan['profile'] == profile and profile['seeds'] == [5001, 5002, 5003, 5004, 5005]
assert profile['ne'] == {'num_generations': 320, 'task_interval': 80, 'pop_size': 64, 'num_evals': 3}
context_keys = ('protocol_version','kind','profile','objective_version','objective_weights','trial_offset',
                'eval_seed_offset','posthoc_episodes','upstream','upstream_commit','python','timeout_seconds',
                'cpu_affinity','thread_environment','runtime','source_sha256')
context = {k: plan[k] for k in context_keys}
focus_settings = {'elite_ratio': .5, 'init_around_mean': False, 'cross_over_rate': 0.,
                  'focus_rate': .3, 'sigma_rate': .1, 'track_target': .9, 'sigma_min': .00001,
                  'explore_fraction': .25}
unique = []
for candidate in conditions:
    focus = candidate['variant'] == 'ga_focus'
    static = candidate['settings'] or {'sigma': .5, 'elite_ratio': .5}
    archive = max(1, int(64 * static['elite_ratio']))
    recipe = {'native_method': 'ga_focus' if focus else 'ga', 'variant': candidate['variant'],
              'program_identity': candidate.get('program', {}).get('canonical_ast_sha256') or candidate['source_sha256'],
              'sigma_initial': static['sigma'], 'archive_fraction': static['elite_ratio'],
              'archive_size': archive, 'offspring_count': 64 - archive - int(focus),
              'memory_initial': [0., 0., 0., 0.] if candidate['variant'] == 'ga_adaptive' else None,
              'focus_settings': focus_settings if focus else None, 'centroid_inside_population_budget': focus,
              'adapter': {'grammar': 'adaptive-width-v1', 'width_bounds': [.001, 2.],
                          'source_sha256': plan['source_sha256']['src/shinka_crl/adaptive.py']}
              if candidate['variant'] == 'ga_adaptive' else None,
              'settings': candidate['settings'],
              'native_fixed': {'hidden_dims': [16, 16], 'num_params': 386, 'objective': 'mean',
                               'obs_norm': False, 'first_task_clean': True, 'task_warmup': 0,
                               'init_around_mean': False, 'refresh': True, 'variation': 'gaussian',
                               'cross_over_rate': 0., 'schedule': 'switch', 'noise_range': .5},
              'evaluation_context_sha256': digest(context)}
    unique.append({'recipe_key': digest(recipe), 'recipe': recipe, 'candidate': candidate,
                   'memberships': [candidate['id']]})
assert len({r['recipe_key'] for r in unique}) == 6 and unique == plan['unique_recipes']
trials = [{'index': 6 * i + j, 'recipe_key': recipe['recipe_key'], 'memberships': recipe['memberships'],
           'seed': seed, 'trial': seed + 1, 'eval_seed': seed + 900000}
          for i, seed in enumerate(range(5001, 5006)) for j, recipe in enumerate(unique)]
assert trials == plan['trial_order'] and plan['planned_trials'] == 30
steps = 320 * 64 * 3 * 500
assert plan['trial_steps_nominal'] == steps == 30720000
assert plan['planned_nominal_training_steps'] == 30 * steps == 921600000
assert plan['planned_fresh_evaluation_episodes'] == 30 * 300 == 9000
assert plan['first_block_trials'] == 6 and plan['session_limit_seconds'] == 14400
assert plan['timeout_seconds'] == 1800 and plan['posthoc_episodes'] == 10
assert plan['objective_weights'] == {'active': .5, 'previous': .5}
assert plan['objective_version'] == 'adaptive-active-previous-v1'
assert plan['reporting']['primary'] == 'paired five-seed combined-score selected minus focus mean and sample SD'
assert plan['reporting']['secondary'] == ['identity', 'arithmetic', 'static_shinka11', 'static_random24']
assert plan['reporting']['early_stop'] == 'integrity or fixed resource limit only; never score-dependent'
assert plan['reporting']['inference'] == 'descriptive only; no significance threshold or promotion rule'
config_paths = list((ROOT / 'results').rglob('config.json'))
reserved_hits, paper_hits = [], []
for path in config_paths:
    config = read(path)
    if config.get('seed') in range(5001, 5006) or config.get('trial') in range(5002, 5007):
        reserved_hits.append(str(path.relative_to(ROOT)))
    if config.get('seed') in range(42, 52) or config.get('trial') in range(1, 11):
        paper_hits.append(str(path.relative_to(ROOT)))
assert not reserved_hits and not paper_hits
result = {
    'schema_version': 1, 'status': 'passed_prepared_review',
    'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
    'freeze': str(FROZEN.relative_to(ROOT)), 'closure': str(CLOSURE.relative_to(ROOT)),
    'implementation_revision': revision,
    'methods': {'independence': 'Standard-library hashes, JSON, AST, SQLite read-only queries, Git source reads, direct arithmetic, and statistics.mean; no repository scoring, selection, or recipe functions invoked.',
                'script': 'reports/adaptive-validation-prepared-checks-20261003/audit.py',
                'script_sha256': sha(Path(__file__)), 'score_comparisons': 'Exact floating-point equality using the frozen summation order; ranking uses unrounded means and earlier generation for exact ties.'},
    'bindings': {'plan_sha256': sha(FROZEN / 'plan.json'), 'freeze_receipt_sha256': sha(FROZEN / 'receipt.json'),
                 'protocol_sha256': sha(FROZEN / 'protocol.md'), 'closure_sha256': sha(CLOSURE),
                 'endpoint_summary_sha256': sha(report / 'summary.json'),
                 'endpoint_checksums_sha256': sha(report / 'checksums.json')},
    'verification': {'frozen_files': len(read(FROZEN / 'receipt.json')), 'committed_repository_source_files': len(plan['source_sha256']),
                     'committed_native_source_files': len(plan['native_source_sha256']),
                     'endpoint_published_hashes': len(endpoint_checks['published_sha256']),
                     'ranked_candidates': 21, 'recomputed_development_seed_trials': trials_checked,
                     'cache_receipts_verified': len(verified_caches), 'identity_included': True,
                     'full_recipe_identities_rederived': True, 'control_memberships_and_settings_unchanged': True,
                     'partition_and_trial_order_rederived': True, 'budget_rederived': True, 'failures': []},
    'selected': {'generation': winner['generation'], 'native_id': winner['native_id'],
                 'development_mean': winner['development_score'], 'source_sha256': winner['program']['source_sha256'],
                 'canonical_ast_sha256': winner['program']['canonical_ast_sha256']},
    'ranked_candidates': ranked,
    'frozen_program_sha256': {c['id']: c['source_sha256'] for c in conditions if c['source']},
    'focus_native_source_sha256': {name: value for name, value in plan['native_source_sha256'].items() if 'focus' in name.lower() or name == 'source/algorithms/ne/ga.py'},
    'recipes': unique,
    'protocol_copy': {'frozen_sha256': sha(FROZEN / 'protocol.md'), 'current_document_sha256': sha(ROOT / 'docs/adaptive-validation.md'), 'reviewed_difference': 'Only This proposal → This protocol and this proposed comparison → this comparison; frozen copy remains authoritative'},
    'audit_attempts': [{'log': 'reports/adaptive-validation-prepared-checks-20261003/audit-attempt001.log', 'exit_code': 1, 'reason': 'Initial audit required byte equality with the live protocol document; its two wording-only corrections after freeze were independently diff-reviewed. Frozen protocol receipt remained correct.'}, {'log': 'reports/adaptive-validation-prepared-checks-20261003/audit-attempt002.log', 'exit_code': 1, 'reason': 'Initial control comparison treated newly derived identity/arithmetic AST metadata as changed control settings. Source, variant, id, and settings are unchanged; added metadata is independently rederived from frozen source.'}, {'log': 'reports/adaptive-validation-prepared-checks-20261003/audit.log', 'exit_code': 0}],
    'allocation': {'unique_recipes': 6, 'trials': 30, 'first_block_trials': 6,
                   'nominal_steps_per_trial': steps, 'nominal_training_steps': 30 * steps,
                   'fresh_evaluation_episodes': 9000, 'validation_seeds': list(range(5001, 5006)),
                   'task_trials': list(range(5002, 5007)), 'evaluation_seeds': list(range(905001, 905006)),
                   'cpu_affinity': [0, 1], 'cumulative_session_seconds': 14400, 'per_process_timeout_seconds': 1800},
    'partition_audit': {'scope': 'All local results/config.json recursively at review time',
                        'files_scanned': len(config_paths), 'reserved_validation_hits': reserved_hits,
                        'paper_reporting_hits': paper_hits},
    'execution_gate': 'Prepared content passes this audit. Exact freeze, closure, review, and completed full-suite/lint evidence must be committed and pushed before the first reserved block; this audit launches no training or model calls.',
    'limitations': ['One development search and one selected rule; no claim about search-method superiority.',
                    'Five reserved validation seeds and longer phases remain separate from paper reporting.',
                    'Preserve failed search slots and RNG recovery deviations; no validation feedback may reopen proposal search.']}
OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': result['status'], 'selected_generation': winner['generation'],
                  'selected_development_mean': winner['development_score'], 'candidate_seed_trials_recomputed': trials_checked,
                  'planned_trials': 30, 'planned_nominal_steps': 30 * steps, 'reserved_hits': reserved_hits,
                  'paper_hits': paper_hits}))
