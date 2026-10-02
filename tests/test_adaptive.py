"""Grammar checks plus numerical delegation checks in the pinned JAX interpreter."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from shinka_crl.adaptive import (
    AST_MAX_NODES,
    MAX_ASSIGNMENTS,
    SOURCE_MAX_BYTES,
    ProgramValidationError,
    load_program,
    validate_source,
)
from shinka_crl.search import SEARCH_THREAD_ENV


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks" / "cartpole_adaptive"
UPSTREAM = ROOT / ".upstream" / "continual_neuroevolution"
JAX_PYTHON = UPSTREAM / ".venv" / "bin" / "python"
IDENTITY = "def update_sigma(sigma, stats, memory):\n    return sigma, memory\n"


@pytest.mark.parametrize("name", ["initial.py", "halving.py", "arithmetic.py"])
def test_declared_programs_have_exact_source_hashes(name):
    path = TASK / name
    program = load_program(path)
    assert program.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert program.grammar_version == "adaptive-width-v1"
    assert 0 < program.ast_nodes <= AST_MAX_NODES


def test_raw_hash_preserves_newlines_but_ast_hash_ignores_formatting(tmp_path):
    path = tmp_path / "program.py"
    path.write_bytes(IDENTITY.replace("\n", "\r\n").encode())
    plain = validate_source(IDENTITY)
    windows = load_program(path)
    commented = validate_source("# provenance comment\n" + IDENTITY)
    assert windows.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len({p.source_sha256 for p in (plain, windows, commented)}) == 3
    assert len({p.canonical_ast_sha256 for p in (plain, windows, commented)}) == 1


def test_standard_library_grammar_does_not_import_jax():
    code = (
        "import sys; from shinka_crl.adaptive import validate_source; "
        f"validate_source({IDENTITY!r}); assert 'jax' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("body", [
    "import os\nreturn sigma, memory",
    "global sigma\nreturn sigma, memory",
    "nonlocal sigma\nreturn sigma, memory",
    "sigma += 1\nreturn sigma, memory",
    "sigma, memory = memory, sigma\nreturn sigma, memory",
    "sigma = memory = 0.5\nreturn sigma, memory",
    "memory[0] = 2\nreturn sigma, memory",
    "return __import__('os').system('false'), memory",
    "return sigma.__class__, memory",
    "return exp.__globals__, memory",
    "return jnp.exp(sigma), memory",
    "return update_sigma(sigma, stats, memory)",
    "return eval('0.5'), memory",
    "return float('nan'), memory",
    "return 1e999, memory",
    "return True, memory",
    "return 1j, memory",
    "return '0.5', memory",
    "return None, memory",
    "return sigma ** 1000000, memory",
    "return sigma // 2, memory",
    "return sigma % 2, memory",
    "return sigma if stats[0] > 0 else 0.5, memory",
    "return lambda: sigma, memory",
    "return (x for x in memory), memory",
    "return [x for x in memory], memory",
    "return sigma, {0: memory}",
    "return sigma, {memory}",
    "return sigma, (*memory,)",
    "return sigma, memory[::-1]",
    "return sigma, memory[stats[0]]",
    "return stats[-1], memory",
    "return stats[5], memory",
    "return memory[4], memory",
    "return memory[0.0], memory",
    "return exp(x=sigma), memory",
    "return clip(sigma, 0.1), memory",
    "return minimum(sigma, 0.2, 0.3), memory",
    "return sigma, stack(memory)",
    "return sigma, stack((memory, memory, memory, memory))",
    "return sigma, stack((0.0, 0.0, 0.0))",
    "return sigma, stack((sigma > 0, 0, 0, 0))",
    "return sigma, memory + stats",
    "return sigma, stats",
    "return sigma, sigma",
    "return memory, memory",
    "return sigma > 0, memory",
    "return where(sigma, sigma, 0.5), memory",
    "return where(0 < sigma < 1, sigma, 0.5), memory",
    "return sigma + (sigma > 0), memory",
    "return abs(sigma > 0), memory",
    "return sigma, memory, stats",
    "return sigma",
    "x = unknown\nreturn sigma, memory",
    "exp = 0\nreturn sigma, memory",
    "_literal = 0\nreturn sigma, memory",
    "def nested():\n    return 0\nreturn sigma, memory",
    "for x in memory:\n    sigma = x\nreturn sigma, memory",
    "while sigma:\n    sigma = 0\nreturn sigma, memory",
    "try:\n    sigma = 0\nexcept:\n    sigma = 1\nreturn sigma, memory",
    "yield sigma, memory",
])
def test_rejects_unsafe_or_out_of_contract_body(body):
    source = "def update_sigma(sigma, stats, memory):\n" + "\n".join(
        "    " + line for line in body.splitlines()
    ) + "\n"
    with pytest.raises(ProgramValidationError):
        validate_source(source)


@pytest.mark.parametrize("source", [
    "import os\n" + IDENTITY,
    IDENTITY + "print('executed')\n",
    "@print('executed')\n" + IDENTITY,
    IDENTITY.replace("memory):", "memory=print('executed')):"),
    IDENTITY.replace("sigma, stats, memory", "sigma, stats, memory, extra"),
    IDENTITY.replace("sigma, stats, memory", "sigma, stats, /, memory"),
    IDENTITY.replace("sigma, stats, memory", "sigma, *, stats, memory"),
    IDENTITY.replace("sigma, stats, memory", "*args"),
    IDENTITY.replace("sigma, stats, memory", "**kwargs"),
    IDENTITY.replace("sigma, stats, memory", "sigma: int, stats, memory"),
    IDENTITY.replace("):", ") -> int:"),
    IDENTITY.replace("def ", "async def "),
    IDENTITY + IDENTITY,
    IDENTITY.replace("return sigma, memory", '"docstring"\n    return sigma, memory'),
])
def test_rejects_module_signature_and_statement_escape_routes(source, capsys):
    with pytest.raises(ProgramValidationError):
        validate_source(source)
    assert "executed" not in capsys.readouterr().out


def test_size_depth_and_assignment_limits():
    with pytest.raises(ProgramValidationError, match="bytes"):
        validate_source("#" * (SOURCE_MAX_BYTES + 1))
    source = "def update_sigma(sigma, stats, memory):\n"
    source += "".join(f"    x{i} = sigma\n" for i in range(MAX_ASSIGNMENTS + 1))
    source += "    return sigma, memory\n"
    with pytest.raises(ProgramValidationError, match="assignments"):
        validate_source(source)
    with pytest.raises(ProgramValidationError, match="depth"):
        validate_source(IDENTITY.replace("return sigma", "return " + "-" * 34 + "sigma"))
    with pytest.raises(ProgramValidationError, match="AST"):
        validate_source(IDENTITY.replace("return sigma", "return " + "+".join(["sigma"] * 200)))


def test_accepts_bounded_stateful_math():
    program = validate_source("""
def update_sigma(sigma, stats, memory):
    good = stats[4] > 0.5
    step = where(good, exp(0.1), exp(-0.1))
    width = clip(sigma * step, 0.001, 2.0)
    saved = stack((memory[0] + 1, stats[0], tanh(stats[1]), sqrt(abs(stats[2]))))
    return width, saved
""")
    assert program.ast_nodes < AST_MAX_NODES


# All numerical assertions run in one process to amortize JAX initialization.
NUMERICAL_CHECKS = r'''
import dataclasses
import json
import jax
import jax.numpy as jnp
import numpy as np
from source.algorithms.ne.ga import GASearcher, FocusGASearcher
from shinka_crl.adaptive import (
    AdaptiveGASearcher, InvalidUpdateError, ProgramValidationError,
    compile_program, training_statistics, validate_source,
)

results = {}
identity = validate_source("def update_sigma(sigma, stats, memory):\n    return sigma, memory\n")
def native(**kwargs):
    return GASearcher(7, 8, sigma_init=kwargs.pop('sigma_init', 0.5), **kwargs)
def source(body):
    return validate_source('def update_sigma(sigma, stats, memory):\n    ' + body + '\n')
def same(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

plain = native()
wrapped = AdaptiveGASearcher(plain, identity)
key = jax.random.PRNGKey(1729)
mean = jnp.arange(7, dtype=jnp.float32) * 0.1
base = plain.init(key, mean)
state = wrapped.init(key, mean)
for name in base._fields:
    same(getattr(base, name), getattr(state, name))
same(state.memory, np.zeros(4, dtype=np.float32))
assert state.memory.dtype == jnp.float32
assert state.invalid_update.dtype == jnp.bool_
assert not bool(state.invalid_update)
assert wrapped.adapts_sigma and wrapped.has_population
assert wrapped.num_offspring == plain.num_offspring
results['initial_state_and_delegation'] = True

for generation in range(5):
    ask_key = jax.random.fold_in(key, generation)
    asked, aux = jax.jit(plain.ask)(ask_key, base)
    adapted, adapted_aux = jax.jit(wrapped.ask)(ask_key, state)
    same(asked, adapted)
    assert aux is None and adapted_aux is None
    # Deliberate ties retain native ordering and archive re-scoring semantics.
    fitness = jnp.asarray([500, 100, 300, 300, 20, 500, 300, 50], dtype=jnp.float32)
    base = jax.jit(plain.tell)(base, asked, fitness)
    state = jax.jit(wrapped.tell)(state, adapted, fitness)
    for name in base._fields:
        same(getattr(base, name), getattr(state, name))
    same(plain.incumbent(base), wrapped.incumbent(state))
    same(plain.population_mean(base), wrapped.population_mean(state))
    same(plain.population(base), wrapped.population(state))
results['identity_ask_tell_five_generations'] = True

fitness = jnp.asarray([2, 4, 4, 6, 2, 4, 4, 6], dtype=jnp.float32)
stats = np.asarray(jax.jit(lambda x: training_statistics(x, 4))(fitness))
expected = [np.mean(fitness) / 500, np.std(fitness) / 500, 6 / 500,
            4 / 500, 0.25]
np.testing.assert_allclose(stats, expected, rtol=1e-6, atol=1e-8)
assert stats.dtype == np.float32
# 4 ties the current archive median and does not count as a success.
assert stats[4] == 0.25
results['current_statistics_and_strict_success'] = True

halving = AdaptiveGASearcher(native(), source('return sigma * 0.5, memory + 1.0'))
initial = halving.init(key, mean)
population, _ = halving.ask(key, initial)
next_state = jax.jit(halving.tell)(initial, population, fitness)
assert float(initial.sigma) == 0.5 and float(next_state.sigma) == 0.25
same(next_state.memory, np.ones(4, dtype=np.float32))
parents, _ = halving.ask(key, next_state._replace(sigma=jnp.float32(0)))
full, _ = halving.ask(key, next_state._replace(sigma=jnp.float32(0.5)))
half, _ = halving.ask(key, next_state)
same(full[4:], half[4:])
same(full[4:], next_state.archive)
np.testing.assert_allclose(2 * (half[:4] - parents[:4]), full[:4] - parents[:4],
                           rtol=1e-6, atol=1e-7)
assert not np.array_equal(full[:4], half[:4])
results['halving_preserves_parents_noise_and_archive'] = True

memory_program = source('return sigma, stack((memory[0] + 1, stats[3], stats[4], stats[1]))')
remember = AdaptiveGASearcher(native(), memory_program)
remember_state = remember.init(key, mean)
for generation in range(2):
    population, _ = remember.ask(key, remember_state)
    remember_state = jax.jit(remember.tell)(remember_state, population, fitness)
np.testing.assert_allclose(remember_state.memory, [2, 4/500, .25, expected[1]], rtol=1e-6)
assert remember_state.memory.dtype == jnp.float32
results['memory_and_current_statistics'] = True

for expression in ['sigma / 0', 'log(-1)']:
    broken = AdaptiveGASearcher(native(), source(f'return {expression}, memory'))
    bad_state = broken.init(key, mean)
    population, _ = broken.ask(key, bad_state)
    bad_state = jax.jit(broken.tell)(bad_state, population, fitness)
    assert bool(bad_state.invalid_update)
    try:
        broken.incumbent(bad_state)
    except InvalidUpdateError as exc:
        assert exc.completed_generations == 1
    else:
        raise AssertionError('nonfinite update did not fail on the host')
    if expression == 'sigma / 0':
        assert float(bad_state.sigma) == 2.0  # clipping must not clear invalid flag
results['nonfinite_width_rejected_before_clipping'] = True

broken = AdaptiveGASearcher(native(), source('return sigma, memory / 0'))
bad_state = broken.init(key, mean)
population, _ = broken.ask(key, bad_state)
bad_state = jax.jit(broken.tell)(bad_state, population, fitness)
assert bool(bad_state.invalid_update)
repaired = bad_state._replace(memory=jnp.zeros(4))
sticky = jax.jit(wrapped.tell)(repaired, population, fitness)
assert bool(sticky.invalid_update)
try:
    wrapped.incumbent(sticky)
except InvalidUpdateError as exc:
    assert exc.completed_generations == 2
else:
    raise AssertionError('nonfinite memory or sticky flag was lost')
results['nonfinite_memory_and_sticky_flag'] = True

for proposed, expected_width in [(3, 2), (-1, 0.001)]:
    clipped = AdaptiveGASearcher(native(), source(f'return {proposed}, memory'))
    state = clipped.init(key, mean)
    population, _ = clipped.ask(key, state)
    state = jax.jit(clipped.tell)(state, population, fitness)
    np.testing.assert_allclose(state.sigma, expected_width, rtol=1e-6)
    assert state.sigma.dtype == jnp.float32 and not bool(state.invalid_update)
results['finite_bounds_and_float32_constants'] = True

for invalid_native in [native(sigma_init=.25), native(elite_ratio=.25),
                       native(cross_over_rate=.1), FocusGASearcher(7, 8, sigma_init=.5)]:
    try:
        AdaptiveGASearcher(invalid_native, identity)
    except ValueError:
        pass
    else:
        raise AssertionError('unexpected native configuration accepted')
results['reject_wrong_method_and_settings'] = True

try:
    compile_program(dataclasses.replace(identity, source_sha256='0'*64))
except ProgramValidationError:
    pass
else:
    raise AssertionError('tampered program metadata accepted')
assert compile_program(source('return 1, memory'))(jnp.float32(.5), jnp.zeros(5),
                                                 jnp.zeros(4))[0].dtype == jnp.float32
results['compiled_metadata_and_dtype_contract'] = True
print(json.dumps(results, sort_keys=True))
'''


@pytest.fixture(scope="module")
def numerical_checks():
    if not JAX_PYTHON.is_file():
        pytest.skip("Pinned upstream JAX interpreter is not installed")
    environment = {**os.environ, **SEARCH_THREAD_ENV,
                   "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(UPSTREAM)])}
    command = [str(JAX_PYTHON), "-c", NUMERICAL_CHECKS]
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))[:2]
        command = ["taskset", "-c", ",".join(str(cpu) for cpu in cpus), *command]
    completed = subprocess.run(command, cwd=UPSTREAM, env=environment, text=True,
                               capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("check", [
    "initial_state_and_delegation", "identity_ask_tell_five_generations",
    "current_statistics_and_strict_success", "halving_preserves_parents_noise_and_archive",
    "memory_and_current_statistics", "nonfinite_width_rejected_before_clipping",
    "nonfinite_memory_and_sticky_flag", "finite_bounds_and_float32_constants",
    "reject_wrong_method_and_settings", "compiled_metadata_and_dtype_contract",
])
def test_numerical_native_adapter_contract(numerical_checks, check):
    assert numerical_checks[check] is True
