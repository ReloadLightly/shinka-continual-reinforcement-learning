"""Restricted mutation programs and a delegated adapter for the pinned native GA.

Grammar inspection uses only the standard library. JAX and the upstream trainer
are imported only when compiling or constructing a searcher in its subprocess.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, NamedTuple


GRAMMAR_VERSION = "adaptive-width-v1"
SOURCE_MAX_BYTES = 8192
AST_MAX_NODES = 512
MAX_EXPR_DEPTH = 32
MAX_ASSIGNMENTS = 32
MEMORY_SIZE = 4
STATS_SIZE = 5
SIGMA_MIN = 0.001
SIGMA_MAX = 2.0
INITIAL_SIGMA = 0.5
FITNESS_SCALE = 500.0
OPERATIONS = {
    "exp": 1, "log": 1, "sqrt": 1, "tanh": 1, "abs": 1,
    "minimum": 2, "maximum": 2, "clip": 3, "where": 3, "stack": 1,
}


@dataclass(frozen=True)
class ProgramSpec:
    source: str
    source_sha256: str
    canonical_ast_sha256: str
    ast_nodes: int
    grammar_version: str = GRAMMAR_VERSION

    def metadata(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "canonical_ast_sha256": self.canonical_ast_sha256,
            "ast_nodes": self.ast_nodes,
            "grammar_version": self.grammar_version,
        }


class ProgramValidationError(ValueError):
    """The source, shape, or dtype is outside the declared executable grammar."""


# A shape has at most one short vector dimension; there is no allocation syntax.
_Type = tuple[tuple[int, ...], str]


def _broadcast(values: list[_Type]) -> tuple[int, ...]:
    shapes = {shape for shape, _ in values if shape}
    if len(shapes) > 1:
        raise ProgramValidationError("Only scalar or equal-length vector broadcasting is allowed")
    return next(iter(shapes), ())


class _Grammar:
    def __init__(self) -> None:
        self.variables: dict[str, _Type] = {
            "sigma": ((), "float"), "stats": ((STATS_SIZE,), "float"),
            "memory": ((MEMORY_SIZE,), "float"),
        }

    def expression(self, node: ast.AST, depth: int = 0) -> _Type:
        if depth > MAX_EXPR_DEPTH:
            raise ProgramValidationError(f"Expression depth exceeds {MAX_EXPR_DEPTH}")

        def descend(value: ast.AST) -> _Type:
            return self.expression(value, depth + 1)

        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ProgramValidationError("Only finite real numeric literals are allowed")
            try:
                finite = math.isfinite(node.value) and abs(node.value) <= 3.4028234663852886e38
            except OverflowError:
                finite = False
            if not finite:
                raise ProgramValidationError("Numeric literals must be finite float32 values")
            return (), "float"
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in self.variables:
                raise ProgramValidationError(f"Unknown value {node.id!r}")
            return self.variables[node.id]
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            if not isinstance(node.value, ast.Name) or node.value.id not in self.variables:
                raise ProgramValidationError("Indexing requires a named vector")
            shape, dtype = self.variables[node.value.id]
            if not (
                shape and isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int and 0 <= node.slice.value < shape[0]
            ):
                raise ProgramValidationError("Vector indices must be in-range nonnegative constants")
            return (), dtype
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            shape, dtype = descend(node.operand)
            if dtype != "float":
                raise ProgramValidationError("Arithmetic requires floating-point values")
            return shape, dtype
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            values = [descend(node.left), descend(node.right)]
            if any(dtype != "float" for _, dtype in values):
                raise ProgramValidationError("Arithmetic requires floating-point values")
            return _broadcast(values), "float"
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or not isinstance(
                node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
            ):
                raise ProgramValidationError("Only a single numeric comparison is allowed")
            values = [descend(node.left), descend(node.comparators[0])]
            if any(dtype != "float" for _, dtype in values):
                raise ProgramValidationError("Comparisons require floating-point values")
            return _broadcast(values), "bool"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in OPERATIONS:
                raise ProgramValidationError("Calls must name a whitelisted operation directly")
            operation = node.func.id
            if node.keywords or len(node.args) != OPERATIONS[operation]:
                raise ProgramValidationError(f"{operation} has fixed positional arity")
            if operation == "stack":
                values_node = node.args[0]
                if not isinstance(values_node, (ast.Tuple, ast.List)) or len(values_node.elts) != 4:
                    raise ProgramValidationError("stack requires exactly four scalar expressions")
                values = [descend(value) for value in values_node.elts]
                if any(value != ((), "float") for value in values):
                    raise ProgramValidationError("stack requires exactly four float scalars")
                return (4,), "float"
            values = [descend(value) for value in node.args]
            if operation == "where":
                if values[0][1] != "bool" or any(dtype != "float" for _, dtype in values[1:]):
                    raise ProgramValidationError("where requires a comparison and two float values")
            elif any(dtype != "float" for _, dtype in values):
                raise ProgramValidationError("Math operations require floating-point values")
            return _broadcast(values), "float"
        raise ProgramValidationError(f"Unsupported expression: {type(node).__name__}")

    def function(self, function: ast.FunctionDef) -> None:
        args = function.args
        if (
            function.name != "update_sigma" or function.decorator_list or function.returns
            or function.type_comment or args.posonlyargs or args.vararg or args.kwonlyargs
            or args.kw_defaults or args.kwarg or args.defaults
            or [arg.arg for arg in args.args] != ["sigma", "stats", "memory"]
            or any(arg.annotation or arg.type_comment for arg in args.args)
            or getattr(function, "type_params", [])
        ):
            raise ProgramValidationError("Expected undecorated update_sigma(sigma, stats, memory)")
        if not function.body or not isinstance(function.body[-1], ast.Return):
            raise ProgramValidationError("The function must end with a two-value return")
        assignments = function.body[:-1]
        if len(assignments) > MAX_ASSIGNMENTS:
            raise ProgramValidationError(f"At most {MAX_ASSIGNMENTS} assignments are allowed")
        for statement in assignments:
            if not (
                isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name) and not statement.type_comment
            ):
                raise ProgramValidationError("Only single local assignments may precede the return")
            name = statement.targets[0].id
            if name.startswith("_") or name in OPERATIONS or name == "update_sigma":
                raise ProgramValidationError("Reserved names cannot be assigned")
            self.variables[name] = self.expression(statement.value)
        returned = function.body[-1].value
        if not isinstance(returned, ast.Tuple) or len(returned.elts) != 2:
            raise ProgramValidationError("Return exactly (next_sigma, next_memory)")
        width, memory = [self.expression(value) for value in returned.elts]
        if width != ((), "float") or memory != ((4,), "float"):
            raise ProgramValidationError("Return a float scalar and a length-four float vector")


def validate_source(source: str) -> ProgramSpec:
    """Validate without importing or executing candidate code, including JAX."""
    if not isinstance(source, str):
        raise ProgramValidationError("Source must be text")
    try:
        encoded = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ProgramValidationError("Source must be valid UTF-8") from exc
    if len(encoded) > SOURCE_MAX_BYTES:
        raise ProgramValidationError(f"Source exceeds {SOURCE_MAX_BYTES} bytes")
    try:
        tree = ast.parse(source, type_comments=True)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ProgramValidationError(f"Invalid Python syntax: {exc}") from exc
    nodes = sum(1 for _ in ast.walk(tree))
    if nodes > AST_MAX_NODES:
        raise ProgramValidationError(f"AST exceeds {AST_MAX_NODES} nodes")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ProgramValidationError("Source must contain only one function")
    _Grammar().function(tree.body[0])
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")
    return ProgramSpec(source, hashlib.sha256(encoded).hexdigest(),
                       hashlib.sha256(canonical).hexdigest(), nodes)


def load_program(path: str | Path) -> ProgramSpec:
    # Read bytes explicitly: universal-newline conversion would misreport the raw hash.
    try:
        data = Path(path).read_bytes()
        if len(data) > SOURCE_MAX_BYTES:
            raise ProgramValidationError(f"Source exceeds {SOURCE_MAX_BYTES} bytes")
        return validate_source(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ProgramValidationError("Source must be valid UTF-8") from exc


class _FloatLiterals(ast.NodeTransformer):
    """Use float32 even for constant-only expressions; leave vector indices as ints."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Call:
        return ast.copy_location(ast.Call(func=ast.Name(id="_literal", ctx=ast.Load()),
                                          args=[node], keywords=[]), node)


def compile_program(program: ProgramSpec):
    """Build an isolated callable and check its exact JAX shapes and float32 dtypes."""
    import jax
    import jax.numpy as jnp

    if validate_source(program.source) != program:
        raise ProgramValidationError("Program source and metadata do not match")
    tree = _FloatLiterals().visit(ast.parse(program.source))
    ast.fix_missing_locations(tree)
    namespace = {"__builtins__": {}, "_literal": jnp.float32}
    namespace.update({name: getattr(jnp, name) for name in OPERATIONS})
    exec(compile(tree, "<adaptive-program>", "exec"), namespace)
    update = namespace["update_sigma"]
    arguments = (
        jax.ShapeDtypeStruct((), jnp.float32),
        jax.ShapeDtypeStruct((STATS_SIZE,), jnp.float32),
        jax.ShapeDtypeStruct((MEMORY_SIZE,), jnp.float32),
    )
    try:
        outputs = jax.eval_shape(update, *arguments)
    except Exception as exc:
        raise ProgramValidationError(f"Program failed JAX shape validation: {exc}") from exc
    if not (
        isinstance(outputs, tuple) and len(outputs) == 2
        and outputs[0].shape == () and outputs[1].shape == (MEMORY_SIZE,)
        and all(output.dtype == jnp.dtype("float32") for output in outputs)
    ):
        raise ProgramValidationError("JAX output must be float32 scalar and length-four memory")
    return update


class AdaptiveGAState(NamedTuple):
    archive: Any
    fitness: Any
    sigma: Any
    generation: Any
    memory: Any
    invalid_update: Any


class InvalidUpdateError(RuntimeError):
    def __init__(self, completed_generations: int) -> None:
        self.completed_generations = completed_generations
        super().__init__(f"Nonfinite adaptive update after {completed_generations} completed generations")


def training_statistics(fitness, num_offspring: int):
    """Five current-batch statistics; never read the native state's stored losses."""
    import jax.numpy as jnp

    if fitness.ndim != 1 or not 0 < num_offspring < fitness.shape[0]:
        raise ValueError("Expected one fitness per offspring followed by the re-scored archive")
    current = jnp.asarray(fitness, dtype=jnp.float32)
    archive = current[num_offspring:]
    return jnp.stack((
        jnp.mean(current) / FITNESS_SCALE,
        jnp.std(current, ddof=0) / FITNESS_SCALE,
        jnp.max(current) / FITNESS_SCALE,
        jnp.mean(archive) / FITNESS_SCALE,
        jnp.mean((current[:num_offspring] > jnp.median(archive)).astype(jnp.float32)),
    ))


class AdaptiveGASearcher:
    """Delegate all native GA behavior, replacing only next sigma and program state."""

    adapts_sigma = True

    def __init__(self, native, program: ProgramSpec) -> None:
        from source.algorithms.ne.ga import GASearcher

        if type(native) is not GASearcher:
            raise ValueError("Adaptive programs require the exact native plain GASearcher")
        if native.sigma_init != INITIAL_SIGMA:
            raise ValueError("Adaptive programs require initial sigma=0.5")
        if native.num_elites != int(native.population_size * 0.5):
            raise ValueError("Adaptive programs require archive fraction=0.5")
        if native.variation_params["cross_over_rate"] != 0.0:
            raise ValueError("Adaptive programs require crossover=0")
        self.native = native
        self.program = program
        self.update_sigma = compile_program(program)

    def __getattr__(self, name):
        # Native runner introspection (population shape, variation, etc.) is preserved.
        return getattr(self.native, name)

    def init(self, key, mean):
        import jax.numpy as jnp

        state = self.native.init(key, mean)
        return AdaptiveGAState(*state, memory=jnp.zeros((MEMORY_SIZE,), dtype=jnp.float32),
                               invalid_update=jnp.asarray(False))

    def ask(self, key, state):
        return self.native.ask(key, state)

    def tell(self, state, aux, fitness, descriptors=None):
        import jax.numpy as jnp

        if fitness.shape != (self.native.population_size,):
            raise ValueError("Fitness must match the unchanged native population size")
        stats = training_statistics(fitness, self.native.num_offspring)
        native_state = self.native.tell(state, aux, fitness, descriptors)
        sigma, memory = self.update_sigma(state.sigma, stats, state.memory)
        invalid = state.invalid_update | ~jnp.isfinite(sigma) | ~jnp.all(jnp.isfinite(memory))
        return native_state._replace(sigma=jnp.clip(sigma, SIGMA_MIN, SIGMA_MAX),
                                     memory=memory, invalid_update=invalid)

    def incumbent(self, state):
        # The pinned outer loop calls this on the host immediately after its JIT step.
        if bool(state.invalid_update):
            raise InvalidUpdateError(int(state.generation))
        return self.native.incumbent(state)

    def population_mean(self, state):
        return self.native.population_mean(state)

    def population(self, state):
        return self.native.population(state)
