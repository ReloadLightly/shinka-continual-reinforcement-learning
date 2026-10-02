"""Subscription-only bridge from Shinka to pinned Headless and Codex.

`--check` checks cached tooling and ChatGPT login without generating a proposal.
Set SHINKA_HEADLESS_COMMAND to `python /absolute/path/to/this_script.py`.
Headless 0.6.1 has no subscription-only billing flag, so a temporary Codex
executable adds enforced ChatGPT authentication and the built-in OpenAI provider.
This script never opens credential files; Codex manages its own login state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

HEADLESS_PACKAGE = "@roberttlange/headless@0.6.1"
AUTH_CONFIG = [
    "--no-daemon",
    "-c", 'model_provider="openai"',
    "-c", 'forced_login_method="chatgpt"',
    "-c", 'openai_base_url="https://api.openai.com/v1"',
    "-c", 'chatgpt_base_url="https://chatgpt.com/backend-api"',
]
STRIP_ENV = {
    "OPENAI_BASE_URL", "OPENAI_API_BASE", "CODEX_ACCESS_TOKEN", "CODEX_MODEL",
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN",
}


def subscription_env(source: dict[str, str]) -> dict[str, str]:
    """Scrub credentials at the subprocess boundary, after Shinka's dotenv load."""
    return {
        key: value for key, value in source.items()
        if not key.endswith("API_KEY") and key not in STRIP_ENV
    }


def configured_model(env: dict[str, str]) -> str:
    """Read only the model setting; never inspect auth.json or credential stores."""
    model = env.get("SHINKA_CODEX_MODEL")
    if not model:
        codex_dir = Path(env.get("CODEX_HOME", str(Path.home() / ".codex")))
        try:
            config = tomllib.loads((codex_dir / "config.toml").read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(
                "Cannot resolve Codex model; set SHINKA_CODEX_MODEL to your ChatGPT model"
            ) from exc
        model = config.get("model")
    if not isinstance(model, str) or not re.fullmatch(r"gpt-[A-Za-z0-9.-]+", model):
        raise ValueError("Set SHINKA_CODEX_MODEL to a GPT model available in your ChatGPT plan")
    return model


def require_tool(name: str, env: dict[str, str]) -> str:
    executable = shutil.which(name, path=env.get("PATH"))
    if not executable:
        raise ValueError(f"Required executable is missing: {name}")
    return str(Path(executable).absolute())


def require_chatgpt_login(codex: str, env: dict[str, str]) -> None:
    # Check the existing login before applying a forced auth method: Codex can
    # invalidate a mismatched stored login while enforcing that restriction.
    result = subprocess.run(
        [codex, "--no-daemon", "login", "status"], env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    status = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or "Logged in using ChatGPT" not in status:
        # Avoid echoing unexpected CLI diagnostics that could contain credentials.
        raise ValueError("Codex must be logged in using ChatGPT; run `codex login` first")
    enforced = subprocess.run(
        [codex, *AUTH_CONFIG, "login", "status"], env=env,
        capture_output=True, text=True, timeout=20, check=False,
    )
    if (enforced.returncode != 0
            or "Logged in using ChatGPT" not in enforced.stdout + enforced.stderr):
        raise ValueError("Codex could not enforce the subscription-only configuration")


def headless_command(env: dict[str, str]) -> list[str]:
    """Use an installed exact release without npm re-resolving its dependencies."""
    explicit = env.get("SHINKA_HEADLESS_CLI")
    cache = Path(env.get("npm_config_cache", str(Path.home() / ".npm")))
    candidates = ([Path(explicit)] if explicit else
                  sorted(cache.glob("_npx/*/node_modules/@roberttlange/headless/dist/cli.js")))
    for candidate in candidates:
        try:
            package = json.loads((candidate.parent.parent / "package.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (candidate.is_file() and candidate.name == "cli.js"
                and package.get("name") == "@roberttlange/headless"
                and package.get("version") == "0.6.1"):
            return [require_tool("node", env), str(candidate.resolve())]
    raise ValueError(
        f"Install {HEADLESS_PACKAGE} first, or set SHINKA_HEADLESS_CLI to its dist/cli.js; "
        "this adapter never downloads packages automatically"
    )


def guarded_codex_args(args: list[str]) -> list[str]:
    """Accept only the known Headless 0.6.1 invocation, rejecting routing escapes."""
    if args in (["--version"], ["-V"]):
        return args
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--sandbox", choices=["read-only"], required=True)
    parser.add_argument("--ask-for-approval", choices=["never"], required=True)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("command", choices=["exec"])
    parser.add_argument("--model", required=True)
    parser.add_argument("-c", "--config", action="append", default=[])
    parser.add_argument("--json", action="store_true", required=True)
    parser.add_argument("--skip-git-repo-check", action="store_true", required=True)
    parser.add_argument("prompt", choices=["-"])
    parsed = parser.parse_args(args)
    if not re.fullmatch(r"gpt-[A-Za-z0-9.-]+", parsed.model):
        raise ValueError("Unsupported model name")
    for override in parsed.config:
        if not re.fullmatch(r'model_reasoning_effort="(low|medium|high|xhigh)"', override):
            raise ValueError("Only a reasoning-effort override is accepted")
    return [
        *AUTH_CONFIG,
        "-c", 'web_search="disabled"',
        "--sandbox", "read-only", "--ask-for-approval", "never", "exec",
        "--model", parsed.model,
        *[item for override in parsed.config for item in ("-c", override)],
        "--json", "--skip-git-repo-check", "-",
    ]


def shim_main(args: list[str]) -> int:
    real_codex = os.environ.get("SHINKA_SUBSCRIPTION_REAL_CODEX")
    if not real_codex or not Path(real_codex).is_absolute():
        raise ValueError("Missing absolute Codex executable in guarded invocation")
    command = [real_codex, *guarded_codex_args(args)]
    os.execve(real_codex, command, subscription_env(dict(os.environ)))
    return 1  # pragma: no cover


def write_shim(directory: Path) -> Path:
    shim = directory / "codex"
    # Python repr safely quotes both paths, without shell interpolation.
    shim.write_text(
        f"#!{sys.executable}\nimport runpy, sys\n"
        f"sys.argv = [{str(Path(__file__).resolve())!r}, '--codex-shim', *sys.argv[1:]]\n"
        f"runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__main__')\n"
    )
    shim.chmod(0o700)
    return shim


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--codex-shim":
        return shim_main(argv[1:])
    env = subscription_env(dict(os.environ))
    codex = require_tool("codex", env)
    headless = headless_command(env)
    model = configured_model(env)
    require_chatgpt_login(codex, env)
    if argv == ["--check"]:
        subprocess.run(
            [*headless, "--help"], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, check=True, timeout=30,
        )
        print(json.dumps({"auth": "chatgpt", "model": model,
                          "headless": HEADLESS_PACKAGE, "inference_calls": 0}))
        return 0

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("agent", choices=["codex"])
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--allow", choices=["read-only"], required=True)
    parser.add_argument("--usage", action="store_true", required=True)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"])
    args = parser.parse_args(argv)
    if not args.prompt_file.is_file() or not args.work_dir.is_dir():
        raise ValueError("Prompt file and work directory must exist")

    with tempfile.TemporaryDirectory(prefix="shinka-subscription-") as temporary:
        write_shim(Path(temporary))
        env["SHINKA_SUBSCRIPTION_REAL_CODEX"] = codex
        env["PATH"] = temporary + os.pathsep + env.get("PATH", "")
        command = [
            *headless, "codex", "--prompt-file", str(args.prompt_file.resolve()),
            "--work-dir", str(args.work_dir.resolve()), "--allow", "read-only", "--usage",
            "--model", model,
        ]
        if args.reasoning_effort:
            command += ["--reasoning-effort", args.reasoning_effort]
        # Native model/usage remain on stdout in Shinka's expected format.
        # Do not retry quota/auth/sandbox failures or weaken the read-only sandbox.
        return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Subscription preflight failed: {error}", file=sys.stderr)
        raise SystemExit(1)
