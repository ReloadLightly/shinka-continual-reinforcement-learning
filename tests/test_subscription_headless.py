"""The subscription boundary must fail closed without invoking a live model."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "subscription_headless", REPO / "scripts/subscription_headless.py"
)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


def headless_args():
    return ["--sandbox", "read-only", "--ask-for-approval", "never", "--search", "exec",
            "--model", "gpt-example", "-c", 'model_reasoning_effort="medium"',
            "--json", "--skip-git-repo-check", "-"]


def test_environment_scrub_preserves_login_location_and_no_secrets():
    env = ADAPTER.subscription_env({
        "OPENAI_API_KEY": "secret", "CODEX_API_KEY": "secret", "GOOGLE_API_KEY": "secret",
        "CODEX_ACCESS_TOKEN": "secret", "OPENAI_BASE_URL": "https://wrong.test",
        "CODEX_HOME": "/private/codex", "PATH": "/bin", "HOME": "/private",
    })
    assert env == {"CODEX_HOME": "/private/codex", "PATH": "/bin", "HOME": "/private"}


def test_model_resolution_reads_config_not_credentials(tmp_path):
    (tmp_path / "config.toml").write_text('model = "gpt-example"\n')
    (tmp_path / "auth.json").mkdir()  # Reading this as credentials would fail.
    assert ADAPTER.configured_model({"CODEX_HOME": str(tmp_path)}) == "gpt-example"
    assert ADAPTER.configured_model({"SHINKA_CODEX_MODEL": "gpt-override"}) == "gpt-override"


@pytest.mark.parametrize("value", ["provider/model", "gpt-foo --oss", "", "not-a-gpt-model"])
def test_rejects_invalid_model(value, tmp_path):
    with pytest.raises(ValueError):
        ADAPTER.configured_model({"SHINKA_CODEX_MODEL": value, "CODEX_HOME": str(tmp_path)})


def test_canonical_invocation_enforces_auth_and_sandbox():
    command = ADAPTER.guarded_codex_args(headless_args())
    assert 'forced_login_method="chatgpt"' in command
    assert 'model_provider="openai"' in command
    assert 'web_search="disabled"' in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--search" not in command
    assert 'model_reasoning_effort="medium"' in command


@pytest.mark.parametrize("escape", [
    ["--profile", "paid"], ["--oss"], ["--dangerously-bypass-approvals-and-sandbox"],
    ["-c", 'model_provider="paid"'], ["-c", 'forced_login_method="api"'],
])
def test_rejects_route_or_sandbox_override(escape):
    with pytest.raises((SystemExit, ValueError)):
        ADAPTER.guarded_codex_args(headless_args() + escape)


def test_refuses_api_login(monkeypatch):
    monkeypatch.setattr(ADAPTER.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args, 0, "", "Logged in using API key"))
    with pytest.raises(ValueError, match="logged in using ChatGPT"):
        ADAPTER.require_chatgpt_login("/bin/codex", {})


def test_check_performs_no_inference(monkeypatch):
    calls = []
    monkeypatch.setenv("SHINKA_CODEX_MODEL", "gpt-example")
    monkeypatch.setattr(ADAPTER, "require_tool", lambda name, env: f"/bin/{name}")
    monkeypatch.setattr(ADAPTER, "headless_command", lambda env: ["/bin/node", "/pin/cli.js"])

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "Logged in using ChatGPT", "")

    monkeypatch.setattr(ADAPTER.subprocess, "run", fake_run)
    assert ADAPTER.main(["--check"]) == 0
    assert len(calls) == 3
    assert calls[0][-2:] == ["login", "status"]
    assert calls[1][-2:] == ["login", "status"]
    assert 'forced_login_method="chatgpt"' not in calls[0]
    assert 'forced_login_method="chatgpt"' in calls[1]
    assert calls[2][-1] == "--help"
    assert all("exec" not in call for call in calls)
    assert calls[2][:2] == ["/bin/node", "/pin/cli.js"]


@pytest.mark.parametrize("version,valid", [("0.6.1", True), ("0.6.2", False)])
def test_headless_package_version_is_enforced(tmp_path, monkeypatch, version, valid):
    package = tmp_path / "headless"
    (package / "dist").mkdir(parents=True)
    cli = package / "dist/cli.js"
    cli.write_text("// fake executable, never run\n")
    (package / "package.json").write_text(json.dumps({
        "name": "@roberttlange/headless", "version": version,
    }))
    monkeypatch.setattr(ADAPTER, "require_tool", lambda name, env: f"/bin/{name}")
    if valid:
        assert ADAPTER.headless_command({"SHINKA_HEADLESS_CLI": str(cli)}) == [
            "/bin/node", str(cli),
        ]
    else:
        with pytest.raises(ValueError, match="Install"):
            ADAPTER.headless_command({"SHINKA_HEADLESS_CLI": str(cli)})


def test_shim_uses_captured_absolute_binary(monkeypatch):
    monkeypatch.setenv("SHINKA_SUBSCRIPTION_REAL_CODEX", "/real/codex")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    captured = []
    monkeypatch.setattr(ADAPTER.os, "execve", lambda *args: captured.append(args))
    ADAPTER.shim_main(headless_args())
    executable, args, env = captured[0]
    assert executable == args[0] == "/real/codex"
    assert "OPENAI_API_KEY" not in env


def test_subscription_config_has_no_auxiliary_model_calls():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((REPO / "tasks/cartpole_ga/shinka-subscription.yaml").read_text())
    evo = config["evo"]
    assert evo["llm_models"] == ["headless/codex?effort=medium"]
    assert all(evo[key] is None for key in (
        "embedding_model", "meta_llm_models", "novelty_llm_models", "prompt_llm_models",
    ))
    assert not evo["evolve_prompts"] and not evo["enable_wandb_logging"]
    assert all(evo[key] == 1 for key in (
        "max_patch_attempts", "max_patch_resamples", "max_novelty_attempts",
    ))
