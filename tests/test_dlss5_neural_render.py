"""Unit tests for the dlss5-neural-render extension's CLI runner.

dlss5_cli.py is pure stdlib (no webui imports), so the real module is
loaded from the extension path without stubbing the WebUI. Path resolution
is relative to this file, so the tests run from the extension root or from
the WebUI repo root.
"""
import importlib.util
import json
from pathlib import Path

import pytest

EXT_ROOT = Path(__file__).resolve().parents[1]
EXT_PATH = EXT_ROOT / "dlss5_cli.py"

pytestmark = pytest.mark.skipif(
    not EXT_PATH.is_file(),
    reason="dlss5-neural-render extension is not installed",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("dlss5_cli", EXT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_mod():
    return _load_module()


def test_build_command_minimal(cli_mod):
    cmd = cli_mod.build_command("C:/VE/VE_CLI.exe", "in", "out", 1, 1.0, None)
    assert cmd == [
        "C:/VE/VE_CLI.exe",
        "render",
        "in",
        "-o",
        "out",
        "--style",
        "1",
        "--strength",
        "1",
        "--format",
        "png",
        "--json",
        "--quiet",
    ]


def test_build_command_with_preset_and_strength_format(cli_mod):
    cmd = cli_mod.build_command("cli", "in", "out", 2, 1.25, "preset.json")
    assert cmd[cmd.index("--preset") + 1] == "preset.json"
    assert cmd[cmd.index("--strength") + 1] == "1.25"
    assert "--gpu" not in cmd


def test_build_preset_shape(cli_mod):
    preset = cli_mod.build_preset(3)
    assert preset["format"] == "dlss5-visual-enhancer-settings-preset"
    assert preset["schema_version"] == 34
    assert preset["name"] == "forge-neural-render"
    assert preset["settings"]["image_enabled_stages"] == ["neural_model"]
    assert preset["settings"]["nr_passes"] == 3


def test_write_preset_roundtrip(cli_mod, tmp_path):
    path = tmp_path / "preset.json"
    cli_mod.write_preset(path, 2)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == cli_mod.build_preset(2)


def test_parse_result_clean_json(cli_mod):
    result = cli_mod.parse_result('{"ok": true, "items": []}\n')
    assert result["ok"] is True
    assert result["items"] == []


def test_parse_result_json_after_noise_lines(cli_mod):
    payload = json.dumps({"ok": False, "exit_code": 1, "error": "boom"})
    stdout = "engine progress line\nwarning: something\n" + payload + "\n"
    assert cli_mod.parse_result(stdout) == {"ok": False, "exit_code": 1, "error": "boom"}


def test_parse_result_garbage_raises(cli_mod):
    with pytest.raises(ValueError):
        cli_mod.parse_result("not json at all\nstill not json\n")


def test_map_items_maps_only_completed_matching(cli_mod):
    items = [
        {"input_path": "in/img_000.png", "output_path": "out/a.png", "state": "complete"},
        {"input_path": "in/img_001.png", "output_path": "out/b.png", "state": "failed"},
        {"input_path": "in/img_002.png", "output_path": "out/c.png", "state": "complete"},
        {"input_path": "in/other.png", "output_path": "out/d.png", "state": "complete"},
    ]
    assert cli_mod.map_items(items) == {0: "out/a.png", 2: "out/c.png"}


def test_timeout_seconds_grows_and_caps(cli_mod):
    assert cli_mod.timeout_seconds(1, 1) == 270
    assert cli_mod.timeout_seconds(2, 2) == 540
    assert cli_mod.timeout_seconds(10, 4) == 1800


def test_sizes_ok_boundaries(cli_mod):
    assert not cli_mod.sizes_ok(63, 512)
    assert cli_mod.sizes_ok(64, 512)
    assert cli_mod.sizes_ok(512, 16384)
    assert not cli_mod.sizes_ok(512, 16385)
