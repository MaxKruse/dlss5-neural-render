"""Unit tests for the generation script's settings-panel and check-button logic.

The webui imports (modules.*, gradio) are stubbed out per the house test
pattern (tests/test_lora_showcase_reuse.py); the real script module and the
real dlss5_cli module are loaded from the extension paths, so the tested
branches and status-line formats are the shipped ones.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

EXT_ROOT = Path(__file__).resolve().parents[1]
EXT_PATH = EXT_ROOT / "dlss5_cli.py"
SCRIPT_PATH = EXT_ROOT / "scripts" / "dlss5_neural_render.py"

pytestmark = pytest.mark.skipif(
    not (EXT_PATH.is_file() and SCRIPT_PATH.is_file()),
    reason="dlss5-neural-render extension is not installed",
)


class _FakeOpts:
    def __init__(self):
        self.data = {}
        self.saved = 0
        self.options = []

    def set(self, key, value):
        self.data[key] = value

    def save(self, filename):
        self.saved += 1

    def add_option(self, key, info):
        self.options.append((key, info))


class _Fakes:
    def __init__(self):
        self.opts = _FakeOpts()
        self.state = types.SimpleNamespace(job="")
        self.callbacks = []
        self.errors = []


class _FakeScriptBase:
    def __init__(self, p=None):
        self.p = p


def _option_info(default, label="", **kw):
    return {"default": default, "label": label, **kw}


def _load_script(monkeypatch):
    """Load the real script with webui imports stubbed; return (module, fakes)."""
    fakes = _Fakes()
    modules_stub = types.ModuleType("modules")
    stubs = {
        "gradio": types.ModuleType("gradio"),
        "modules": modules_stub,
        "modules.errors": types.SimpleNamespace(report=fakes.errors.append),
        "modules.sd_samplers_common": types.SimpleNamespace(
            sample_to_image=lambda *a, **k: None,
            images_tensor_to_samples=lambda *a, **k: None,
        ),
        "modules.script_callbacks": types.SimpleNamespace(
            on_ui_settings=fakes.callbacks.append
        ),
        "modules.scripts": types.SimpleNamespace(
            Script=_FakeScriptBase, AlwaysVisible="AlwaysVisible"
        ),
        "modules.shared": types.SimpleNamespace(
            opts=fakes.opts,
            state=fakes.state,
            config_filename="configs/config.json",
            OptionInfo=_option_info,
        ),
        "modules.ui_components": types.SimpleNamespace(
            InputAccordion=lambda *a, **k: None
        ),
    }
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    for name, stub in stubs.items():
        if "." in name:
            setattr(modules_stub, name.split(".", 1)[1], stub)

    cli_spec = importlib.util.spec_from_file_location("dlss5_cli", EXT_PATH)
    cli_module = importlib.util.module_from_spec(cli_spec)
    cli_spec.loader.exec_module(cli_module)
    monkeypatch.setitem(sys.modules, "dlss5_cli", cli_module)

    spec = importlib.util.spec_from_file_location("dlss5nr_script_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, fakes


def _success_info():
    return {
        "ok": True,
        "version": "1.0.0",
        "gpus": [
            {
                "name": "NVIDIA GeForce RTX 5090",
                "vram_gb": 31.8,
                "driver": "616.92",
                "ai_compatible": True,
            }
        ],
        "error": "",
    }


def test_check_install_reports_missing_folder(monkeypatch):
    module, _ = _load_script(monkeypatch)
    script = module.DLSS5NeuralRenderScript()
    assert "FAILED: no Visual Enhancer folder set" in script._check_install()


def test_check_install_blocks_during_generation(monkeypatch):
    module, fakes = _load_script(monkeypatch)
    fakes.state.job = "a generation"
    script = module.DLSS5NeuralRenderScript()
    assert "in progress" in script._check_install()


def test_check_install_surfaces_cli_error(monkeypatch):
    module, fakes = _load_script(monkeypatch)
    fakes.opts.data[module.OPT_KEY] = "C:/nowhere"
    monkeypatch.setattr(
        module.dlss5_cli,
        "check_install",
        lambda folder: {"ok": False, "version": "", "gpus": [],
                        "error": "VE_CLI.exe not found in 'C:/nowhere'"},
    )
    script = module.DLSS5NeuralRenderScript()
    assert script._check_install() == "FAILED: VE_CLI.exe not found in 'C:/nowhere'"


def test_check_install_success_line(monkeypatch):
    module, fakes = _load_script(monkeypatch)
    fakes.opts.data[module.OPT_KEY] = "C:/VE"
    monkeypatch.setattr(module.dlss5_cli, "check_install", lambda folder: _success_info())
    script = module.DLSS5NeuralRenderScript()
    assert (
        script._check_install()
        == "Visual Enhancer CLI 1.0.0 - NVIDIA GeForce RTX 5090 "
        "(31.8 GB, driver 616.92) - Neural Rendering supported"
    )


def test_migrate_moves_legacy_folder_into_opts(monkeypatch, tmp_path):
    module, fakes = _load_script(monkeypatch)
    legacy = tmp_path / "dlss5_neural_render.json"
    legacy.write_text(json.dumps({"ve_folder": "C:/VE"}), encoding="utf-8")
    monkeypatch.setattr(module, "LEGACY_CONFIG_PATH", legacy)

    module._migrate_legacy_config()

    assert fakes.opts.data[module.OPT_KEY] == "C:/VE"
    assert fakes.opts.saved == 1
    assert not legacy.is_file()


def test_migrate_keeps_existing_opts_value(monkeypatch, tmp_path):
    module, fakes = _load_script(monkeypatch)
    legacy = tmp_path / "dlss5_neural_render.json"
    legacy.write_text(json.dumps({"ve_folder": "C:/legacy"}), encoding="utf-8")
    monkeypatch.setattr(module, "LEGACY_CONFIG_PATH", legacy)
    fakes.opts.data[module.OPT_KEY] = "C:/existing"

    module._migrate_legacy_config()

    assert fakes.opts.data[module.OPT_KEY] == "C:/existing"
    assert not legacy.is_file()


def test_migrate_noop_without_file(monkeypatch, tmp_path):
    module, fakes = _load_script(monkeypatch)
    monkeypatch.setattr(module, "LEGACY_CONFIG_PATH", tmp_path / "absent.json")

    module._migrate_legacy_config()

    assert fakes.opts.data == {}
    assert fakes.opts.saved == 0


def test_migrate_keeps_file_on_corrupt_json(monkeypatch, tmp_path):
    module, fakes = _load_script(monkeypatch)
    legacy = tmp_path / "dlss5_neural_render.json"
    legacy.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(module, "LEGACY_CONFIG_PATH", legacy)

    module._migrate_legacy_config()

    assert legacy.is_file()
    assert fakes.opts.data == {}


def test_on_ui_settings_registers_panel_option(monkeypatch, tmp_path):
    module, fakes = _load_script(monkeypatch)
    monkeypatch.setattr(module, "LEGACY_CONFIG_PATH", tmp_path / "absent.json")

    fakes.callbacks[-1]()  # the registered on_ui_settings

    assert len(fakes.opts.options) == 1
    key, info = fakes.opts.options[0]
    assert key == "dlss5_neural_render_ve_folder"
    assert info["section"] == ("dlss5_neural_render", "DLSS 5 Neural Render")
