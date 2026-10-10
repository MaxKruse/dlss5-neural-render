"""DLSS 5 Neural Render: enhance generated images with the official
Merserk Visual Enhancer CLI (VE_CLI.exe).

Always-visible script on txt2img and img2img: every generated image -
or the pre-hires base images, depending on the selected apply stage -
is rendered through VE's DLSS 5 Neural Rendering pass. Generation never
breaks: on any failure the original images are kept and the error is
reported.

The Visual Enhancer folder is configurable in the standard WebUI Settings
panel (section "DLSS 5 Neural Render"). No Visual Enhancer code is bundled;
the extension only drives the official CLI.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import errors, sd_samplers_common, script_callbacks, scripts, shared
from modules.ui_components import InputAccordion

import dlss5_cli

logger = logging.getLogger(__name__)

OPT_KEY = "dlss5_neural_render_ve_folder"
LEGACY_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "dlss5_neural_render.json"

VE_MISSING_REPORTED = False


def _read_ve_folder() -> str:
    """Read the Visual Enhancer folder from the WebUI settings."""
    return str(shared.opts.data.get(OPT_KEY, "")).strip()


def _migrate_legacy_config() -> None:
    """One-shot move of the folder value out of the old settings-tab config file."""
    if not LEGACY_CONFIG_PATH.is_file():
        return
    try:
        legacy = json.loads(LEGACY_CONFIG_PATH.read_text(encoding="utf-8"))
        folder = str(legacy.get("ve_folder", "")).strip()
        if folder and not str(shared.opts.data.get(OPT_KEY, "")).strip():
            shared.opts.set(OPT_KEY, folder)
            shared.opts.save(shared.config_filename)
        LEGACY_CONFIG_PATH.unlink()
    except Exception:
        logger.warning(
            "Could not migrate the legacy dlss5 config file; "
            "set the folder in Settings > DLSS 5 Neural Render",
            exc_info=True,
        )


def tensors_to_pils(t: torch.Tensor) -> list:
    """(N,3,H,W) float 0..1 image tensor -> list of RGB PIL images."""
    pils = []
    for i in range(t.shape[0]):
        arr = (t[i].detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        pils.append(Image.fromarray(np.moveaxis(arr, 0, 2)))
    return pils


def pils_to_tensors(pils: list) -> torch.Tensor:
    """RGB PIL images -> (N,3,H,W) cpu float32 tensor in 0..1.

    Same layout modules/processing.py builds for the image-upscaler hires
    branch (processing.py:1510-1524).
    """
    arr = np.stack([np.asarray(p, dtype=np.float32) / 255.0 for p in pils])
    return torch.from_numpy(np.moveaxis(arr, 3, 1))


def render_batch(cli: Path, pils: list, style: str, intensity: float, passes: int, job: str) -> tuple:
    """Run one VE_CLI render call for the whole list.

    Returns (results, errors), both aligned with pils; errors[i] is ''
    when slot i is ok. Originals are kept for out-of-range/failed slots.
    Never raises for expected CLI failures.
    """
    results = list(pils)
    errors_list = [""] * len(pils)
    valid = []
    for i in range(len(pils)):
        w, h = pils[i].width, pils[i].height
        if dlss5_cli.sizes_ok(w, h):
            valid.append(i)
        else:
            errors_list[i] = (
                f"image {i} is {w}x{h}, outside the {dlss5_cli.NR_MIN}..{dlss5_cli.NR_MAX}px "
                f"Neural Rendering range - kept as-is"
            )
    if not valid:
        return results, errors_list
    return _render_cli(cli, pils, results, errors_list, valid, style, intensity, passes, job)


def _render_cli(cli, pils, results, errors_list, valid, style, intensity, passes, job):
    passes_i = int(passes)
    in_dir = tempfile.mkdtemp(prefix="dlss5nr_in_")
    out_dir = tempfile.mkdtemp(prefix="dlss5nr_out_")
    try:
        for i in valid:
            pils[i].convert("RGB").save(Path(in_dir) / f"img_{i:03d}.png")
        preset = None
        if passes_i != 1:
            preset = dlss5_cli.write_preset(Path(in_dir) / "preset.json", passes_i)
        cmd = dlss5_cli.build_command(
            cli, in_dir, out_dir, dlss5_cli.style_index(style), float(intensity), preset
        )
        shared.state.job = f"DLSS 5 Neural Render ({job}, {len(valid)} image(s))"
        try:
            rc, stdout, _ = dlss5_cli.run_render(cmd, dlss5_cli.timeout_seconds(len(valid), passes_i))
            result = dlss5_cli.parse_result(stdout)
        except Exception:
            errors.report(f"DLSS 5 Neural Render ({job}) failed; keeping original images", exc_info=True)
            return results, errors_list
        if not result.get("ok"):
            reason = result.get("error") or f"exit code {rc}"
            log_path = result.get("log_path", "n/a")
            for i in valid:
                errors_list[i] = f"VE CLI: {reason} (log: {log_path})"
            return results, errors_list
        _collect_results(result, results, errors_list, valid)
        return results, errors_list
    finally:
        shutil.rmtree(in_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def _collect_results(result: dict, results: list, errors_list: list, valid: list):
    mapping = dlss5_cli.map_items(result.get("items", []))
    log_path = result.get("log_path", "n/a")
    for i in valid:
        out_path = mapping.get(i)
        if out_path is None:
            errors_list[i] = f"VE CLI failed image {i}: {_item_error(result.get('items', []), i)} (log: {log_path})"
            continue
        # Copy pixels into memory: the temp dirs are removed right after.
        with Image.open(out_path) as im:
            results[i] = im.copy()


def _item_error(items: list, i: int) -> str:
    name = f"img_{i:03d}.png"
    for item in items or []:
        if Path(item.get("input_path", "")).name == name:
            return item.get("error") or "unknown error"
    return "no result item"


def _report_slot_errors(errs: list) -> None:
    for err in errs:
        if err:
            errors.report(f"DLSS 5 Neural Render: {err}")


def _report_missing_cli() -> None:
    global VE_MISSING_REPORTED
    if VE_MISSING_REPORTED:
        return
    VE_MISSING_REPORTED = True
    errors.report("Visual Enhancer not configured - set the folder in Settings > DLSS 5 Neural Render")


def _record_pre_pass(p, errs: list) -> int:
    """Add the successfully NR'd global indices to p.dlss5nr_pre_ok; returns how many."""
    n_ok = 0
    for i, err in enumerate(errs):
        if not err:
            p.dlss5nr_pre_ok.add(p.dlss5nr_pre_seq + i)
            n_ok += 1
    p.dlss5nr_pre_seq += len(errs)
    return n_ok


def _add_comment(p, state: dict) -> None:
    """Add the infotext comment at most once per generation."""
    if getattr(p, "dlss5nr_commented", False):
        return
    p.comment(
        f"DLSS 5 Neural Render: {dlss5_cli.stage_label(state['when'], state['enable_hr'])}, "
        f"Style {dlss5_cli.style_index(state['style'])}, intensity {float(state['intensity']):g}, passes {int(state['passes'])}"
    )
    p.dlss5nr_commented = True


def _run_pre_pass(p, sample, decoded_sample, state: dict):
    """NR the pre-hires base images; returns (sample, decoded_sample), possibly replaced."""
    if decoded_sample is not None and decoded_sample.ndim == 4:
        return _pre_image_mode(p, sample, decoded_sample, state)
    if sample is not None and sample.ndim == 4:
        return _pre_latent_mode(p, sample, state)
    return sample, decoded_sample


def _pre_image_mode(p, sample, decoded_sample, state):
    """Image-upscaler hires mode: the base images tensor is already decoded (0..1)."""
    pils = tensors_to_pils(decoded_sample)
    results, errs = render_batch(
        state["cli"], pils, state["style"], state["intensity"], state["passes"], "pre-hires"
    )
    _report_slot_errors(errs)
    if _record_pre_pass(p, errs) == 0:
        return sample, decoded_sample
    _add_comment(p, state)
    return sample, pils_to_tensors(results)


def _pre_latent_mode(p, sample, state):
    """Latent-upscaler hires mode (default): the base is still latent - decode per index."""
    pils = [
        sd_samplers_common.sample_to_image(sample, i, approximation=0)
        for i in range(sample.shape[0])
    ]
    results, errs = render_batch(
        state["cli"], pils, state["style"], state["intensity"], state["passes"], "pre-hires"
    )
    _report_slot_errors(errs)
    if _record_pre_pass(p, errs) == 0:
        return sample, None
    _add_comment(p, state)
    # Re-encode via the module's own image->latent helper (processing.py:1529).
    new_sample = sd_samplers_common.images_tensor_to_samples(pils_to_tensors(results), model=p.sd_model)
    return new_sample, None


def _postprocess_pass(p, processed, style, intensity, passes, when) -> None:
    enable_hr = bool(getattr(p, "enable_hr", False))
    idxs = dlss5_cli.final_indices(
        when, enable_hr, len(processed.images), getattr(p, "dlss5nr_pre_ok", set())
    )
    if not idxs:
        return
    cli = dlss5_cli.cli_path(_read_ve_folder())
    if not cli.is_file():
        _report_missing_cli()
        return
    pils = [processed.images[i] for i in idxs]
    results, errs = render_batch(cli, pils, style, float(intensity), int(passes), "post-hires")
    _report_slot_errors(errs)
    state = {"style": style, "intensity": intensity, "passes": passes, "when": when, "enable_hr": enable_hr}
    replaced = 0
    for k, i in enumerate(idxs):
        if not errs[k]:
            processed.images[i] = results[k]
            replaced += 1
    if replaced > 0:
        _add_comment(p, state)


def _install_pre_wrapper(p, cli: Path, state: dict) -> None:
    """Wrap p.sample_hr_pass so the base images are NR'd before the hires fix.

    p may be reused across passes (e.g. lora_showcase): the original method
    is captured once, and the wrapper re-reads its state from p on each call.
    """
    orig = getattr(p, "dlss5nr_orig_sample_hr_pass", None)
    if orig is None:
        orig = p.sample_hr_pass
        p.dlss5nr_orig_sample_hr_pass = orig
    p.dlss5nr_pre_ok = set()
    p.dlss5nr_pre_seq = 0
    p.dlss5nr_pre_state = {**state, "cli": cli}

    def wrapped(sample, decoded_sample, *args, **kwargs):
        try:
            sample, decoded_sample = _run_pre_pass(p, sample, decoded_sample, p.dlss5nr_pre_state)
        except Exception:
            errors.report("DLSS 5 pre-hires pass failed; continuing with the original base image", exc_info=True)
        return orig(sample, decoded_sample, *args, **kwargs)

    p.sample_hr_pass = wrapped


class DLSS5NeuralRenderScript(scripts.Script):
    def title(self):
        return "DLSS 5 Neural Render"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(
            value=False,
            label="DLSS 5 Neural Render",
            elem_id=self.elem_id("main-accordion"),
        ) as enable:
            style = gr.Dropdown(
                choices=dlss5_cli.STYLES,
                value=dlss5_cli.STYLES[0],
                label="Style",
                info="Visual Enhancer neural-rendering style",
                elem_id=self.elem_id("style"),
            )
            intensity = gr.Slider(
                0.0,
                2.0,
                value=1.0,
                step=0.05,
                label="NR Intensity",
                info="Neural-rendering strength, 0.0 to 2.0 (1.0 is the Visual Enhancer default)",
                elem_id=self.elem_id("intensity"),
            )
            passes = gr.Slider(
                1,
                4,
                value=1,
                step=1,
                label="NR Passes",
                info="Neural-rendering passes per image (a Visual Enhancer preset is used when greater than 1)",
                elem_id=self.elem_id("passes"),
            )
            stage = gr.Dropdown(
                choices=dlss5_cli.STAGE_CHOICES,
                value="Post-hires only",
                label="Apply stage",
                info="When to run the enhancement relative to the hires fix",
                elem_id=self.elem_id("apply-stage"),
            )
            check_button = gr.Button("Check Visual Enhancer", variant="secondary")
            check_output = gr.Textbox(
                label="Check result",
                interactive=False,
                lines=3,
                elem_id=self.elem_id("check-result"),
            )
            check_button.click(fn=self._check_install, outputs=[check_output])
        return [enable, style, intensity, passes, stage]

    def _check_install(self) -> str:
        """Check button handler: validate the configured Visual Enhancer without generating."""
        if getattr(shared.state, "job", ""):
            return "A generation in progress - wait for it to finish before checking."
        folder = _read_ve_folder()
        if not folder:
            return "FAILED: no Visual Enhancer folder set - set it in Settings > DLSS 5 Neural Render."
        try:
            info = dlss5_cli.check_install(folder)
        except Exception as exc:
            return f"FAILED: {exc}"
        if not info["ok"]:
            return f"FAILED: {info['error']}"
        gpu = next(g for g in info["gpus"] if g["ai_compatible"])
        return (
            f"Visual Enhancer CLI {info['version']} - {gpu['name']} "
            f"({gpu['vram_gb']} GB, driver {gpu['driver']}) - Neural Rendering supported"
        )

    def before_process(self, p, enable, style, intensity, passes, when):
        # Pre-hires wrapping only: hires fix off means pre and post are the
        # same single image (the final pass covers it), and img2img has no
        # enable_hr attribute (getattr default False covers it).
        if not (enable and when in ("Pre-hires only", "Both") and bool(getattr(p, "enable_hr", False))):
            return
        cli = dlss5_cli.cli_path(_read_ve_folder())
        if not cli.is_file():
            # The final pass reports the missing CLI - single report path.
            return
        _install_pre_wrapper(p, cli, {
            "style": style,
            "intensity": intensity,
            "passes": passes,
            "when": when,
            "enable_hr": True,
        })

    def postprocess(self, p, processed, enable, style, intensity, passes, when):
        if not enable:
            return
        try:
            _postprocess_pass(p, processed, style, intensity, passes, when)
        except Exception:
            errors.report("DLSS 5 Neural Render post-hires pass failed; keeping original images", exc_info=True)


def on_ui_settings():
    """Register the extension settings in the standard WebUI Settings panel."""
    _migrate_legacy_config()
    shared.opts.add_option(
        OPT_KEY,
        shared.OptionInfo(
            "",
            "Visual Enhancer folder: path to the Visual Enhancer installation folder (must contain VE_CLI.exe)",
            section=("dlss5_neural_render", "DLSS 5 Neural Render"),
        ),
    )


script_callbacks.on_ui_settings(on_ui_settings)
