"""DLSS 5 Neural Render: enhance generated images with the official
Merserk Visual Enhancer CLI (VE_CLI.exe).

Always-visible script on txt2img and img2img: every final generated
image is rendered through VE's DLSS 5 Neural Rendering pass after the
hires fix (if any). Generation never breaks: on any failure the
original images are kept and the error is reported.

The Visual Enhancer folder is configurable in the standard WebUI Settings
panel (section "DLSS 5 Neural Render"). No Visual Enhancer code is bundled;
the extension only drives the official CLI.
"""

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr
from PIL import Image

from modules import errors, images, script_callbacks, scripts, shared
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


def _print_status(message: str) -> None:
    """One console status line; print() reaches the WebUI CLI even at the default log level."""
    print(f"DLSS 5 Neural Render: {message}")


def _render_cli(cli, pils, results, errors_list, valid, style, intensity, passes, job):
    passes_i = int(passes)
    in_dir = tempfile.mkdtemp(prefix="dlss5nr_in_")
    out_dir = tempfile.mkdtemp(prefix="dlss5nr_out_")
    started = time.monotonic()
    _print_status(
        f"{job} pass starting - {len(valid)} image(s), style {dlss5_cli.style_index(style)}, "
        f"intensity {float(intensity):g}, passes {passes_i}"
    )
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
            _print_status(f"{job} pass failed - {reason} (log: {log_path})")
            for i in valid:
                errors_list[i] = f"VE CLI: {reason} (log: {log_path})"
            return results, errors_list
        _collect_results(result, results, errors_list, valid)
        n_ok = sum(1 for i in valid if not errors_list[i])
        note = f", {len(valid) - n_ok} failed" if n_ok < len(valid) else ""
        _print_status(
            f"{job} pass done in {time.monotonic() - started:.1f}s - "
            f"{n_ok}/{len(valid)} image(s) enhanced{note}"
        )
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


def _add_comment(p, state: dict) -> None:
    """Add the infotext comment at most once per generation."""
    if getattr(p, "dlss5nr_commented", False):
        return
    p.comment(
        f"DLSS 5 Neural Render: Style {dlss5_cli.style_index(state['style'])}, "
        f"intensity {float(state['intensity']):g}, passes {int(state['passes'])}"
    )
    p.dlss5nr_commented = True


def _resave_to_disk(orig, new) -> None:
    """Overwrite the output-folder file of a replaced image with the enhanced one.

    The core writes each image to disk during the batch loop (processing.py),
    before the script postprocess hook runs, so after processed.images is
    replaced in memory the folder still holds the pre-enhancement image.
    Re-saving keeps the folder - and anything else reading it - in sync with
    the gallery and the Save button, which use the in-memory images.
    """
    saved_as = getattr(orig, "already_saved_as", None)
    if not saved_as:
        return
    fullfn = Path(saved_as)
    if not fullfn.is_file():
        return
    tmp = fullfn.parent / f"{fullfn.name}.tmp"
    try:
        images.save_image_with_geninfo(
            new, orig.info.get("parameters"), str(tmp), extension=fullfn.suffix
        )
        os.replace(tmp, fullfn)
    except Exception as e:
        errors.display(e, "updating the saved image in the output folder")
        logger.exception("DLSS 5 Neural Render: could not update %s with the enhanced image", fullfn)
        tmp.unlink(missing_ok=True)
    new.already_saved_as = str(fullfn)


def _postprocess_pass(p, processed, style, intensity, passes) -> None:
    cli = dlss5_cli.cli_path(_read_ve_folder())
    if not cli.is_file():
        _report_missing_cli()
        return
    pils = list(processed.images)
    results, errs = render_batch(cli, pils, style, float(intensity), int(passes), "post-hires")
    _report_slot_errors(errs)
    state = {"style": style, "intensity": intensity, "passes": passes}
    replaced = 0
    for i in range(len(pils)):
        if not errs[i]:
            orig = processed.images[i]
            processed.images[i] = results[i]
            _resave_to_disk(orig, results[i])
            replaced += 1
    if replaced > 0:
        _add_comment(p, state)


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
            check_button = gr.Button("Check Visual Enhancer", variant="secondary")
            check_output = gr.Textbox(
                label="Check result",
                interactive=False,
                lines=3,
                elem_id=self.elem_id("check-result"),
            )
            check_button.click(fn=self._check_install, outputs=[check_output])
        return [enable, style, intensity, passes]

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

    def postprocess(self, p, processed, enable, style, intensity, passes):
        if not enable:
            return
        try:
            _postprocess_pass(p, processed, style, intensity, passes)
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
