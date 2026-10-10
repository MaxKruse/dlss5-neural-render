"""Visual Enhancer CLI runner for the DLSS 5 Neural Render extension.

Drives the official Merserk Visual Enhancer CLI (VE_CLI.exe) as a subprocess.
Kept pure stdlib (no webui imports) so the logic can be exercised by the
unit tests without stubbing the WebUI.
"""

import json
import os
import re
import subprocess
from pathlib import Path

PRESET_FORMAT = "dlss5-visual-enhancer-settings-preset"
PRESET_SCHEMA_VERSION = 34
STYLES = [
    "Style 0 (default; balanced - good for: general scenes, 3D renders, landscapes)",
    "Style 1 (natural; gentle, skin-tone preserving - good for: portraits, characters, faces)",
    "Style 2 (cinematic; filmic, reduced shine - good for: dramatic lighting, mood, cutscene looks)",
]

_STYLE_RE = re.compile(r"^Style (\d+)")


def style_index(style: str) -> int:
    """Dropdown label ('Style 1 (natural; ...)') -> 1 for the CLI --style flag."""
    m = _STYLE_RE.match(style)
    if m is None:
        raise ValueError(f"unknown style: {style!r}")
    return int(m.group(1))


NR_MIN = 64
NR_MAX = 16384
STAGE_CHOICES = ["Pre-hires only", "Post-hires only", "Both"]

# Input files are written by the extension as img_000.png, img_001.png, ...;
# the CLI result items reference them by path.
_ITEM_NAME_RE = re.compile(r"^img_(\d{3})\.png$")


def cli_path(ve_folder: str) -> Path:
    return Path(ve_folder) / "VE_CLI.exe"


def sizes_ok(w: int, h: int) -> bool:
    return NR_MIN <= w <= NR_MAX and NR_MIN <= h <= NR_MAX


def build_command(cli, in_dir, out_dir, style: int, intensity: float, preset) -> list:
    """One render call for the whole input dir.

    No --gpu flag: VE's "auto" selection is correct for single-GPU machines
    (the common case); multi-GPU targeting is a deliberate follow-up.
    """
    cmd = [
        str(cli),
        "render",
        str(in_dir),
        "-o",
        str(out_dir),
        "--style",
        str(style),
        "--strength",
        f"{float(intensity):g}",
        "--format",
        "png",
        "--json",
        "--quiet",
    ]
    if preset is not None:
        cmd.extend(["--preset", str(preset)])
    return cmd


def build_preset(nr_passes: int) -> dict:
    # image_enabled_stages must be set: with a preset present, the CLI does
    # not inject the ("neural_model",) default and nothing would render.
    return {
        "format": PRESET_FORMAT,
        "schema_version": PRESET_SCHEMA_VERSION,
        "name": "forge-neural-render",
        "settings": {
            "image_enabled_stages": ["neural_model"],
            "nr_passes": int(nr_passes),
        },
    }


def write_preset(path, nr_passes: int) -> Path:
    path = Path(path)
    path.write_text(json.dumps(build_preset(nr_passes), indent=2) + "\n", encoding="utf-8")
    return path


def timeout_seconds(n_images: int, n_passes: int) -> int:
    # 180s covers PyInstaller cold start + engine init; 90s per image-pass unit;
    # hard cap 30min.
    return min(1800, 180 + 90 * n_images * n_passes)


def _creation_flags() -> int:
    if os.name == "nt":
        return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    return 0


def _run(cmd: list, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=_creation_flags(),
    )


def _kill_tree(e: subprocess.TimeoutExpired) -> None:
    # PID-specified tree kill - never by image name.
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(e.pid)], capture_output=True, timeout=15)
    except Exception:
        pass


def run_render(cmd: list, timeout: int) -> tuple:
    """Run the CLI; returns (returncode, stdout, stderr). Raises TimeoutError on timeout."""
    try:
        proc = _run(cmd, timeout)
    except subprocess.TimeoutExpired as e:
        _kill_tree(e)
        raise TimeoutError(f"VE_CLI timed out after {timeout}s (pid {e.pid})") from e
    return proc.returncode, proc.stdout, proc.stderr


def parse_result(stdout: str) -> dict:
    """Parse the single JSON result object the CLI writes to stdout.

    Falls back to scanning from the last line backwards for a JSON object in
    case unexpected chatter is mixed in; fails closed with ValueError.
    """
    text = (stdout or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"unparseable CLI output: {text[-400:]!r}")


def map_items(items: list) -> dict:
    """Map input indices (img_XXX.png) to output paths for completed items only."""
    out = {}
    for item in items or []:
        match = _ITEM_NAME_RE.match(Path(item.get("input_path", "")).name)
        if match and item.get("state") == "complete":
            out[int(match.group(1))] = item.get("output_path")
    return out


def final_indices(when: str, enable_hr: bool, n: int, pre_ok: set) -> list:
    """Which final-image indices the post-hires pass must still NR.

    Best effort for "Pre-hires only": an image whose pre-pass failed - or
    none ran at all (e.g. VE not configured) - is NR'd on the final image.
    """
    if not enable_hr or when in ("Post-hires only", "Both"):
        return list(range(n))
    return [i for i in range(n) if i not in pre_ok]


def stage_label(when: str, enable_hr: bool) -> str:
    if not enable_hr:
        return "no hires fix"
    return {
        "Pre-hires only": "pre-hires",
        "Post-hires only": "post-hires",
        "Both": "pre+post hires",
    }.get(when, "post-hires")


def check_install(ve_folder: str) -> dict:
    """Probe the CLI (--version, devices --json) for the install check status line."""
    result = {"ok": False, "version": "", "gpus": [], "error": ""}
    cli = cli_path(ve_folder)
    if not cli.is_file():
        result["error"] = f"VE_CLI.exe not found in '{ve_folder}'"
        return result
    try:
        version_proc = _run([str(cli), "--version"], 20)
    except (subprocess.TimeoutExpired, OSError) as e:
        result["error"] = f"VE_CLI.exe --version failed: {e}"
        return result
    lines = [ln.strip() for ln in (version_proc.stdout or "").splitlines() if ln.strip()]
    result["version"] = (lines[-1] if lines else "").removeprefix("Visual Enhancer CLI ").strip()
    if version_proc.returncode != 0:
        result["error"] = f"VE_CLI.exe --version exited {version_proc.returncode}"
        return result
    try:
        devices_proc = _run([str(cli), "devices", "--json"], 30)
    except (subprocess.TimeoutExpired, OSError) as e:
        result["error"] = f"VE_CLI.exe devices --json failed: {e}"
        return result
    try:
        data = parse_result(devices_proc.stdout)
    except ValueError as e:
        result["error"] = str(e)
        return result
    if not data.get("ok"):
        result["error"] = f"devices command failed: {data.get('error', 'unknown')}"
        return result
    for dev in data.get("devices", []):
        result["gpus"].append({
            "name": dev.get("name", ""),
            "vram_gb": round((dev.get("memory_mb") or 0) / 1024, 1),
            "driver": dev.get("driver", ""),
            "ai_compatible": bool(dev.get("ai_compatible")),
        })
    if not any(g["ai_compatible"] for g in result["gpus"]):
        result["error"] = "no AI-compatible GPU found (need an NVIDIA RTX GPU with a current driver)"
        return result
    result["ok"] = True
    return result
