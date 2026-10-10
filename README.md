# DLSS 5 Neural Render

Applies NVIDIA DLSS 5 Neural Rendering to generated images by driving the
official [Merserk Visual Enhancer](https://github.com/Merserk/dlss5-visual-enhancer)
CLI (`VE_CLI.exe`) as a subprocess - the same engine behind VE's
"Enhance (Beta)" feature.

## What it provides

1. **DLSS 5 Neural Render settings tab** - configure the Visual Enhancer
   folder (where `VE_CLI.exe` lives), check the install (CLI version,
   GPU, driver, Neural Rendering support), and save the setting. Persisted
   to `extensions/dlss5-neural-render/data/dlss5_neural_render.json`.
2. **DLSS 5 Neural Render script** (always visible, txt2img + img2img
   dropdown) - enable flag plus the three official control labels:
   **Style** (0/1/2), **NR Intensity** (0-2), **NR Passes** (1-4), and an
   **Apply stage** selector.

## Requirements

- [Visual Enhancer v15.0+](https://github.com/Merserk/dlss5-visual-enhancer/releases)
  installed separately (the extension ships no engine code, no DLLs).
- An NVIDIA RTX GPU with a driver current enough for `ai_compatible` (the
  tab's **Check** button reports this).
- This WebUI's Python venv (the extension adds no dependencies).

## Setup

1. Install Visual Enhancer (ZIP, ~693 MiB) and unzip it to a folder,
   e.g. `C:\Apps\VisualEnhancer`.
2. Open the **DLSS 5 Neural Render** settings tab.
3. Enter the folder that directly contains `VE_CLI.exe`.
4. Press **Check** - a green status line confirms CLI version, GPU and
   driver; press **Save**.

## Usage

1. Open **txt2img** (or img2img), select **DLSS 5 Neural Render** in the
   *Script* dropdown.
2. Tick **Enable DLSS 5 Neural Render** - the other four controls appear.
3. Pick a **Style** (0/1/2), **NR Intensity** (0-2, default 1.0) and
   **NR Passes** (1-4, default 1).
4. Choose the **Apply stage**:

   | Stage | What happens |
   |---|---|
   | Post-hires only (default) | Every final image is NR'd after generation. |
   | Pre-hires only | The base (pre-hires) images are NR'd, then the hires fix upscales the enhanced base; a second post pass would double-process, so the final pass is skipped for images already NR'd. If hires fix is off, the pre and post stages are the same image and everything is NR'd once. |
   | Both | The base is NR'd before the hires fix AND the final image is NR'd again afterwards. |

Notes:

- The hires fix upscaler/latent-scale mode does not matter - the pre pass
  hooks the base images the hires fix is about to consume, in both modes.
- If the Visual Enhancer folder is not configured (or `VE_CLI.exe` is
  missing), generation proceeds untouched and a one-time error is reported:
  "Visual Enhancer not configured - set the folder in the 'DLSS 5 Neural
  Render' tab".
- Images outside the Neural Rendering size range (64-16384 px per side)
  are kept as-is with a per-image error; generation continues.
- Each enhanced generation gets an infotext comment, e.g.
  `DLSS 5 Neural Render: pre+post hires, Style 1, intensity 1, passes 2`.
- The job progress label switches to "DLSS 5 Neural Render (pre-hires|post-hires, N image(s))"
  while a render runs.
- NR Passes > 1 sends the official `nr_passes` preset to the CLI (the
  extension's one supported knob; other VE settings stay at their defaults).

## API usage

The script is always visible, so it takes a slot in `script_args`:

```json
{
  "prompt": "1girl, detailed",
  "steps": 30,
  "width": 1024,
  "height": 1024,
  "script_name": "dlss5 neural render",
  "script_args": [true, "Style 1", 1.0, 2, "Both"]
}
```

`script_args` order: enable (bool), style, intensity, passes, apply stage.

## Troubleshooting

- **"VE_CLI.exe not found in '...' "** - the folder must *contain*
  `VE_CLI.exe` (no subfolder).
- **"no AI-compatible GPU found"** - update the NVIDIA driver; Neural
  Rendering needs a recent RTX-capable driver.
- **"VE_CLI timed out after Ns (pid M)"** - the render exceeded its
  timeout (180 s cold start + 90 s per image-pass, capped at 30 min) and
  was killed; originals are kept. Lower intensity/passes or batch size.
- **Per-image errors** - each failure is reported with the VE log path
  (`log: <path>`); the log holds the engine-side detail.
- The temp input/output dirs (`dlss5nr_in_*`/`dlss5nr_out_*` in the
  system temp folder) are removed automatically.

## License

Unofficial community tool. Not affiliated with NVIDIA or Merserk.
This extension contains no Visual Enhancer code; the engine is downloaded
and run from the official release. The Visual Enhancer project is
distributed under its own Merserk Source License 1.0, which forbids
redistribution - install it separately from the release page.

## Limits

- Single-GPU: VE's automatic device selection is used (no `--gpu`
  targeting for multi-GPU machines).
- Generation pages only - there is no postprocessing tab and no batch
  file mode (point the official CLI at files for that).
- Only the three official control labels are exposed; the full VE preset
  (color boost, etc.) stays at engine defaults.
