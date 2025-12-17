# Copilot instructions (MAA_for_Millennium_Tour)

## Big picture
- This repo is a **MaaFramework** automation script bundle: pipelines/images/models live under `assets/`, and a Python **AgentServer** provides custom actions/recognitions.
- Release packaging is **generated**, not hand-edited: `tools/install.py` creates `install/` by copying MaaFramework binaries from `deps/` plus this repo’s `assets/` + `agent/`.
- This workspace may also contain `MaaQNZL-main/` as a reference project; do **not** modify it unless explicitly asked.

## Repo layout to know
- `assets/interface.json`: entry config consumed by MaaFramework GUI/CLI.
- `assets/resource/`: the real “product” (pipelines, images, models).
  - `assets/resource/pipeline/*.json`: task pipelines.
  - `assets/resource/image/**`: template images grouped by feature.
- `agent/`: custom Python logic loaded by MaaFramework.
  - `agent/main.py`: AgentServer entrypoint; **requires** a `socket_id` argument.
  - `agent/my_action.py`: registers custom actions via `@AgentServer.custom_action("...")`.
  - `agent/my_reco.py`: registers custom recognitions via `@AgentServer.custom_recognition("...")`.

## Critical workflows (what to run)
- Validate resources the same way CI does:
  - `python ./check_resource.py ./assets/resource/`
  - CI installs `maafw --pre` first; if local validation fails, ensure `maafw` is available.
- Build a distributable `install/` folder:
  - Put MaaFramework release artifacts under `deps/` (CI downloads/extracts to `deps/`).
  - Run `python ./tools/install.py <version-tag>` (defaults to `v0.0.1`).
    - This also runs `tools/configure.py` to ensure OCR model exists at `assets/resource/model/ocr`.

## Project-specific conventions
- Custom action/recognition params are JSON strings:
  - Actions read `argv.custom_action_param` and should return `True/False`.
  - Recognitions implement `analyze(...)` and typically return `CustomRecognition.AnalyzeResult(box=..., detail=...)`.
- Prefer pipeline-level overrides via `Context.override_pipeline(...)` (see `agent/my_reco.py`) rather than hard-coding per-node changes.

## Formatting/automation
- Pre-commit hooks format `yaml/json` via Prettier and optimize PNG via `oxipng` (see `.pre-commit-config.yaml`).
- When editing pipeline JSON, keep existing indentation/style; CI/resource checks are stricter than formatting.
