# IDETC 2026 — Autodesk neuralCAD-Edit (GroundedCAD)

This folder is the **Night Owls** team submission for the Autodesk track.

## Submission contents

- Final implementation: `groundedcad/`, with benchmark integration in `src/`
- 48 output pairs: `cad_outputs/` (one `.step` and one `.stl` per request)
- Per-request scores: `scores/groundedcad_claude_run48_scores.json`
- Aggregate scores: Chamfer similarity **0.9416**, Volume F1 **0.5518**, Diff F1 **0.1415**
- Final presentation: `presentation/Night_Owls_final_submission_presentation.pdf`
- Required five-example qualitative images: `presentation/qualitative_examples/`

Official starter: [grndnl/IDETC26-Hackathon-Autodesk-neuralCAD-Edit](https://github.com/grndnl/IDETC26-Hackathon-Autodesk-neuralCAD-Edit)

**Scored task:** 48 **text-conditioned** CAD edits. Automatic metrics (higher is better): Chamfer similarity, Volume F1, Diff F1.

## What we built

**GroundedCAD** is an agentic harness around a foundation model:

1. Inspect input STEP (geometry census)
2. Ground the text instruction to local targets
3. Plan a typed edit (constrained CadQuery tools first, raw script fallback)
4. Execute in a sandbox
5. Deterministic verify + independent critic (no self-declared completion)

It plugs into the official `run_harness.py` as model id `groundedcad`.

## Setup

```bash
# recommended
uv sync
# or
pip install -e .
```

**Dataset (required for scoring):** download the **hackathon** zip from the official README (not older neuralCAD-Edit dumps) and extract to `data/edit_192_external/`.

Then keep text-only requests:

```bash
uv run python src/scripts/filter_dataset_text_only.py --config src/config/edit_192_external.json
uv run python src/scripts_benchmark_inference/filter_parquet_text_only.py --input data/edit_192_external/parquets/val_edit_all.parquet
```

## Run our harness (48 text edits)

PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
$env:PYTHONPATH=(Get-Location).Path
uv run python src/scripts_benchmark_inference/run_harness.py `
  --config src/config/edit_192_external.json `
  --input data/edit_192_external/parquets/val_edit_text.parquet `
  --output_dir data/model_outputs `
  --harness src/harnesses/cadquery_script.py `
  --userId groundedcad `
  --required-extensions step
```

Offline dry-run (no API): `$env:GROUNDEDCAD_PROVIDER="mock"` and set `llm_provider` to `mock` in `src/config/edit_192_external.json` under `benchmark_models.groundedcad`.

Export STL + views if needed:

```powershell
uv run python src/scripts_preprocess/cadquery_convert.py data/model_outputs
```

Score:

```powershell
uv run python src/scripts/run_all_benchmarks.py --config src/config/edit_192_external.json
```

Or: `src/scripts/run_groundedcad.ps1`

## API keys

```powershell
$env:OPENAI_API_KEY="..."
$env:ANTHROPIC_API_KEY="..."
$env:GOOGLE_API_KEY="..."
```

Config: `src/config/edit_192_external.json` → `benchmark_models.groundedcad` (`llm_provider`, `llm_model`).

## Layout

| Path | Role |
|------|------|
| `src/` | Official benchmark, eval, CadQuery harness |
| `groundedcad/` | Our agent (inspect, plan, tools, verify) |
| `src/vlms/groundedcad.py` | Adapter for `run_harness.py` |
| `data/edit_192_external/` | Dataset (download; gitignored) |
| `data/model_outputs/` | Our STEP/STL outputs |
| `docs/` | Method notes + presentation |
| `examples/synthetic/` | Tiny offline fixtures |

## Presentation

`presentation/Night_Owls_final_submission_presentation.pdf`

## Contacts (track)

Daniele Grandi, Toby Perrett, William McCarthy — Autodesk
