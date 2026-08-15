# Baseline reproduction & failure taxonomy notes

## Official baseline (neuralCAD-Edit)

Vendored reference: `vendor/neuralCAD-Edit`

Harness behavior (`src/vlms/base_vlm.py` → `cadquery_script`):

1. Provide request video/transcript/events + input STEP
2. Ask VLM for `my_cad_function` JSON
3. Execute CadQuery script, capture stdout + single PNG
4. Loop ≤ 10 iterations until model sets `complete=true`

Paper findings used for our taxonomy:

| Observation | Implication for GroundedCAD |
|---|---|
| GPT best but ~53pp below human acceptance | Room for harness gains |
| More iterations/tokens → better | Keep budget but prefer tools over rewrites |
| Drawing helps humans more than FMs | Explicit temporal grounding |
| False / early completion risk | Independent critic + deterministic gates |
| Auto metrics ≠ human ratings | Still required for hackathon quant track |

## Local reproduction status

On this development machine, CadQuery/OCP native DLLs are blocked by Windows Application Control (`ImportError: DLL load failed ... Application Control policy`). GroundedCAD therefore includes a **pure-Python fallback geometry backend** so the agent loop, tests, and packaging remain runnable offline.

Production scoring on neuralCAD-Edit should use CadQuery (conda env from upstream `environment.yml`) on an unblocked machine / Linux CI.

## Synthetic baseline (this repo)

```bash
python scripts/make_synthetic_examples.py
python scripts/run_ablations.py --output outputs/ablations
```

Observed (fallback backend, mock LLM):

- valid_rate = 1.0 across 5 task types
- accept_rate = 1.0 for heuristic / full-mock under deterministic checks
- mean_iters ≈ 1.0 (tools succeed first pass on synthetic tasks)

These synthetic results validate the harness plumbing; they are **not** a claim of neuralCAD-Edit leaderboard performance.

## Failure categories we log

See `docs/failure_taxonomy.md`. Pipeline writes `failure_category` into `settings.json` / `pipeline_result.json`.
