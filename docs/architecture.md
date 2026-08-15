# GroundedCAD Architecture

The pipeline is a **minimal local edit** loop. It does not rebuild the part from scratch.

```text
                    ┌─────────────────┐
                    │ User instruction │
                    └────────┬────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Instruction Parser  │
                  │                     │
                  │ target              │
                  │ operation           │
                  │ dimensions          │
                  │ location            │
                  │ preserve            │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Geometry Inspector  │
                  │                     │
                  │ faces               │
                  │ edges               │
                  │ holes               │
                  │ dimensions          │
                  │ bounding box        │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Edit classifier     │
                  │ typed slots:        │
                  │ type, target, dims, │
                  │ direction           │
                  └──────────┬──────────┘
                             ↓
         hole / move / add / blend / …
                             ↓
                  ┌─────────────────────┐
                  │ Per-type strategy   │
                  │ (no LLM CAD)        │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ CadQuery Generator  │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ CadQuery Execution  │
                  │ (neuralCAD harness; │
                  │  Autodesk comments  │
                  │  call this FreeCAD) │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Geometry Validator  │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Render 7 views      │
                  │ iso/front/back/     │
                  │ left/right/top/bot  │
                  └──────────┬──────────┘
                             ↓
                  ┌─────────────────────┐
                  │ Edit Critic         │
                  │                     │
                  │ Did ONLY requested  │
                  │ geometry change?    │
                  └──────────┬──────────┘
                             │
                    NO ──────┴────── YES
                    ↓                 ↓
              revise edit          output
```

## Stage modules

| Stage | Code |
|---|---|
| Instruction Parser | `groundedcad/agents/instruction_parser.py` |
| Geometry Inspector | `groundedcad/geometry/inspect.py` |
| Classify edit | `groundedcad/agents/classifier.py` |
| Strategy | `groundedcad/agents/patterns.py` |
| Minimal Edit Plan | `groundedcad/agents/planner.py` |
| CadQuery Generator | `groundedcad/tools/cadquery_gen.py` |
| Execution | `groundedcad/runtime/sandbox.py` (CadQuery; official harness) |
| Geometry Validator | `groundedcad/verify/geometric.py` |
| Render 7 views | `groundedcad/geometry/render.py` `CANONICAL_VIEWS` |
| Edit Critic | `groundedcad/verify/agent.py` — instruction, preservation, dimensions, position, count, topology, magnitude → `verification.json` |
| Loop | `groundedcad/pipeline.py` |

Revisions always restart from the **original** starting STEP.

## Optimization hierarchy (edits, not generations)

The official task measures **edits**, not generations. Rank objectives in this order only:

1. Instruction correctness
2. Correct edited feature
3. Preserve original geometry (unrelated change hurts Diff F1)
4. Correct dimensions / location
5. Valid topology
6. Visual similarity **last**

                         USER INSTRUCTION
                                │
                                ▼
                     ┌────────────────────┐
                     │ Instruction Parser │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │ Geometry Inspector │
                     │ OCC / CadQuery     │
                     │ bbox, holes, axes, │
                     │ normals, symmetry  │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │ Edit classifier +  │
                     │ per-type strategy  │
                     └─────────┬──────────┘
                               │
                         (slots + MODEL brief)
                               ▼
                     ┌────────────────────┐
                     │ LLM slot-fill only │
                     └─────────┬──────────┘
                               │
                               ▼
                       CadQuery tools
                               │
                               ▼
                         STEP output
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             Edit-delta validator    Render views
                    │                     │
                    └──────────┬──────────┘
                               ▼
                         Edit Critic
                               │
                         incorrect → revise
                         correct   → FINAL

OCC computes geometry facts (`geometry_brief`). The LLM does not infer holes from an image.

Edit-delta (`groundedcad/verify/edit_delta.py`) compares START → PRED to the requested slots. Envelope / thickness / extra bodies that were not requested fail as **unintended geometry**.

A/B: `python scripts/ab_compare.py --n-rows 8` writes `docs/ab_compare.json` (A = frozen `docs/baseline_scores.json`).

