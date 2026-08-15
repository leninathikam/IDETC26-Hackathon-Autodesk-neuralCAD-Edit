# GroundedCAD — IDETC 2026 Autodesk Track (3-minute slide content)

## Slide 1 — Problem
- Engineers edit CAD far more than they generate it.
- neuralCAD-Edit: expert multimodal edit requests; GPT-class models still far below humans.
- Harness quality (tools, inspection, iteration) drives performance.

## Slide 2 — Method (GroundedCAD)
1. Inspect STEP → topology census  
2. Ground instruction (+ cursor/drawing cues) → targets  
3. Plan local `EditSpec` using constrained CadQuery tools  
4. Execute in sandbox → verify validity/volume/bbox/connectivity  
5. Independent critic accepts or forces revision (no false completion)

## Slide 3 — Why different from baseline
- Baseline: LLM freely rewrites CadQuery and self-declares done.
- Ours: explicit grounding, typed plans, deterministic gates, best-valid retention.
- CadQuery-first for headless reproducibility; Fusion optional for live demo.

## Slide 4 — Evidence (fill with numbers after API run)
- Synthetic offline suite: fillets, holes, transforms, duplicates, chamfers.
- Ablations: heuristic vs strict-verify vs full loop (`outputs/ablations/ablation_report.json`).
- Target on official text-48: ↑ validity, ↓ false completion, improved chamfer/DINO vs baseline harness.

## Slide 5 — Qualitative (5 examples)
1. Fillet box edges (2mm)  
2. Hole near boss (cursor-grounded)  
3. Linear duplicate  
4. Translate body  
5. Chamfer edges  

(Attach iso views from `outputs/submission_demo/`.)

## Slide 6 — Impact & next steps
- Closer to real collaborator-style CAD assist.
- Next: full neuralCAD-Edit text-48 scoring, stronger multimodal video grounding, Fusion timeline tools for parametric history edits.

## Contacts / repo
- Method: GroundedCAD agent harness
- Code: this repository
- Upstream benchmark: AutodeskAILab/neuralCAD-Edit
