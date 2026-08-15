# Edit patterns from the 48 text-conditioned requests

These labels come from reading `val_edit_text.parquet` (48 instructions), not from the README.

Human GT scores in `data/edit_192_external/results/all_results.json` stay high on Chamfer when the instruction is a **local** blend or hole. Visual similarity is **not** the objective: a convincing picture that rewrites unrelated faces loses Diff F1.

That is the reusable strategy: classify, then apply a small specialized edit. Preserve everything the instruction did not name.

```text
Instruction
     ↓
Classify edit
     ↓
┌───────────────────────┐
│ Hole edit             │
│ Feature translation   │
│ Feature addition      │
│ Feature deletion      │
│ Dimension change      │
│ Pattern               │
│ Fillet/chamfer        │
│ Boolean modification  │
└───────────┬───────────┘
            ↓
Use specialized strategy
            ↓
Generate CAD edit
```

## What appears in the 48

| Pattern | Examples in the 48 | Strategy |
|---|---|---|
| Fillet/chamfer | 00 hole-edge chamfer; 03 replace fillet with chamfer; 12,15,20,24,33,42 rounds | `chamfer_*` / `fillet_*` on existing edges |
| Hole edit | 01 connecting hole+grooves; 17 mounting holes; 35 concentric hole; 38 hex through hole; 45 hole pattern; 46 change hole Ø | `drill_hole_at_point` (+ optional chamfer) |
| Feature addition | 02 rib; 07 rod; 08 screws; 09–10 ports; 13–14 switch/buttons; 21 pin; 26 covers; 47 pin heads | `add_box` / LLM only if shape is novel |
| Feature deletion | 03 remove fillet; 30 remove cordholder; 32 remove collision | local cut, not a rebuild |
| Feature translation | 27 prolong lever; 34 move surface for symmetry | `translate_body` or local move |
| Dimension change | 16 scale 10x; 18 taller 0.5cm; 36 shallower 10mm; 44 height −1mm; 46 Ø +1mm | `scale_uniform` or hole re-cut |
| Pattern / symmetry | 11 slot pattern; 25 copy blade; 29 copy buttons; 39 mirror; 40×4; 41×8 rotate | `duplicate_linear` |
| Boolean / localized cut | 06 slot through; 23 cutouts; 28 rectangular opening | `boolean_cut_box` |
| Ambiguous / profile | 04 flower→hex; 05 spur teeth; 31 Europlug; 22 locking mechanism | LLM with census, still local |

Code: `groundedcad/agents/patterns.py` (`classify_edit`, `apply_strategy`).
