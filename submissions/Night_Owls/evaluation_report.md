# GroundedCAD baseline (text-conditioned 48)

Scored with `python scripts/score_groundedcad_baseline.py`. Predictions: `data/model_outputs/.../groundedcad/` (`tmp.step`). GT/start: dataset STLs. **41/48** have a prediction; missing edits score **0**.

This is **not** the official Open3D / voxel-128 pipeline. Voxel divisor is 64, occupancy is vertex-based + hole fill. Use it to rank **edit classes**, not to quote leaderboard numbers.

## Overall

| Split | n | Chamfer | Volume F1 | Diff F1 | Valid |
|---|---:|---:|---:|---:|---:|
| All 48 (missing=0) | 48 | 0.830 | 0.543 | 0.073 | 41/48 |
| Scored only | 41 | 0.972 | 0.636 | 0.086 | 41/41 |

Chamfer is high because most outputs stay close to the start solid. **Diff F1 is the metric that is dying.**

## Mean by edit class (missing scored 0)

| Type | n | Chamfer | Volume F1 | Diff F1 | Valid |
|---|---:|---:|---:|---:|---:|
| feature_translation | 1 | .978 | .639 | **.243** | 1/1 |
| fillet_chamfer | 14 | .884 | .491 | .104 | 13/14 |
| hole_edit | 7 | .691 | .499 | .088 | 5/7 |
| pattern | 6 | .791 | .468 | .076 | 5/6 |
| ambiguous | 2 | .493 | .462 | .065 | 1/2 |
| feature_addition | 11 | .894 | .645 | **.048** | 10/11 |
| dimension_change | 3 | .660 | .404 | .047 | 2/3 |
| feature_deletion | 2 | .977 | .847 | **.015** | 2/2 |
| boolean_modification | 2 | .985 | .818 | **.000** | 2/2 |

On the 41 scored rows, class Diff F1 is similar except missing-pred zeros: **boolean 0.000**, **deletion 0.015**, **addition 0.052**, **pattern 0.091**, **hole 0.124**, **fillet 0.112**.

## Per-edit table

| # | Type | Chamfer | Vol F1 | Diff F1 | Valid? | Instruction (truncated) |
|---:|---|---:|---:|---:|:---:|---|
| 001 | fillet_chamfer | .984 | .796 | .290 | ✓ | 0.2 mm chamfer on hole edges |
| 002 | hole_edit | .976 | .659 | .058 | ✓ | 1.7 mm hole + 0.1 mm grooves |
| 003 | feature_addition | .982 | .793 | .085 | ✓ | 1.5 mm rib |
| 004 | fillet_chamfer | .987 | .949 | .180 | ✓ | remove fillet, add 1 mm chamfer |
| 005 | ambiguous | .987 | .925 | .130 | ✓ | flower → hex profile |
| 006 | fillet_chamfer | .987 | .936 | .156 | ✓ | round edges → spur teeth |
| 007 | boolean_modification | .983 | .838 | .000 | ✓ | slot through complete body |
| 008 | feature_addition | .962 | .636 | .028 | ✓ | 200 mm fixation rod |
| 009 | hole_edit | .983 | .753 | .142 | ✓ | insert screws in both holes |
| 010 | feature_addition | .981 | .705 | .190 | ✓ | inlet/outlet ports |
| 011 | feature_addition | .979 | .757 | .101 | ✓ | radiator filling system |
| 012 | pattern | .981 | .708 | .137 | ✓ | slot pattern top and bottom |
| 013 | fillet_chamfer | .983 | .546 | .009 | ✓ | 2 mm fillet on scroll wheel |
| 014 | feature_addition | .983 | .544 | .003 | ✓ | sliding switch |
| 015 | feature_addition | .983 | .517 | .001 | ✓ | two 2 mm click buttons |
| 016 | fillet_chamfer | .981 | .166 | .057 | ✓ | rounds all edges R=0.2 mm |
| 017 | fillet_chamfer | .569 | .006 | .001 | ✓ | scale 10× + drafts + rounds |
| 018 | hole_edit | .911 | .561 | .242 | ✓ | 0.5 mm flange (not fill center) |
| 019 | dimension_change | .992 | .334 | .141 | ✓ | frame +0.5 cm taller |
| 020 | fillet_chamfer | .989 | .256 | .118 | ✓ | support by offsetting inner |
| 021 | fillet_chamfer | .994 | .284 | .000 | ✓ | replace largest radius |
| 022 | feature_addition | .990 | .859 | .115 | ✓ | cylindrical pin in bearing |
| 023 | feature_addition | .991 | .824 | .000 | ✓ | locking mechanism at hook |
| 024 | boolean_modification | .986 | .798 | .000 | ✓ | cutouts to decrease weight |
| 025 | fillet_chamfer | .992 | .572 | .133 | ✓ | radius on missing blade edge |
| 026 | pattern | .954 | .427 | .045 | ✓ | third rotor blade |
| 027 | feature_addition | .990 | .610 | .000 | ✓ | two cover parts |
| 028 | feature_translation | .978 | .639 | .243 | ✓ | prolong lever +5 cm |
| 029 | fillet_chamfer | .976 | .644 | .260 | ✓ | rectangular opening (misclassified fillet) |
| 030 | pattern | .977 | .643 | .269 | ✓ | two extra front buttons |
| 031 | feature_deletion | .976 | .817 | .030 | ✓ | add matching stand (misclassified deletion) |
| 032 | fillet_chamfer | .978 | .838 | .000 | ✓ | Europlug (misclassified fillet) |
| 033 | feature_deletion | .978 | .877 | .000 | ✓ | remove handle/coffeepot collision |
| 034 | fillet_chamfer | .981 | .613 | .037 | ✓ | missing whistle radii |
| 035 | fillet_chamfer | .978 | .263 | .215 | ✓ | make whistle symmetric |
| 036 | hole_edit | .982 | .707 | .044 | ✓ | hang ring D=4 cm / hole 2 cm |
| 037 | dimension_change | .988 | .878 | .000 | ✓ | shallower cutout by 10 mm |
| 038 | feature_addition | .989 | .851 | .000 | ✓ | ZX symmetry + embossed text |
| 039 | hole_edit | .988 | .812 | .132 | ✓ | inscribed hex in Sketch4 hole |
| 040 | pattern | .870 | .643 | .000 | ✓ | mirror and merge |
| 041 | pattern | .963 | .384 | .006 | ✓ | multiply cylindrical feature to 4 |
| 042 | pattern | .000 | .000 | .000 | ✗ | 8× bearing instances (no pred) |
| 043 | fillet_chamfer | .000 | .000 | .000 | ✗ | heatbreak chamfer (no pred) |
| 044 | ambiguous | .000 | .000 | .000 | ✗ | 3-point → 4-point mount (no pred) |
| 045 | dimension_change | .000 | .000 | .000 | ✗ | nozzle −1 mm height (no pred) |
| 046 | hole_edit | .000 | .000 | .000 | ✗ | more holes in platforms (no pred) |
| 047 | hole_edit | .000 | .000 | .000 | ✗ | long-hole profile + diameters (no pred) |
| 048 | feature_addition | .000 | .000 | .000 | ✗ | pin heads (no pred) |

Raw JSON: `docs/baseline_scores.json`.

## What is killing the score

1. **Wrong-class / identity edits (Diff F1 ≈ 0)** while Chamfer stays ~0.98: through-cuts, weight-reduction booleans, deletions, locking mechanisms, covers, mirror-merge. The solid looks like the start; GT changed a different region.
2. **Feature addition of new parts** (rods, buttons, switches, covers): Volume F1 can look OK; Diff F1 does not.
3. **Global ops mislabeled as fillet** (scale 10×): only row with Chamfer collapse (.57) and Volume F1 ~0.
4. **Coverage**: 7/48 missing predictions are automatic zeros (pattern / hole / dimension / addition).
5. Classifier noise: some “fillet” rows are openings, plugs, or symmetry — do not optimize fillets just because row 001 Chamfer is .98.
