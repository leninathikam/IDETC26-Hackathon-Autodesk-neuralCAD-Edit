# Failure taxonomy (GroundedCAD on 48 text edits)

Classified from **instruction vs tool actually executed** (`iterations/*/log.json`) plus Diff F1. The critic marked many of these `accept=True` anyway; that is a separate **false completion** overlay, not a geometry class.

Almost every row is a Diff F1 failure. Chamfer ~0.98 only means the solid still looks like the start.

```
FAILURES
│
├── Coverage (no prediction)                         7
│
├── Interpretation errors                            14
│   ├── wrong operation (keyword hijack)             8
│   ├── wrong target / underspecified add            5
│   └── misunderstood instruction (multi-clause)     1
│
├── Preservation errors                              22
│   ├── unchanged / identity start STEP              20
│   ├── changed unrelated feature                    1
│   └── changed global dimensions                    1
│
├── Geometry identification                          4
│   ├── selected wrong hole / cylinder               2
│   ├── selected wrong edges (length-sorted)         2
│   └── selected wrong feature                       (overlaps interp)
│
├── Dimension errors                                 5
│   ├── wrong diameter                               3
│   ├── wrong blend radius / distance                2
│   └── wrong depth                                  0 scored
│
├── Position errors                                  3
│   ├── wrong XYZ (census / bbox center)             3
│   └── wrong reference                              (Sketch4 unused)
│
└── Topology / execution                             4
    ├── invalid / zero-volume solid                  2
    ├── failed boolean / api_code                    2
    └── unintended extra body                        1
```

Primary label is the **first** cause that explains Diff F1. Secondary tags in the table.

## Dominant mechanism (not “needs better reasoning”)

1. **`raw_cadquery` fallback is an identity edit.** `heuristic_needs_llm` sends feature-add / ambiguous / deletion to the LLM. When the LLM returns `raw_cadquery` without a working script, the scaffold copies the start solid. Sibling requests on the same start part have **identical volume** (e.g. 07–09 = 426801; 10–12 = 7.255e6; 22–24 = 16217). Diff F1 is 0 because GT changed voxels and we changed none.

2. **Keyword classifier picks the first regex hit.** `fillet|chamfer|rounds|radius` beats “scale 10x”, “Europlug”, “mirror”, “make symmetric”. The executor then fillets 12 longest edges at default **r=0.5**.

3. **Constrained tools bind to the wrong entity.** `chamfer_circular_edges` / `fillet_edges_by_length` take up to 12 edges by length, not “the hole edges” / “the scroll-wheel slot” / “the long blade edge without a radius”. `drill_hole_at_point` uses a grounded face center; default diameter **4.0** when parse fails.

4. **Verifier does not look at occupancy XOR vs start.** `accept=True` on identity and on wrong-op fillets. `failure_category` is almost always `None`.

## Per-example labels

| # | Diff F1 | Tool that ran | Primary | Secondary | Why |
|---:|---:|---|---|---|---|
| 001 | .290 | `chamfer_circular_edges` d=0.2 | geometry: wrong edges | dim OK | Asked hole-edge 0.2 mm; tool chamfers up to 12 circular edges |
| 002 | .058 | `drill_hole_at_point` Ø**4.0** at census | dimension: diameter | position: XYZ; interp: missed grooves | Asked Ø1.7 + 0.1 mm grooves |
| 003 | .085 | `raw_cadquery` identity | preservation: identity | interp: rib | LLM fallback, no rib |
| 004 | .180 | `fillet_edges_by_length` r=0.5 | interp: wrong op | dim: 0.5 vs 1 mm chamfer; geometry: not front-center | Asked **remove** fillet then **chamfer** 1 mm |
| 005 | .130 | `raw_cadquery` identity | interp: wrong target | preservation: identity | Flower → hex needs profile replace; no tool |
| 006 | .156 | `fillet_edges_by_length` r=0.5 | interp: wrong op | geometry: wrong edges | “Round edges” ≠ fillet; asked spur teeth |
| 007 | .000 | `raw_cadquery` identity | preservation: identity | interp: through-cut | Slot-through never applied |
| 008 | .028 | `raw_cadquery` identity | interp: new part | preservation: identity | 200 mm rod not in toolset |
| 009 | .142 | `raw_cadquery` identity | interp: wrong op | preservation: identity | “Insert screws” ≠ hole edit; no fasteners |
| 010 | .190 | `raw_cadquery` identity | interp: new features | position: top-right / bottom-left unused | Inlet/outlet ports |
| 011 | .101 | `raw_cadquery` identity | interp: new features | preservation: identity | Pouring + cap |
| 012 | .137 | `raw_cadquery` identity | interp: wrong op | preservation: identity | Slot **pattern** not `duplicate_linear` |
| 013 | .009 | `fillet_edges_by_length` r=**0.5** | geometry: wrong edges | dimension: 0.5 vs 2 mm | Not the scroll-wheel slot |
| 014 | .003 | `raw_cadquery` identity | interp: new part | preservation: identity | Sliding switch |
| 015 | .001 | `raw_cadquery` identity | interp: new parts | preservation: identity | Two 2 mm buttons |
| 016 | .057 | `raw_cadquery` identity | preservation: identity | dim: R=0.2 unused | All-edge rounds never ran (comma decimal) |
| 017 | .002 | `drill_hole_at_point` Ø2 | interp: wrong op | preservation: global size **not** 10× | Regex: `rounds` → fillet family → hole; no scale/draft |
| 018 | .242 | `raw_cadquery` identity | interp: flange vs hole | preservation: identity | Classifier said hole_edit because of “central part”; no flange |
| 019 | .141 | `raw_cadquery` identity | interp: height vs rebuild | preservation: identity | +0.5 cm taller, profile fixed; no extrude |
| 020 | .118 | `raw_cadquery` (LLM described offset) | preservation: identity | interp: support ≠ fillet | Script did not apply the described 20 mm offset |
| 021 | .000 | `fillet_circular_edges` **failed** | topology: api_code | interp: replace radius | Submitted start STEP after tool error |
| 022 | .115 | `raw_cadquery` identity | interp: add pin | geometry: bearing area unused | |
| 023 | .000 | `raw_cadquery` identity | interp: locking mechanism | preservation: identity | |
| 024 | .000 | `raw_cadquery` identity | interp: cutouts | preservation: identity | Weight-reduction boolean never ran |
| 025 | .133 | `fillet_edges_by_length` r=**0.5** | geometry: wrong edge | dimension: 0.5 vs **6.35 mm** | Asked the one long edge missing a radius |
| 026 | .045 | `raw_cadquery` identity | interp: copy body | preservation: identity | Third blade; `duplicate_linear` not used |
| 027 | .000 | `raw_cadquery` identity | interp: new covers | preservation: identity | |
| 028 | .243 | `raw_cadquery` identity | interp: prolong feature | preservation: identity | 5 cm lever; `translate_body` would have moved **whole** assembly (56 solids) |
| 029 | .260 | `boolean_cut_box` **failed** | topology: api_code | position: guessed rear Y | Opening misclassified as fillet; cut args invalid |
| 030 | .269 | `raw_cadquery` identity | interp: add buttons | preservation: identity | Pattern of two buttons |
| 031 | .030 | `raw_cadquery` identity | interp: copy stand | geometry: “other side” unused | Misclassified deletion because of later text |
| 032 | .000 | `fillet_edges_by_length` r=0.5 | interp: wrong op | preservation: unrelated blends | Europlug; `\brounds?\b` matched “two round pins” |
| 033 | .000 | `raw_cadquery` identity | interp: collision trim | preservation: identity | Deletion/boolean never ran |
| 034 | .037 | `raw_cadquery` identity | geometry: left/bigger part unused | preservation: identity | Radii-same-as-existing never measured |
| 035 | .215 | `fillet_edges_by_length` r=5 | interp: wrong op | geometry: wrong edges | Asked **move** protruding face for symmetry |
| 036 | .044 | `drill_hole_at_point` Ø**4.0** | dimension: diameter | interp: missing boss; position | Asked **add** D=40 mm disk + D=20 mm hole (cm); drilled Ø4 at a cylinder |
| 037 | .000 | `raw_cadquery` vol=**0** | topology: invalid solid | preservation: identity-ish | Shallower cutout 10 mm not applied |
| 038 | .000 | `raw_cadquery` vol=**0** | topology: invalid solid | interp: plateau + text | |
| 039 | .132 | `boolean_cut_box` at origin | position: wrong XYZ | interp: hex vs box; geometry: Sketch4 unused | Inscribed hex in hole → axis-aligned box cut |
| 040 | .000 | `fillet_edges_by_length` r=0.5 | interp: wrong op | preservation: unrelated | Mirror+merge; “rounded end” triggered fillet |
| 041 | .006 | `raw_cadquery` identity | interp: pattern count | preservation: identity | Multiply cylinders to 4 + legs |
| 042–048 | .000 | **no pred** | coverage | — | Batch stopped at 41 |

## Counts that should drive the next code change

| If you fix this | Rows it would even be *possible* to score | Why Diff F1 is 0 today |
|---|---|---|
| Stop submitting identity `raw_cadquery` / start STEP as success | ~20 | GT voxels changed, pred XOR is empty |
| Stop keyword-first fillet/hole | 006, 017, 032, 035, 040 | Wrong tool family |
| Bind blends to named edges, not 12 longest | 001, 004, 013, 025, 034 | Local but wrong locus |
| Parse mm/cm and stop default Ø4 | 002, 016, 025, 036 | Dim miss even when op is close |
| Feature-add tools that are not “one box at census center” | 008, 010, 011, 014, 015, 022, 023, 027, 030 | No matching operator |
| Finish the 7 missing preds | 042–048 | Automatic zeros |

Do **not** start with “better LLM reasoning” on fillets: row 001 already chamfers and still only Diff F1 .29 because of **edge selection**. Do **not** enable `translate_body` / `duplicate_linear` / `scale_uniform` as the default for 028/026/017: those tools mutate the **whole** solid (preservation: global).
