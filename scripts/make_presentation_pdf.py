"""Create presentation HTML + PNG slides (and PDF when possible)."""

from __future__ import annotations

from pathlib import Path


def build_presentation(out_dir: Path) -> dict[str, Path]:
    from PIL import Image, ImageDraw, ImageFont

    slides = [
        (
            "GroundedCAD",
            [
                "IDETC/CIE 2026 — Autodesk Track",
                "AI harness for real-world 3D CAD editing",
                "neuralCAD-Edit aligned, CadQuery-first",
            ],
        ),
        (
            "Problem",
            [
                "Editing >> generation in engineering practice",
                "Frontier VLMs lag human CAD experts",
                "Harness design (tools/inspection/iteration) is decisive",
            ],
        ),
        (
            "Method",
            [
                "1. Geometry census of input STEP",
                "2. Temporal grounding (text/cursor/drawing)",
                "3. Local EditSpec via constrained tools",
                "4. Sandbox execute + deterministic verify",
                "5. Independent critic (no false completion)",
            ],
        ),
        (
            "Vs baseline harness",
            [
                "Baseline: free CadQuery rewrite + self-declare done",
                "Ours: grounded targets + typed plan + gates",
                "Best-valid retention under iteration budget",
                "Benchmark-compatible STEP/STL/settings.json",
            ],
        ),
        (
            "Results & examples",
            [
                "Synthetic suite: fillet, hole, duplicate, translate, chamfer",
                "Ablations in outputs/ablations/ablation_report.json",
                "Next: full text-48 neuralCAD-Edit auto metrics",
                "Attach 5 iso views for qualitative slide",
            ],
        ),
        (
            "Impact & next",
            [
                "Closer to collaborator-style CAD assistance",
                "Stronger video grounding + parametric Fusion tools",
                "Open, reproducible agent harness",
            ],
        ),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>GroundedCAD Presentation</title>",
        "<style>body{font-family:Arial,sans-serif;background:#0c121c;color:#e6ebf0;margin:0}",
        ".slide{max-width:960px;margin:24px auto;padding:32px;background:#121a28;border-radius:12px}",
        "h1{background:#145aa0;margin:-32px -32px 24px;padding:20px 32px}",
        "li{margin:10px 0;font-size:20px}</style></head><body>",
    ]
    png_files: list[Path] = []
    for idx, (title, bullets) in enumerate(slides, start=1):
        img = Image.new("RGB", (1280, 720), (12, 18, 28))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.rectangle([0, 0, 1280, 90], fill=(20, 90, 160))
        draw.text((40, 35), title, fill=(255, 255, 255), font=font)
        y = 140
        for b in bullets:
            draw.text((60, y), f"- {b}", fill=(230, 235, 240), font=font)
            y += 50
        draw.text((40, 670), "GroundedCAD — IDETC 2026", fill=(140, 160, 180), font=font)
        png_path = out_dir / f"slide_{idx:02d}.png"
        img.save(png_path, format="PNG")
        paths[f"slide_{idx:02d}"] = png_path
        png_files.append(png_path)
        html_parts.append(f"<section class='slide'><h1>{title}</h1><ul>")
        for b in bullets:
            html_parts.append(f"<li>{b}</li>")
        html_parts.append(f"</ul><p><img src='{png_path.name}' width='100%'/></p></section>")

    html_parts.append("</body></html>")
    html_path = out_dir / "GroundedCAD_Presentation.html"
    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    paths["html"] = html_path

    pdf_path = out_dir / "GroundedCAD_Presentation.pdf"
    try:
        import img2pdf

        pdf_path.write_bytes(img2pdf.convert([str(p) for p in png_files]))
        paths["pdf"] = pdf_path
    except Exception:
        # Fallback: write a printable single HTML into docs; PDF optional
        note = out_dir / "PDF_NOTE.txt"
        note.write_text(
            "Install img2pdf for PDF export: pip install img2pdf\n"
            "HTML + PNG slides are the primary presentation deliverable.\n",
            encoding="utf-8",
        )
        paths["note"] = note

    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    docs_html = docs / "GroundedCAD_Presentation.html"
    # Self-contained docs copy without relative png deps: inline text only
    docs_html.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'><title>GroundedCAD</title>",
                "<style>body{font-family:Arial;margin:40px} h1{color:#145aa0}</style></head><body>",
                *[
                    f"<h1>{t}</h1><ul>{''.join(f'<li>{b}</li>' for b in bullets)}</ul>"
                    for t, bullets in slides
                ],
                "</body></html>",
            ]
        ),
        encoding="utf-8",
    )
    paths["docs_html"] = docs_html
    if pdf_path.exists():
        (docs / "GroundedCAD_Presentation.pdf").write_bytes(pdf_path.read_bytes())
        paths["docs_pdf"] = docs / "GroundedCAD_Presentation.pdf"
    return paths


if __name__ == "__main__":
    result = build_presentation(Path("docs/presentation_assets"))
    print({k: str(v) for k, v in result.items()})
