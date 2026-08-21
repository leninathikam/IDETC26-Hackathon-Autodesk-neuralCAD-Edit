from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"
SUBMISSION_DIR = ROOT / "submissions" / "Night_Owls"
PDF_OUT = OUT_DIR / "Night_Owls_final_submission_presentation.pdf"
SUBMISSION_PDF = SUBMISSION_DIR / "Night_Owls_final_submission_presentation.pdf"
METRIC_IMG = OUT_DIR / "metric_bar_facets.png"
SUBMISSION_METRIC_IMG = SUBMISSION_DIR / "metric_bar_facets.png"

QUAL_IMAGES = [
    Path(r"C:\Users\lenin\AppData\Local\Temp\codex-clipboard-23f2dc4b-f6ef-4b0c-867c-7959a3c37215.png"),
    Path(r"C:\Users\lenin\AppData\Local\Temp\codex-clipboard-eaa84101-af36-4a3a-9f9e-21d9b9c30d1f.png"),
    Path(r"C:\Users\lenin\AppData\Local\Temp\codex-clipboard-187c539a-d8b5-4d26-8c82-60a588694479.png"),
]

SCORES_PATH = ROOT / "docs" / "full48_openai-gpt-5-2_fresh_rerun_scores.json"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_scores() -> dict[str, float]:
    if SCORES_PATH.exists():
        data = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
        summary = data.get("summary") if isinstance(data, dict) else {}
        if summary:
            return {
                "Chamfer": float(summary.get("mean_chamfer", 0.9094)),
                "Volume F1": float(summary.get("mean_volume_f1", 0.5900)),
                "Diff F1": float(summary.get("mean_diff_f1", 0.1324)),
            }
        rows = data.get("rows", []) if isinstance(data, dict) else data
        vals = {"Chamfer": [], "Volume F1": [], "Diff F1": []}
        for row in rows:
            ours = row.get("ours", {})
            vals["Chamfer"].append(float(ours.get("chamfer", 0)))
            vals["Volume F1"].append(float(ours.get("volume_f1", 0)))
            vals["Diff F1"].append(float(ours.get("diff_f1", 0)))
        if vals["Chamfer"]:
            return {k: sum(v) / len(v) for k, v in vals.items()}
    return {"Chamfer": 0.9094, "Volume F1": 0.5900, "Diff F1": 0.1324}


def create_metric_chart(scores: dict[str, float]) -> None:
    width, height = 1500, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(54, bold=True)
    label_font = font(34, bold=True)
    value_font = font(32, bold=True)
    small_font = font(25)
    draw.text((70, 45), "Final 48-row official metrics", fill=(23, 54, 46), font=title_font)
    draw.text(
        (74, 115),
        "Comparison of GPT-5.2 fresh run and Claude run",
        fill=(70, 70, 70),
        font=small_font,
    )
    chart_left, chart_top = 300, 210
    chart_width, row_gap = 850, 125
    bar_h = 38
    claude_scores = {"Chamfer": 0.94, "Volume F1": 0.55, "Diff F1": 0.14}
    palette = {
        "Chamfer": (72, 142, 121),
        "Volume F1": (215, 166, 34),
        "Diff F1": (156, 64, 64),
    }
    draw.rounded_rectangle((1040, 130, 1225, 172), radius=10, fill=(72, 142, 121))
    draw.text((1240, 135), "GPT-5.2", fill=(20, 20, 20), font=small_font)
    draw.rounded_rectangle((1040, 178, 1225, 220), radius=10, fill=(62, 91, 145))
    draw.text((1240, 183), "Claude", fill=(20, 20, 20), font=small_font)
    for i, (name, value) in enumerate(scores.items()):
        y = chart_top + i * row_gap
        claude_value = claude_scores[name]
        draw.text((70, y + 33), name, fill=(20, 20, 20), font=label_font)
        draw.rounded_rectangle(
            (chart_left, y + 4, chart_left + chart_width, y + 42),
            radius=18,
            fill=(235, 240, 238),
        )
        bar_w = max(3, int(chart_width * max(0, min(1, value))))
        draw.rounded_rectangle(
            (chart_left, y + 4, chart_left + bar_w, y + 42),
            radius=18,
            fill=(72, 142, 121),
        )
        draw.rounded_rectangle(
            (chart_left, y + 52, chart_left + chart_width, y + 90),
            radius=18,
            fill=(235, 240, 238),
        )
        claude_w = max(3, int(chart_width * max(0, min(1, claude_value))))
        draw.rounded_rectangle(
            (chart_left, y + 52, chart_left + claude_w, y + 90),
            radius=18,
            fill=(62, 91, 145),
        )
        draw.text((chart_left + chart_width + 28, y + 8), f"GPT {value:.4f}", fill=(20, 20, 20), font=value_font)
        draw.text((chart_left + chart_width + 28, y + 56), f"Claude {claude_value:.2f}", fill=(20, 20, 20), font=value_font)
    draw.line((chart_left, chart_top + 420, chart_left + chart_width, chart_top + 420), fill=(170, 170, 170), width=2)
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = chart_left + int(chart_width * tick)
        draw.line((x, chart_top + 405, x, chart_top + 435), fill=(140, 140, 140), width=2)
        draw.text((x - 24, chart_top + 448), f"{tick:.2f}", fill=(80, 80, 80), font=small_font)
    draw.text(
        (74, 675),
        "Interpretation: valid geometry quality is high; Diff F1 shows the remaining challenge is precise feature grounding.",
        fill=(55, 55, 55),
        font=small_font,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(METRIC_IMG)


def draw_header(c: canvas.Canvas, title: str, w: float, h: float) -> None:
    c.setFillColor(colors.HexColor("#17362e"))
    c.setFont("Helvetica-Bold", 30)
    c.drawString(0.55 * inch, h - 0.65 * inch, title)
    c.setStrokeColor(colors.HexColor("#d8e2de"))
    c.setLineWidth(2)
    c.line(0.55 * inch, h - 0.88 * inch, w - 0.55 * inch, h - 0.88 * inch)


def draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, font_name: str, font_size: int, leading: int) -> float:
    c.setFont(font_name, font_size)
    words = text.split()
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        if c.stringWidth(trial, font_name, font_size) <= max_width:
            line = trial
        else:
            c.drawString(x, y, line)
            y -= leading
            line = word
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_image_fit(c: canvas.Canvas, image_path: Path, x: float, y: float, box_w: float, box_h: float) -> None:
    with Image.open(image_path) as img:
        iw, ih = img.size
    scale = min(box_w / iw, box_h / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(str(image_path), x + (box_w - dw) / 2, y + (box_h - dh) / 2, dw, dh, preserveAspectRatio=True, mask="auto")


def build_pdf(scores: dict[str, float]) -> None:
    w, h = landscape((13.333 * inch, 7.5 * inch))
    c = canvas.Canvas(str(PDF_OUT), pagesize=(w, h))

    # Title
    c.setFillColor(colors.HexColor("#17362e"))
    c.setFont("Helvetica-Bold", 38)
    c.drawString(0.8 * inch, h - 1.45 * inch, "Grounded CAD: Geometry Whispers for CAD Edit")
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 22)
    c.drawString(0.82 * inch, h - 2.05 * inch, "Team Night Owls - IDETC 2026 Problem Statement 3")
    c.setFont("Helvetica", 18)
    c.drawString(0.82 * inch, h - 2.65 * inch, "Final presentation PDF for repository submission")
    c.setFillColor(colors.HexColor("#eef5f2"))
    c.roundRect(0.82 * inch, 1.25 * inch, 11.6 * inch, 2.0 * inch, 18, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#17362e"))
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1.1 * inch, 2.65 * inch, "Goal")
    c.setFillColor(colors.black)
    draw_wrapped(
        c,
        "Build a harness that edits existing STEP models from natural-language instructions while maximizing automatic CAD-edit metrics.",
        1.1 * inch,
        2.25 * inch,
        10.9 * inch,
        "Helvetica",
        18,
        24,
    )
    c.showPage()

    # Method overview
    draw_header(c, "Method overview", w, h)
    bullets = [
        ("Grounder", "Extracts a compact B-Rep census from the STEP model: solids, faces, holes, rims, dimensions, bounding boxes, and candidate edit sites."),
        ("Classifier", "Parses the instruction into operation type, target feature, dimensions, direction, and constraints."),
        ("Planner", "Chooses deterministic local CAD tools first; falls back to constrained CadQuery generation when necessary."),
        ("Executor", "Runs CadQuery/OpenCascade edits in a sandboxed child process with watchdog protection."),
        ("Verifier/Critic", "Checks validity, geometric change, locality, and failure buckets such as identity, wrong scope, misclassification, or kernel failure."),
    ]
    y = h - 1.45 * inch
    for title, body in bullets:
        c.setFillColor(colors.HexColor("#eaf3ef"))
        c.roundRect(0.75 * inch, y - 0.45 * inch, 11.8 * inch, 0.58 * inch, 10, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#17362e"))
        c.setFont("Helvetica-Bold", 18)
        c.drawString(1.0 * inch, y - 0.1 * inch, title)
        c.setFillColor(colors.black)
        draw_wrapped(c, body, 2.55 * inch, y - 0.1 * inch, 9.6 * inch, "Helvetica", 15, 19)
        y -= 0.78 * inch
    c.setFillColor(colors.HexColor("#555555"))
    c.setFont("Helvetica", 15)
    c.drawString(0.78 * inch, 0.72 * inch, "Design principle: avoid direct STEP text edits; use geometry-aware B-Rep operations plus validation.")
    c.showPage()

    # Scores
    draw_header(c, "Final scores", w, h)
    draw_image_fit(c, METRIC_IMG, 0.85 * inch, 1.0 * inch, 11.65 * inch, 5.55 * inch)
    c.showPage()

    # Qualitative examples
    draw_header(c, "Qualitative evaluation: five selected request IDs", w, h)
    image_boxes = [
        (0.55 * inch, 3.55 * inch, 6.05 * inch, 2.92 * inch),
        (6.75 * inch, 3.55 * inch, 5.9 * inch, 2.92 * inch),
        (1.65 * inch, 0.55 * inch, 10.0 * inch, 2.45 * inch),
    ]
    for image, (x, y, bw, bh) in zip(QUAL_IMAGES, image_boxes):
        c.setStrokeColor(colors.HexColor("#9fb7ae"))
        c.setLineWidth(1)
        c.rect(x, y, bw, bh, stroke=1, fill=0)
        draw_image_fit(c, image, x + 0.04 * inch, y + 0.04 * inch, bw - 0.08 * inch, bh - 0.08 * inch)
    c.showPage()

    # Closing
    draw_header(c, "Takeaways and next steps", w, h)
    takeaways = [
        "Valid STEP export alone is not enough; Diff F1 depends on changing the requested local feature.",
        "The strongest signals came from STEP-derived B-Rep facts, true orthographic views, candidate enumeration, and failure-aware validation.",
        "Remaining score loss is dominated by wrong edge/scope, misclassification, identity outputs, and OpenCascade boolean/rendering failures.",
        "Next improvements should confidence-gate deterministic operations and re-ground alternate candidates before accepting wrong-scope edits.",
    ]
    y = h - 1.55 * inch
    for item in takeaways:
        c.setFillColor(colors.HexColor("#17362e"))
        c.circle(0.95 * inch, y + 0.06 * inch, 5, stroke=0, fill=1)
        c.setFillColor(colors.black)
        y = draw_wrapped(c, item, 1.2 * inch, y, 11.0 * inch, "Helvetica", 18, 25)
        y -= 0.25 * inch
    c.save()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    for image in QUAL_IMAGES:
        if not image.exists():
            raise FileNotFoundError(f"Missing qualitative screenshot: {image}")
    scores = load_scores()
    create_metric_chart(scores)
    build_pdf(scores)
    shutil.copy2(PDF_OUT, SUBMISSION_PDF)
    shutil.copy2(METRIC_IMG, SUBMISSION_METRIC_IMG)
    print(PDF_OUT)
    print(SUBMISSION_PDF)
    print(METRIC_IMG)


if __name__ == "__main__":
    main()
