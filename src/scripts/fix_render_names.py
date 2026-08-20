"""One-off fix: copy tmp.png -> toprightiso.png in every groundedcad output
folder so the DB's image-detection logic (which requires filenames ending in
"toprightiso.png") picks them up for DINO feature extraction.

Safe to re-run - skips folders that already have toprightiso.png.

Usage (PowerShell, from repo root):
    uv run python scripts/fix_render_names.py
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "data" / "model_outputs"


def main():
    if not OUTPUTS.exists():
        raise SystemExit(f"Can't find {OUTPUTS}")

    fixed = 0
    skipped_no_png = 0
    already_ok = 0

    for tmp_png in OUTPUTS.rglob("tmp.png"):
        dest = tmp_png.parent / "toprightiso.png"
        if dest.exists():
            already_ok += 1
            continue
        shutil.copy2(tmp_png, dest)
        fixed += 1

    print(f"Copied {fixed} new toprightiso.png files")
    print(f"Already had toprightiso.png: {already_ok}")

    # Also report folders with NO tmp.png at all - these have no render
    # to copy from, and are the ones that will still be missing DINO scores.
    no_render = []
    for settings in OUTPUTS.rglob("settings.json"):
        folder = settings.parent
        if not (folder / "tmp.png").exists() and not (folder / "toprightiso.png").exists():
            no_render.append(folder)

    print(f"\nFolders with NO render image at all: {len(no_render)}")
    for f in no_render[:10]:
        print(f"  {f}")
    if len(no_render) > 10:
        print(f"  ... and {len(no_render) - 10} more")


if __name__ == "__main__":
    main()
