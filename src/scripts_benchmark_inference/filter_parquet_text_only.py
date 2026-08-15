#!/usr/bin/env python3
"""
Create a text-only request parquet for the harness.

The shipped ``val_edit_all.parquet`` contains all 192 edit requests across every
modality (video, transcript, events, text). The harness (run_harness.py) runs one
row per request, so to evaluate only the text-conditioned subset we filter down to
rows whose ``request_text`` column is populated.

This has been verified to exactly match the DB's ``modality == "text"`` set
(the same subset produced by ``src/scripts/filter_dataset_text_only.py``), so no
database access is required here.
"""

import argparse
import os.path as osp

import numpy as np
import pandas as pd

TEXT_COLUMN = "request_text"


def _is_non_empty(value) -> bool:
    """Return True when a parquet cell holds a real (non-empty) value."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, (list, np.ndarray)):
        return len(value) > 0
    if isinstance(value, str):
        return value.strip() not in ("", "[]", "null", "None")
    return bool(value)


def filter_parquet_text_only(input_path: str, output_path: str) -> int:
    """
    Write a text-only request parquet from an unfiltered request parquet.

    Keeps only rows whose ``request_text`` column is populated (verified to match
    the DB's ``modality == "text"`` subset). Returns the number of rows kept.
    """
    df = pd.read_parquet(input_path)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{TEXT_COLUMN}' not found in {input_path}. "
            f"Available columns: {list(df.columns)}"
        )

    mask = df[TEXT_COLUMN].apply(_is_non_empty)
    filtered = df[mask].reset_index(drop=True)

    filtered.to_parquet(output_path, index=False)
    print(f"Read {len(df)} requests from {input_path}")
    print(f"Kept {len(filtered)} text-only requests -> {output_path}")
    return len(filtered)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="data/edit_192_external/parquets/val_edit_all.parquet",
        help="Path to the unfiltered request parquet.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path. Defaults to '<input_dir>/val_edit_text.parquet'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = args.output
    if output_path is None:
        output_path = osp.join(osp.dirname(args.input), "val_edit_text.parquet")

    filter_parquet_text_only(args.input, output_path)


if __name__ == "__main__":
    main()
