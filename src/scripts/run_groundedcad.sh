#!/usr/bin/env bash
set -euo pipefail
CONFIG_FILE=src/config/edit_192_external.json
export PYTHONPATH="$(pwd)"

# uv run python src/scripts_benchmark_inference/filter_parquet_text_only.py \
#   --input data/edit_192_external/parquets/val_edit_all.parquet \
#   --output data/edit_192_external/parquets/val_edit_text.parquet

uv run python src/scripts_benchmark_inference/run_harness.py --config ${CONFIG_FILE} \
  --input data/edit_192_external/parquets/val_edit_text.parquet \
  --output_dir data/model_outputs \
  --harness src/harnesses/cadquery_script.py \
  --userId groundedcad \
  --required-extensions step

# uv run python src/scripts_preprocess/cadquery_convert.py data/model_outputs
uv run python src/scripts/run_all_benchmarks.py --config ${CONFIG_FILE}
