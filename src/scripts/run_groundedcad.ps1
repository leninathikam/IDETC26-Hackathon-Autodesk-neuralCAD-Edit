# Run GroundedCAD on the official 48 text-edit parquet, then convert views.

$ErrorActionPreference = "Stop"
$CONFIG_FILE = "src/config/edit_192_external.json"
$env:PYTHONPATH = (Get-Location).Path

# 1) Filter parquet to text-only (48 requests) if you have val_edit_all.parquet
# uv run python src/scripts_benchmark_inference/filter_parquet_text_only.py `
#     --input data/edit_192_external/parquets/val_edit_all.parquet `
#     --output data/edit_192_external/parquets/val_edit_text.parquet

# 2) Run GroundedCAD harness
# Set OPENAI_API_KEY (or ANTHROPIC_API_KEY / GOOGLE_API_KEY) first.
# For offline dry-run: $env:GROUNDEDCAD_PROVIDER="mock"
uv run python src/scripts_benchmark_inference/run_harness.py --config $CONFIG_FILE `
    --input data/edit_192_external/parquets/val_edit_text.parquet `
    --output_dir data/model_outputs `
    --harness src/harnesses/cadquery_script.py `
    --userId groundedcad `
    --required-extensions step

# 3) Export STL + 7 views from STEP if missing
# uv run python src/scripts_preprocess/cadquery_convert.py data/model_outputs

# 4) Score
uv run python src/scripts/run_all_benchmarks.py --config $CONFIG_FILE
