CONFIG_FILE=src/config/edit_192_external.json
PYTHON_PATH=$(pwd)

# 1. CadQuery harness runs
# commented out as they're already in the database, but you can modify to do your own runs.
# MODELS=(
#     "gemini-3-pro_cadquery-script"
#     # "gpt-5.2_cadquery-script"
#     # "claude-sonnet-4.5_cadquery-script"
# )
# # Loop through all models
# for model in "${MODELS[@]}"; do
#     echo "Running harness for model: $model"
#     PYTHONPATH=${PYTHON_PATH} python src/scripts_benchmark_inference/run_harness.py --config ${CONFIG_FILE} \
#     --input /path/to/edit_database/edit_v2/parquets \
#     --output_dir /path/to/model_outputs/edit_v2 \
#     --harness src/harnesses/cadquery_script.py \
#     --userId "$model" \
#     --required-extensions step
# done

# 2. If your harness outputs only .step files, run cadquery_convert.py to generate .stl and view images:
# PYTHONPATH=${PYTHON_PATH} python src/scripts_preprocess/cadquery_convert.py <output_dir>

# 3. Ingest 
# commented out as the existing runs are already in the database, but you can modify to do your own ingests.
# PYTHONPATH=${PYTHON_PATH} python src/scripts/build_instructions_db.py --config ${CONFIG_FILE}

# 4. export for ground truth rating if required
# commented out as all ground truth ratings have already been done.
# PYTHONPATH=${PYTHON_PATH} python src/scripts_groundtruth/export_for_gt.py --config ${CONFIG_FILE}
# 5. ingest ground truth ratings
# PYTHONPATH=${PYTHON_PATH} python src/scripts_groundtruth/ingest_gt.py --config ${CONFIG_FILE}

PYTHONPATH=${PYTHON_PATH} python src/scripts/run_all_benchmarks.py --config ${CONFIG_FILE}
