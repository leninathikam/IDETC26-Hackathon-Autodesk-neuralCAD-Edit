import os.path as osp

from src.utils.args import parse_args
from src.utils.process_config import load_config
from src.utils.db import DatabaseManager
from src.scripts_benchmark_inference.filter_parquet_text_only import (
    filter_parquet_text_only,
)


def main():
    args = parse_args()
    config = load_config(args.config)

    dbm = DatabaseManager(config)

    request_count = len(list(dbm.requests.find()))
    text_count = len(list(dbm.requests.find({"modality": "text"})))

    if request_count != text_count:
        dbm.prune_to_modalities(keep_modalities=["text"], delete_files=False)
    else:
        print(f"Database already has {text_count} text-only requests; skipping DB prune.")

    dbm.compact_breps_to_referenced()
    dbm.cleanup_orphan_files()
    dbm.print_db_schema_counts()
    dbm.close_connection()

    # Also produce a text-only request parquet for the harness. The shipped
    # val_edit_all.parquet contains all modalities; filtering by populated
    # request_text reproduces the same subset as the DB prune above.
    parquet_dir = osp.join(config["storage_dir"]["path"], "parquets")
    input_parquet = osp.join(parquet_dir, "val_edit_all.parquet")
    output_parquet = osp.join(parquet_dir, "val_edit_text.parquet")

    if osp.exists(input_parquet):
        filter_parquet_text_only(input_parquet, output_parquet)
    else:
        print(f"Skipping parquet filtering; input parquet not found: {input_parquet}")


if __name__ == "__main__":
    main()
