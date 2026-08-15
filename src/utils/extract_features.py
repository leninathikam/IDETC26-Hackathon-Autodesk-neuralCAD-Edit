from src.utils.args import parse_args
from src.utils.process_config import load_config
import os.path as osp
from src.utils.db import DatabaseManager
from transformers import pipeline


def extract_dino(db: DatabaseManager, config: dict, feature_info: list[dict]) -> None:

    # get infos where the brep doesn't already have a dino feature
    print("\nExtracting DINO features for breps...")

    to_extract = []
    for info in feature_info:
        if "feature_dino" not in db.breps.find_one({"_id": info["brep_id"]}) and info["frame_path"] is not None:
            to_extract.append(info)

    if not to_extract:
        print("No breps to extract features for.")
        return

    # extract dino features
    extractor = pipeline(
        task="image-feature-extraction",
        model=config["dino_model"],
    )

    input_paths = [e["frame_path"] for e in to_extract]

    features = extractor(
        input_paths,
        batch_size=config["dino_batch_size"],
        return_tensors=True
    )

    # loop over all features and insert them into the database
    for i, info in enumerate(to_extract):
        feature = features[i]
        feature = feature.squeeze().numpy()
        feature = feature.flatten().tolist()  # flatten the feature to a list
        db.breps.update_one(
            {"_id": info["brep_id"]},
            {"$set": {"feature_dino": feature}}
        )


def extract_all_features(db: DatabaseManager, config: dict) -> None:

    # go through all requests and extract features from the breps
    all_requests_iterator = db.requests.find()

    feature_info = []

    for request in all_requests_iterator:
        brep_id = request["brep_start"]
        if not brep_id:
            continue
        frame_path = db.get_brep_images(brep_id)
        frame_path = osp.join(db.root_dir, frame_path[0]) if frame_path else None
        info = {"request_or_edit": "request", "id": request["_id"], "frame_path": frame_path, "brep_id": brep_id}
        feature_info.append(info)

    for edit in db.edits.find():
        brep_id = edit["brep_end"]
        if not brep_id:
            continue
        frame_path = db.get_brep_images(brep_id)
        frame_path = osp.join(db.root_dir, frame_path[0]) if frame_path else None

        info = {"request_or_edit": "edit", "id": edit["_id"], "frame_path": frame_path, "brep_id": brep_id}
        feature_info.append(info)

    extract_dino(db, config, feature_info)


def main():
    # Parse command-line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    db = DatabaseManager(config)

    extract_all_features(db, config)

    db.print_db_summary()

    db.close_connection()

if __name__ == "__main__":
    main()
