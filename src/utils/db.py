# from pymongo import MongoClient, ASCENDING
from mongita import MongitaClientDisk as MongoClient
import os
import os.path as osp
import shutil

class DatabaseManager:
    def __init__(self, config, use_aws=False, aws_uri=None):
        """
        use_aws: If True, connect to AWS DocumentDB using aws_uri.
        aws_uri: The connection string for AWS DocumentDB.
        local_uri: The connection string for local MongoDB.
        """

        self.config = config
        self.root_dir = config["storage_dir"]["path"]

        if use_aws:
            if not aws_uri:
                raise ValueError("aws_uri must be provided when use_aws=True")
            self.client = MongoClient(aws_uri, tls=True, tlsAllowInvalidCertificates=True)
        else:
            db_path = osp.join(config["storage_dir"]["path"], "mongita_db")
            os.makedirs(db_path, exist_ok=True)
            self.client = MongoClient(host=db_path)

        self.db = self.client[config["db_name"]]
        # Collections
        self.users = self.db['users']
        breps_collection = config.get("breps_collection", "breps")
        self.breps = self.db[breps_collection]
        self.requests = self.db['requests']
        self.edits = self.db['edits']
        self.ratings = self.db['ratings']

        self.make_dirs()

    def strip_root_dir(self, path):
        """
        Strip the root directory from the given path.
        """
        if path.startswith(self.root_dir):
            path = path[len(self.root_dir):]
        if path.startswith(os.sep):
            path = path[1:]
        return path

    def make_dirs(self):
        self.frames_dir = "frames"
        os.makedirs(osp.join(self.root_dir, self.frames_dir), exist_ok=True)
        self.brep_dir = "breps"
        os.makedirs(osp.join(self.root_dir, self.brep_dir), exist_ok=True)
        self.model_edit_dir = "model_edits"
        os.makedirs(osp.join(self.root_dir, self.model_edit_dir), exist_ok=True)
        self.model_rate_dir = "model_ratings"
        os.makedirs(osp.join(self.root_dir, self.model_rate_dir), exist_ok=True)


    # USERS
    def user_exists(self, user_id):
        return self.users.count_documents({"_id": user_id}) > 0

    def insert_user(self, user_id, email=None, vlm_config=None, is_human=True):
        if not self.user_exists(user_id):
            user_doc = {
                "_id": user_id,
                "email": email,
                "vlm_config": vlm_config,
                "is_human": is_human
            }
            id = self.users.insert_one(user_doc)
            print("User inserted successfully!")
            return id.inserted_id
        else:
            print("User already exists!")
            return None


    # BREPS
    def get_brep_id_str(self, user, end_time):
        return f"{user}_{end_time}"

    def brep_exists(self, user, end_time):
        brep_id = self.get_brep_id_str(user, end_time)
        return self.breps.count_documents({"_id": brep_id}) > 0

    def insert_brep(self, user, orig_path, end_time):
        brep_id = self.get_brep_id_str(user, end_time)
        if not self.brep_exists(user, end_time):
            id = self.breps.insert_one({
                "_id": brep_id,
                "user": user,
                "orig-path": "",
                "end_time": end_time
            })
            print("Brep inserted successfully!")
        else:
            print("Brep already exists!")


        # check if path exists
        if os.path.exists(orig_path):

            extensions = ['stp', 'obj', 'png', 'jpg', 'stl', 'step']

            # check if orig_path is a directory
            if os.path.isdir(orig_path):
                files = os.listdir(orig_path)
                files = [osp.join(orig_path, f) for f in files if f.endswith(tuple(extensions))]
            else:
                files = [orig_path]

            ext_files = {ext: [] for ext in extensions}
            for f in files:
                ext = osp.splitext(f)[-1][1:]
                if ext in ext_files:
                    ext_files[ext].append(f)
            # remove empty lists
            ext_files = {k: v for k, v in ext_files.items() if v}

            for ext, ext_list in ext_files.items():
                insert_list = []
                for f in ext_list:
                    if ext in ['png', 'jpg']:
                        view = osp.basename(f)
                        view = osp.splitext(view)[0]
                        view = view.split('_')[-1]
                        view_str = f"_{view}"
                    else:
                        view_str = ""

                    cp_target = osp.join(self.root_dir, self.brep_dir, f"{brep_id}{view_str}.{ext}")
                    if not osp.exists(f):
                        print(f"File {f} does not exist.")
                    else:
                        insert_list.append(self.strip_root_dir(cp_target))
                        if not osp.exists(cp_target):
                            shutil.copy(f, cp_target)  
                self.breps.update_one({"_id": brep_id}, {"$set": {ext: insert_list}})
            
        return brep_id
    
    def get_brep_images(self, brep_id, views=["toprightiso"], format=["jpg", "png"]):
        """
        Get the path to the frame image for a given brep_id.
        Assumes the frame image is stored as a PNG file in the breps directory.
        """
        brep = self.breps.find_one({"_id": brep_id})
        if not brep:
            print(f"Brep {brep_id} not found.")
            return None
        
        if isinstance(format, str):
            all_formats = [format]
        else:
            all_formats = format

        all_formats = [f for f in all_formats if f in brep]

        if not all_formats:
            print(f"No images found for format {format} in brep {brep_id}.")
            return []
        
        all_format_image_list = []

        for format in all_formats:
            image_list = brep[format]
            image_list = [img for img in image_list if any(img.endswith(f"{v}.{format}" ) for v in views)]

            if image_list:
                all_format_image_list.extend(image_list) 
                break

        return all_format_image_list


    # REQUESTS
    def get_request_id_str(self, user, difficulty, end_time):
        return f"{user}_{difficulty}_{end_time}"

    def request_exists(self, request_id):
        return self.requests.count_documents({
            "_id": request_id,
        }) > 0

    def insert_request(self, request_id, user, difficulty=None, brep_start=None, start_time=None, end_time=None, text=None, events=[], frames_dir=None, filename=None, request_type=None, prompt=None):
        if not self.request_exists(request_id):
            self.requests.insert_one({
                "_id": request_id,
                "user": user,
                "difficulty": difficulty,
                "brep_start": brep_start,
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "events": events,
                "frames_dir": self.strip_root_dir(frames_dir) if frames_dir else None,
                "filename": filename,
                "request_type": request_type,
                "prompt": prompt
            })

            print("Request inserted successfully!")
            return request_id
        else:
            print("Request already exists!")

    # EDITS
    def get_edit_id_str(self, user_id, end_time):
        return f"{user_id}_{end_time}"

    def edit_exists(self, edit_id):
        return self.edits.count_documents({"_id": edit_id}) > 0

    def insert_edit(self, edit_id, request_id, brep_end_id, user_id, start_time=None, end_time=None, events=[], frames_dir=None, filename=None, token_counts=None, completion=None, prompt_completion=None, failed_run=False):
        # edit_id = self.get_edit_id_str(user_id, end_time)
        if not self.edit_exists(edit_id):
            id = self.edits.insert_one({
                "_id": edit_id,
                "request": request_id,
                "brep_end": brep_end_id,
                "user": user_id,
                "start_time": start_time,
                "end_time": end_time,
                "events": events,
                "frames_dir": self.strip_root_dir(frames_dir),
                "filename": filename,
                "token_counts": token_counts,
                "completion": completion,
                "prompt_completion": prompt_completion,
                "failed_run": failed_run
            })
            print("Edit inserted successfully!")
            return id.inserted_id
        else:
            print("Edit already exists!")

    # RATINGS
    def rating_exists(self, user, edit):
        return self.ratings.count_documents({"user": user, "edit": edit}) > 0

    def insert_rating(self, user, edit, **kwargs):
        if not self.rating_exists(user, edit):
            rating_doc = {
                "user": user,
                "edit": edit,
            }
            rating_doc.update(kwargs)
            id = self.ratings.insert_one(rating_doc)
            print("Rating inserted successfully!")
            return id.inserted_id
        else:
            print("Rating already exists!")

    # OTHER UTILS - CLEANING, PRINTING etc.
    def _referenced_brep_ids(self):
        needed_ids = set()
        for request in self.requests.find():
            if request.get("brep_start"):
                needed_ids.add(request["brep_start"])
        for edit in self.edits.find():
            if edit.get("brep_end"):
                needed_ids.add(edit["brep_end"])
        return needed_ids

    def compact_breps_to_referenced(self):
        """Keep only brep documents referenced by requests/edits (preserves feature fields)."""
        needed_ids = self._referenced_brep_ids()
        kept_docs = []
        missing = []
        for brep_id in sorted(needed_ids):
            doc = self.breps.find_one({"_id": brep_id})
            if doc:
                kept_docs.append(doc)
            else:
                missing.append(brep_id)
        if missing:
            print(f"Warning: {len(missing)} referenced breps missing from collection")
        mongita_path = osp.join(self.root_dir, "mongita_db")
        breps_path = osp.join(mongita_path, "v2_db.breps")
        self.close_connection()
        if osp.isdir(breps_path):
            shutil.rmtree(breps_path)
            print(f"Removed breps collection directory for compaction: {breps_path}")
        db_path = osp.join(self.root_dir, "mongita_db")
        self.client = MongoClient(host=db_path)
        self.db = self.client[self.config["db_name"]]
        self.users = self.db['users']
        breps_collection = self.config.get("breps_collection", "breps")
        self.breps = self.db[breps_collection]
        self.requests = self.db['requests']
        self.edits = self.db['edits']
        self.ratings = self.db['ratings']
        for doc in kept_docs:
            self.breps.insert_one(doc)
        print(f"Compacted breps collection to {len(kept_docs)} referenced documents")
        return len(kept_docs)

    def drop_legacy_collection(self, collection_name):
        """Drop a mongita collection by name if it exists."""
        if collection_name not in self.db.list_collection_names():
            print(f"Collection {collection_name} not found; skipping drop.")
            return
        self.db.drop_collection(collection_name)
        print(f"Dropped collection: {collection_name}")

    def remove_brep_extension_files(self, extension):
        """Delete on-disk files for an extension without rewriting brep documents."""
        removed_files = 0
        brep_dir = osp.join(self.root_dir, self.brep_dir)
        if osp.isdir(brep_dir):
            for fname in os.listdir(brep_dir):
                if fname.endswith(f".{extension}"):
                    abs_path = osp.join(brep_dir, fname)
                    if osp.isfile(abs_path):
                        os.remove(abs_path)
                        removed_files += 1
        print(f"Deleted {removed_files} .{extension} files from breps/")

    def get_latest_edit_ids(self, request_id_list):
        latest_edit_dict = {}
        for edit in self.db.edits.find({"request": {"$in": request_id_list}}):
            req_id = edit["request"]
            user_id = edit["user"]
            key = (req_id, user_id)
            if key not in latest_edit_dict or float(edit["end_time"]) > float(latest_edit_dict[key]["end_time"]):
                latest_edit_dict[key] = edit
        edit_id_list = [edit["_id"] for edit in latest_edit_dict.values()]
        return edit_id_list
    
    def clean_db_single_edit_per_user_per_request(self):
        request_id_list = [request["_id"] for request in self.requests.find()]
        edit_id_list = self.get_latest_edit_ids(request_id_list)

        all_edits_iterator = self.edits.find({"_id": {"$nin": edit_id_list}})
        for edit in all_edits_iterator:
            self.breps.delete_many({"_id": edit["brep_end"]})
            self.ratings.delete_many({"edit": edit["_id"]})
            self.edits.delete_one({"_id": edit["_id"]})

    def print_db(self):
        """Print all collections and their contents."""
        for collection_name in self.db.list_collection_names():
            collection = self.db[collection_name]
            print(f"\nCollection: {collection_name}")
            docs = list(collection.find())
            if docs:
                for doc in docs:
                    print(doc)
            else:
                print("(empty)")

    def print_db_summary(self, count_limits={}):
        # only print _id of each document in each collection
        for collection_name in self.db.list_collection_names():

            if count_limits:
                if collection_name in count_limits:
                    count_limit = count_limits[collection_name]
                else:
                    continue
            else:
                count_limit = 999999

            full_fields = [
                "_id",
                "user",
                "request",
                "edit",
                "filename"
            ]

            collection = self.db[collection_name]
            print(f"\nCollection: {collection_name}")
            for doc in collection.find().limit(count_limit):
                # remove keys that are empty or None:
                doc = {k: v for k, v in doc.items() if v}

                # if the length of a value is > 30 characters, set to ''
                doc = {k: (v if (len(str(v)) < 30 or k in full_fields) else '...') for k, v in doc.items()}
                print(doc)

    def print_db_schema_counts(self):
        # print the number of documents in each collection
        for collection_name in self.db.list_collection_names():
            collection = self.db[collection_name]
            try:
                count = collection.count_documents({})
            except Exception as e:
                count = f"unavailable ({e})"
            print(f"Collection: {collection_name}, Count: {count}")

    def verify_db(self):
        # check all edits have a corresponding request
        for edit in self.edits.find():
            request = self.requests.find_one({"_id": edit["request"]})
            if not request:
                print(f"Edit {edit['_id']} does not have a corresponding request.")
        # check all requests have a corresponding user
        for request in self.requests.find():
            user = self.users.find_one({"_id": request["user"]})
            if not user:
                print(f"Request {request['_id']} does not have a corresponding user.")
        # check all breps have a corresponding user
        for brep in self.breps.find():
            user = self.users.find_one({"_id": brep["user"]})
            if not user:
                print(f"Brep {brep['_id']} does not have a corresponding user.")
        # check all ratings have a corresponding user
        for rating in self.ratings.find():
            user = self.users.find_one({"_id": rating["user"]})
            if not user:
                print(f"Rating {rating['_id']} does not have a corresponding user.")
        # check all ratings have a corresponding edit
        for rating in self.ratings.find():
            edit = self.edits.find_one({"_id": rating["edit"]})
            if not edit:
                print(f"Rating {rating['_id']} does not have a corresponding edit.")

    def prune_to_modalities(self, keep_modalities=None, delete_files=True):
        """
        Keep only requests whose modality is in keep_modalities and cascade-delete
        dependent records and optionally on-disk files.
        """
        if keep_modalities is None:
            keep_modalities = ["text"]

        print("Before pruning:")
        self.print_db_schema_counts()

        all_requests = list(self.requests.find())
        keep_ids = {
            request["_id"]
            for request in all_requests
            if request.get("modality") in keep_modalities
        }
        delete_ids = {request["_id"] for request in all_requests if request["_id"] not in keep_ids}

        print(f"Keeping {len(keep_ids)} requests with modalities {keep_modalities}")
        print(f"Deleting {len(delete_ids)} requests")

        deleted_brep_ids = set()
        deleted_frame_dirs = set()

        for request_id in delete_ids:
            edits = list(self.edits.find({"request": request_id}))
            for edit in edits:
                if edit.get("brep_end"):
                    deleted_brep_ids.add(edit["brep_end"])
                if edit.get("frames_dir"):
                    deleted_frame_dirs.add(edit["frames_dir"])
                self.ratings.delete_many({"edit": edit["_id"]})

            request = self.requests.find_one({"_id": request_id})
            if request and request.get("brep_start"):
                deleted_brep_ids.add(request["brep_start"])
            if request and request.get("frames_dir"):
                deleted_frame_dirs.add(request["frames_dir"])

            self.edits.delete_many({"request": request_id})
            self.requests.delete_one({"_id": request_id})

        referenced_brep_ids = set()
        for request in self.requests.find():
            if request.get("brep_start"):
                referenced_brep_ids.add(request["brep_start"])
        for edit in self.edits.find():
            if edit.get("brep_end"):
                referenced_brep_ids.add(edit["brep_end"])

        orphan_brep_ids = deleted_brep_ids - referenced_brep_ids
        for brep_id in orphan_brep_ids:
            brep = self.breps.find_one({"_id": brep_id})
            if brep and delete_files:
                self._delete_brep_files(brep)
            self.breps.delete_one({"_id": brep_id})

        referenced_frame_dirs = set()
        for request in self.requests.find():
            if request.get("frames_dir"):
                referenced_frame_dirs.add(request["frames_dir"])
        for edit in self.edits.find():
            if edit.get("frames_dir"):
                referenced_frame_dirs.add(edit["frames_dir"])

        orphan_frame_dirs = deleted_frame_dirs - referenced_frame_dirs
        if delete_files:
            videos_dir = osp.join(self.root_dir, "videos")
            if osp.isdir(videos_dir):
                shutil.rmtree(videos_dir)
                print(f"Removed directory: {videos_dir}")

            for frame_dir in orphan_frame_dirs:
                abs_frame_dir = osp.join(self.root_dir, frame_dir)
                if osp.isdir(abs_frame_dir):
                    shutil.rmtree(abs_frame_dir)
                    print(f"Removed directory: {abs_frame_dir}")

        print("After pruning:")
        self.print_db_schema_counts()

    def cleanup_orphan_files(self):
        """Remove videos and brep/frame files not referenced by the current database."""
        needed_brep_ids = set()
        needed_frame_dirs = set()

        for request in self.requests.find():
            if request.get("brep_start"):
                needed_brep_ids.add(request["brep_start"])
            if request.get("frames_dir"):
                needed_frame_dirs.add(request["frames_dir"])
        for edit in self.edits.find():
            if edit.get("brep_end"):
                needed_brep_ids.add(edit["brep_end"])
            if edit.get("frames_dir"):
                needed_frame_dirs.add(edit["frames_dir"])

        videos_dir = osp.join(self.root_dir, "videos")
        if osp.isdir(videos_dir):
            shutil.rmtree(videos_dir)
            print(f"Removed directory: {videos_dir}")

        brep_dir = osp.join(self.root_dir, self.brep_dir)
        removed_brep_files = 0
        if osp.isdir(brep_dir):
            for fname in os.listdir(brep_dir):
                matched = False
                for brep_id in needed_brep_ids:
                    if fname.startswith(brep_id):
                        remainder = fname[len(brep_id):]
                        if remainder == "" or remainder.startswith((".", "_")):
                            matched = True
                            break
                if not matched:
                    os.remove(osp.join(brep_dir, fname))
                    removed_brep_files += 1
        if removed_brep_files:
            print(f"Removed {removed_brep_files} orphan brep files")

        frames_root = osp.join(self.root_dir, self.frames_dir)
        if osp.isdir(frames_root):
            for entry in os.listdir(frames_root):
                rel_dir = osp.join(self.frames_dir, entry)
                if rel_dir not in needed_frame_dirs:
                    abs_dir = osp.join(frames_root, entry)
                    if osp.isdir(abs_dir):
                        shutil.rmtree(abs_dir)
                        print(f"Removed directory: {abs_dir}")

    def _delete_brep_files(self, brep):
        extensions = ["stp", "obj", "png", "jpg", "stl", "step"]
        removed = 0
        for ext in extensions:
            if ext not in brep:
                continue
            for rel_path in brep[ext]:
                abs_path = osp.join(self.root_dir, rel_path)
                if osp.isfile(abs_path):
                    os.remove(abs_path)
                    removed += 1
        if removed:
            print(f"Removed {removed} brep files for {brep['_id']}")

    def close_connection(self):
        if self.client is not None:
            self.client.close()
            self.client = None
