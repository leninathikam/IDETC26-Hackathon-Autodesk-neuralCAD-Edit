from src.utils.args import parse_args
from src.utils.process_config import load_config
import os.path as osp
from src.utils.db import DatabaseManager
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist
from probreg import cpd
import copy


def load_stl_as_point_cloud(stl_path, num_samples=10000):
    """
    Load an STL file and convert it to a point cloud.
    
    Args:
        stl_path (str): Path to the STL file
        num_samples (int): Number of points to sample from the mesh surface
        
    Returns:
        numpy.ndarray: Point cloud as (N, 3) array
    """
    try:
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(str(stl_path))
        
        if len(mesh.vertices) == 0:
            raise ValueError(f"Failed to load mesh from {stl_path}")
        
        # print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        
        # Sample points uniformly from the mesh surface
        if len(mesh.triangles) > 0:
            point_cloud = mesh.sample_points_uniformly(number_of_points=num_samples)
            points = np.asarray(point_cloud.points)
        else:
            # If no triangles, use vertices directly
            points = np.asarray(mesh.vertices)
            if len(points) > num_samples:
                # Randomly sample if too many points
                indices = np.random.choice(len(points), num_samples, replace=False)
                points = points[indices]
        
        # print(f"Generated point cloud with {len(points)} points")
        return points
        
    except Exception as e:
        raise RuntimeError(f"Error loading STL file {stl_path}: {str(e)}")





def compute_chamfer_distance(points1, points2, normalize=False):
    """
    Compute chamfer distance between two point clouds.
    
    Chamfer distance = mean(min_dist(p1 -> p2)) + mean(min_dist(p2 -> p1))
    
    Args:
        points1 (numpy.ndarray): First point cloud (N1, 3)
        points2 (numpy.ndarray): Second point cloud (N2, 3)
        normalize (bool): Whether to normalize by the diagonal of bounding box
        
    Returns:
        tuple: (chamfer_distance, forward_distance, backward_distance)
    """
    # Compute pairwise distances
    distances_1_to_2 = cdist(points1, points2, metric='euclidean')
    distances_2_to_1 = cdist(points2, points1, metric='euclidean')
    
    # Find minimum distances
    min_distances_1_to_2 = np.min(distances_1_to_2, axis=1)
    min_distances_2_to_1 = np.min(distances_2_to_1, axis=1)
    
    # Compute forward and backward distances
    forward_distance = np.mean(min_distances_1_to_2)
    backward_distance = np.mean(min_distances_2_to_1)
    
    # Chamfer distance is the sum of both directions
    chamfer_dist = forward_distance + backward_distance
    
    if normalize:
        # Normalize by the diagonal of the combined bounding box
        all_points = np.vstack([points1, points2])
        bbox_min = np.min(all_points, axis=0)
        bbox_max = np.max(all_points, axis=0)
        bbox_diagonal = np.linalg.norm(bbox_max - bbox_min)
        
        chamfer_dist /= bbox_diagonal
        forward_distance /= bbox_diagonal
        backward_distance /= bbox_diagonal
    
    return chamfer_dist, forward_distance, backward_distance


def pair_cosine_similarity(f1, f2, db=None):
    return cosine_similarity(
        np.array(f1).reshape(1, -1),
        np.array(f2).reshape(1, -1)
    )[0][0]


def align_point_clouds(source_pc, target_pc, num_points=1000, num_initializations=8):
    initializations = [(0,0,0)]
    initializations.extend([np.random.uniform(0, 2*np.pi, size=3) for _ in range(num_initializations-1)])

    if isinstance(source_pc, np.ndarray):
        source_pc_o3d = o3d.geometry.PointCloud()
        source_pc_o3d.points = o3d.utility.Vector3dVector(source_pc)
    else:
        source_pc_o3d = copy.deepcopy(source_pc)

    if isinstance(target_pc, np.ndarray):
        target_pc_o3d = o3d.geometry.PointCloud()
        target_pc_o3d.points = o3d.utility.Vector3dVector(target_pc)
    else:
        target_pc_o3d = copy.deepcopy(target_pc)

    # downsample
    source_pc_ds = source_pc_o3d.random_down_sample(min(1.0, float(num_points) / len(source_pc_o3d.points)))
    target_pc_ds = target_pc_o3d.random_down_sample(min(1.0, float(num_points) / len(target_pc_o3d.points)))

    best_sigma2 = float('inf')
    best_tf_param = None
    best_R = None
    best_center = None

    for cur_initialization in initializations:
        # Random rotation about bbox center
        R = o3d.geometry.get_rotation_matrix_from_xyz(cur_initialization)
        bbox_center = source_pc_ds.get_axis_aligned_bounding_box().get_center()
        source_pc_rotated = copy.deepcopy(source_pc_ds).rotate(R, center=bbox_center)

        registration_result = cpd.registration_cpd(source_pc_rotated, target_pc_ds, tol=1e-10, maxiter=100)

        print(registration_result, registration_result.sigma2)
        tf_param, _, _ = registration_result
        # o3d.visualization.draw_geometries([target_pc, source_pc_rotated])

        if registration_result.sigma2 < best_sigma2:
            best_sigma2 = registration_result.sigma2
            best_tf_param = tf_param
            best_R = R
            best_center = bbox_center

    # Apply the best transformation to the original source point cloud
    # First rotate, then apply CPD transformation
    source_pc_o3d_rotated = copy.deepcopy(source_pc_o3d)
    source_pc_o3d_rotated.rotate(best_R, center=best_center)
    
    # Apply CPD transformation and convert back to numpy array
    transformed_points = best_tf_param.transform(np.asarray(source_pc_o3d_rotated.points))
    
    return transformed_points


def chamfer_similarity_norm(source_brep, target_brep, db, pre_align=False):
    """
    Computes a normalized Chamfer similarity between two BREP objects.

    The raw Chamfer distance is divided by the ground truth bounding box diagonal
    to produce a scale-invariant distance in [0, 1], then converted to similarity
    as 1 - normalized_distance.

    Args:
        source_brep (str): Path to the ground truth BREP STL file.
        target_brep (str): Path to the predicted BREP STL file.
        db (DatabaseManager): Database manager instance (provides root_dir).
        pre_align (bool): Whether to align the prediction to the GT before computing.

    Returns:
        float: Normalized similarity in [0, 1]. 1.0 = perfect match, 0.0 = worst
        (including empty or unloadable geometry).
    """
    if isinstance(source_brep, list):
        source_brep = source_brep[0]
    if isinstance(target_brep, list):
        target_brep = target_brep[0]

    pts_gt = load_stl_as_point_cloud(osp.join(db.root_dir, source_brep))
    pts_pred = load_stl_as_point_cloud(osp.join(db.root_dir, target_brep))

    if pts_gt.shape[0] == 0 or pts_pred.shape[0] == 0:
        return 0.0

    gt_bbox_diag = np.linalg.norm(pts_gt.max(axis=0) - pts_gt.min(axis=0))
    if gt_bbox_diag < 1e-8:
        return 0.0

    if pre_align:
        pts_pred = align_point_clouds(pts_pred, pts_gt)

    raw_dist = compute_chamfer_distance(pts_gt, pts_pred, normalize=False)[0]
    norm_dist = min(raw_dist / gt_bbox_diag, 1.0)
    return 1.0 - norm_dist


def align_meshes(mesh_source, mesh_target, num_points=1000, num_initializations=8):
    """
    Returns the transformed mesh_source aligned to mesh_target.
    """

    # convert meshes to point clouds
    source_pc = mesh_source.sample_points_uniformly(number_of_points=num_points)
    target_pc = mesh_target.sample_points_uniformly(number_of_points=num_points)

    initializations = [(0,0,0)]
    initializations.extend([np.random.uniform(0, 2*np.pi, size=3) for _ in range(num_initializations-1)])

    best_sigma2 = float('inf')
    best_tf_param = None
    best_R = None
    best_center = None

    for cur_initialization in initializations:
        # Random rotation about bbox center
        R = o3d.geometry.get_rotation_matrix_from_xyz(cur_initialization)
        bbox_center = source_pc.get_axis_aligned_bounding_box().get_center()
        source_pc_rotated = copy.deepcopy(source_pc).rotate(R, center=bbox_center)

        # Compute the transformation parameters using CPD
        registration_result = cpd.registration_cpd(source_pc_rotated, target_pc, tol=1e-10, maxiter=100)

        print(registration_result, registration_result.sigma2)
        tf_param, _, _ = registration_result
        # o3d.visualization.draw_geometries([target_pc, source_pc_rotated])

        if registration_result.sigma2 < best_sigma2:
            best_sigma2 = registration_result.sigma2
            best_tf_param = tf_param
            best_R = R
            best_center = bbox_center

    # Apply the best transformation to the source mesh
    transformed_mesh = copy.deepcopy(mesh_source)
    # transformed_vertices = best_tf_param.transform(np.asarray(mesh_source.vertices))

    # rotate transformed_mesh
    transformed_mesh.rotate(best_R, center=best_center)

    transformed_vertices = best_tf_param.transform(np.asarray(transformed_mesh.vertices))
    transformed_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    return transformed_mesh


def run_feature_gt_similarity_eval(config: dict, dbm: DatabaseManager, feature_key: str = "feature_dino", description: str = "dino similarity", distance_func=pair_cosine_similarity, request_type: str = "edit", distance_func_kwargs=None, force=False):
    all_requests_iterator = dbm.requests.find({"request_type": request_type})

    for request in all_requests_iterator:

        brep_start_id = request["brep_start"]
        gt_user = request["user"]
        brep_start = dbm.breps.find_one({"_id": brep_start_id})
        start_feature = brep_start.get(feature_key, None)

        gt_edit = dbm.edits.find_one({"request": request["_id"], "user": gt_user})
        gt_brep = dbm.breps.find_one({"_id": gt_edit["brep_end"]})
        gt_feature = gt_brep.get(feature_key, None)

        # get edits with the same request id and different user
        all_other_user_edits = dbm.edits.find({"request": request["_id"], "user": {"$ne": gt_user}})

        all_edits = dbm.edits.find({"request": request["_id"]})
        all_edits = list(all_edits)

        gt_sim_str = f"{description} gt"
        start_sim_str = f"{description} start"

        for edit in all_other_user_edits:
            valid_user = False
            user = dbm.users.find_one({"_id": edit["user"]})


            if user["_id"]  in config["benchmark_eval_users"][request_type]:
                valid_user = True
            if "other human" in config["benchmark_eval_users"][request_type] and user.get("is_human", True):
                valid_user = True

            if not valid_user:
                print(f"Skipping edit {edit['_id']} because user {edit['user']} is not in benchmark_eval_users")
                continue

            brep_end_id = edit["brep_end"]
            brep_end = dbm.breps.find_one({"_id": brep_end_id})

            if brep_end is None:
                print(f"Skipping edit {edit['_id']} because brep_end is None")
                continue

            # get the feature of the end brep
            edit_feature = brep_end.get(feature_key, None)


            if gt_feature is None or edit_feature is None:
                print(f"Skipping edit {edit['_id']} because features are missing")
                continue

            if dbm.rating_exists("similarity_eval", edit["_id"]):
                rating = dbm.ratings.find_one({"edit": edit["_id"], "user": "similarity_eval"})
                rating_id = rating["_id"]
                if not force and gt_sim_str in rating and start_sim_str in rating:
                    print(f"Skipping edit {edit['_id']} because ratings already exist")
                    continue
            else:
                rating_id = dbm.insert_rating(
                    user="similarity_eval",
                    edit=edit["_id"]
                )


            print(f"Computing {description} for edit {edit['_id']} with gt {gt_edit['_id']} and request {request['_id']}")

            gt_similarity = distance_func(gt_feature, edit_feature, db=dbm, **(distance_func_kwargs or {}))
            start_similarity = distance_func(start_feature, edit_feature, db=dbm, **(distance_func_kwargs or {})) if start_feature and edit_feature else 0.0

            dbm.ratings.update_one(
                {"_id": rating_id},
                {"$set": {
                    gt_sim_str: gt_similarity,
                    start_sim_str: start_similarity
                }}
            )


def main():
    # Parse command-line arguments
    args = parse_args()
    # Load configuration
    config = load_config(args.config)

    dbm = DatabaseManager(config)

    run_feature_gt_similarity_eval(config, dbm, feature_key="feature_dino", description="dino similarity", distance_func=pair_cosine_similarity, request_type="edit")

    run_feature_gt_similarity_eval(config, dbm, feature_key="stl", description="chamfer similarity norm", distance_func=chamfer_similarity_norm, request_type="edit")

if __name__ == "__main__":
    main()