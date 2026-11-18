# UniVTAC simulation data (see docs/en/data_preparation.md). Layout under data_root:
#   zarr_train/<task>/<task>_all.zarr     demonstrations 0-49 (pi0.5 zarr format, head + wrist cameras)
#   head_cam/zarr_train/<task>/...        head-camera-only copy (Put Bottle recipe)
#   cond/                                 condition arrays and targets (tools/data/make_cond.py, make_code_targets.py)
#   norm/<repo_id>/UniVTAC_<task>/        normalization statistics (tools/data/compute_norm_stats.py)
data = dict(
    data_root="data/univtac",
    zarr_dir="zarr_train",
    cond_dir="cond",
    norm_root="norm",
    repo_id="univtac_6task",
    domain_prefix="UniVTAC_",
    tasks=["insert_hole", "insert_tube", "lift_bottle", "lift_can", "pull_out_key", "put_bottle_in_shelf"],
    cond_keys=("field", "field_hist", "code_tgt", "aux_hist"),  # "proprio" is added when model.proprio_history > 0
    num_workers=12,
    check_norm_params_snapshot=False,
)
deploy = dict(cameras=("head", "wrist"), domain_prefix="UniVTAC_")
