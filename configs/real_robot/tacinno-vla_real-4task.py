# TacInno-VLA on the four real-world tasks (released checkpoint real/four_tasks): Franka FR3 with one
# Xense gel at 10 Hz; the right-gel input is a copy of the left gel. lambda_code / lambda_contact
# calibrated on this data (15 % of L_flow each).
_base_ = ["../_base_/models/tacinno_vla_pi05.py", "../_base_/schedules/single_task_20k.py",
          "../_base_/policy_runtime.py"]
model = dict(proprio_history=0, code_loss_weight=0.11798233046789909, contact_loss_weight=0.40614703135347074)
data = dict(
    data_root="data/real_robot",
    zarr_dir="zarr_train",
    cond_dir="cond",
    norm_root="norm",
    repo_id="real_4task",
    domain_prefix="Real_",
    tasks=["real_wipe_board", "real_lift_can", "real_insert_plug", "real_unscrew_cap"],
    cond_keys=("field", "field_hist", "code_tgt", "aux_hist"),
    num_workers=12,
    check_norm_params_snapshot=False,
)
deploy = dict(cameras=("head", "wrist"), domain_prefix="Real_", control_hz=10)
