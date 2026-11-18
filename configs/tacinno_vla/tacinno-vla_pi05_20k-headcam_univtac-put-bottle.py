# Put Bottle (released checkpoint put_bottle_in_shelf/ema_19999): as the single-task recipe, but trained and deployed
# with the head camera only (the benchmark's camera setting for this task).
_base_ = ["../_base_/models/tacinno_vla_pi05.py", "../_base_/datasets/univtac_6task.py",
          "../_base_/schedules/single_task_20k.py", "../_base_/policy_runtime.py"]
model = dict(proprio_history=0)
data = dict(tasks=["put_bottle_in_shelf"], zarr_dir="head_cam/zarr_train")
deploy = dict(cameras=("head",))
