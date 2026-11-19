# TacInno-VLA on Pull-out Key (released checkpoint sim/pull_out_key).
_base_ = ["../_base_/models/tacinno_vla_pi05.py", "../_base_/datasets/univtac_6task.py",
          "../_base_/schedules/single_task_20k.py", "../_base_/policy_runtime.py"]
model = dict(proprio_history=0)
data = dict(tasks=["pull_out_key"])
