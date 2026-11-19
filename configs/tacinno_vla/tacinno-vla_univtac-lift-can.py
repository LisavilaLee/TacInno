# TacInno-VLA on Lift Can (released checkpoint sim/lift_can).
# Initialized from init/tacinno-vla_univtac-lift-can_init.py.
_base_ = ["../_base_/models/tacinno_vla_pi05.py", "../_base_/datasets/univtac_6task.py",
          "../_base_/schedules/finetune_17k.py", "../_base_/policy_runtime.py"]
model = dict(proprio_history=8)
data = dict(tasks=["lift_can"])
init_weights = dict(type="checkpoint", path="work_dirs/tacinno-vla_univtac-lift-can_init/16999")
