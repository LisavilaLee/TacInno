# TacInno-VLA on Insert Hole (released checkpoint sim/insert_hole).
# Initialized from init/tacinno-vla_univtac-insert-hole_init.py.
_base_ = ["../_base_/models/tacinno_vla_pi05.py", "../_base_/datasets/univtac_6task.py",
          "../_base_/schedules/finetune_17k.py", "../_base_/policy_runtime.py"]
model = dict(proprio_history=0)
data = dict(tasks=["insert_hole"])
init_weights = dict(type="checkpoint", path="work_dirs/tacinno-vla_univtac-insert-hole_init/16999")
