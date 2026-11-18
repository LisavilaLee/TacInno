# Insert Hole (released checkpoint insert_hole/ema_16999): continue the joint model without proprio tokens on
# Insert Hole demonstrations only, 17000 steps at a constant 5e-6, weight EMA 0.99.
_base_ = ["../_base_/models/tacinno_vla_pi05.py", "../_base_/datasets/univtac_6task.py",
          "../_base_/schedules/finetune_17k.py", "../_base_/policy_runtime.py"]
model = dict(proprio_history=0)
data = dict(tasks=["insert_hole"])
init_weights = dict(type="checkpoint", path="work_dirs/tacinno-vla_pi05_joint-17k-noproprio_univtac-6task/16999")
