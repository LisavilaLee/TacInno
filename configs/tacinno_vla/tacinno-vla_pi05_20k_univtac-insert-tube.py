# Insert Tube (released checkpoint insert_tube/ema_19999): single-task training from the pi0.5 base weights, no proprio tokens,
# 20000 steps, warm-up 500 to 5e-5 then cosine to 5e-6, weight EMA 0.99.
_base_ = ["../_base_/models/tacinno_vla_pi05.py", "../_base_/datasets/univtac_6task.py",
          "../_base_/schedules/single_task_20k.py", "../_base_/policy_runtime.py"]
model = dict(proprio_history=0)
data = dict(tasks=["insert_tube"])
