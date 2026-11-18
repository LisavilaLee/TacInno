# Joint six-task stage without proprio tokens (initialization of the Insert Hole recipe).
_base_ = ["../_base_/models/tacinno_vla_pi05.py", "../_base_/datasets/univtac_6task.py",
          "../_base_/schedules/joint_17k.py", "../_base_/policy_runtime.py"]
model = dict(proprio_history=0, code_loss_weight=0.11862334646115223, contact_loss_weight=0.46727895124740054)
