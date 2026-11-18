# Joint six-task stage with 8 proprio tokens (initialization of the Lift Can recipe).
# lambda_code / lambda_contact: calibrated so that each auxiliary loss is 15 % of L_flow over steps 100-199 of a
# 200-step run with both weights 1 (see docs/en/policy.md).
_base_ = ["../_base_/models/tacinno_vla_pi05.py", "../_base_/datasets/univtac_6task.py",
          "../_base_/schedules/joint_17k.py", "../_base_/policy_runtime.py"]
model = dict(proprio_history=8, code_loss_weight=0.11862334646115223, contact_loss_weight=0.46727895124740054)
