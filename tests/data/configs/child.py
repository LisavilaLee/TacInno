_base_ = ["./_base_/base_model.py", "./_base_/base_sched.py"]
import math  # modules are not config values
model = dict(width=16, head=dict(_delete_=True, type="MLP", hidden=32), tags=["c"])
_private = 1
