# Ablation "two-frame code": the window has only frames t and t+15 (same span); the future frame gets all
# 15 action tokens and the decoder predicts X_{t+15} - X_t only.
_base_ = ['./tacinno-tokenizer_univtac-6task.py']
model = dict(offsets=(0, 15))
