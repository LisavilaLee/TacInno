# Ablation: encoder and decoder see the executed actions but not the camera images.
_base_ = ['./tacinno-tokenizer_univtac-6task.py']
model = dict(context='action')
