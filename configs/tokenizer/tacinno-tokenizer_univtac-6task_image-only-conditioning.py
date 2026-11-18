# Ablation "image-only conditioning": encoder and decoder see the camera images but not the executed actions.
_base_ = ['./tacinno-tokenizer_univtac-6task.py']
model = dict(context='image')
