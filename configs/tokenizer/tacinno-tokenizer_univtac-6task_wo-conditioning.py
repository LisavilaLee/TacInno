# Ablation "w/o conditioning": neither encoder nor decoder sees images or actions, so the code describes the
# full future tactile change instead of the tactile innovation.
_base_ = ['./tacinno-tokenizer_univtac-6task.py']
model = dict(context='none')
