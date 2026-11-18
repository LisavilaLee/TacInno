# TacInno tokenizer used in the paper (one tokenizer shared by the six UniVTAC tasks).
_base_ = [
    '../_base_/models/tacinno_tokenizer.py',
    '../_base_/datasets/univtac_tokenizer.py',
    '../_base_/schedules/tokenizer_12k.py',
]
