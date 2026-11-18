"""Models of TacInno.

* :mod:`tacinno.models.tokenizer`: the TacInno tokenizer (code model) that turns the tactile innovation of a 4-frame
  window into a 16-bit discrete code. Imported eagerly (it only needs PyTorch).
* :mod:`tacinno.models.policy`: TacInno-VLA on the pi0.5 backbone. Its members are imported on first use because they
  need the vendored openpi package; call :func:`tacinno.utils.register_all_modules` to register them.
* :mod:`tacinno.models.losses`: the code loss and the contact-state loss of TacInno-VLA.
"""
from . import policy, tokenizer
from .tokenizer import TacInnoTokenizer

__all__ = ["policy", "tokenizer", "TacInnoTokenizer"]
