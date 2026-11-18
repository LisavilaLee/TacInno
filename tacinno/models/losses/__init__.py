"""Losses of TacInno-VLA.

* Flow-matching action loss (velocity MSE between ``noise - actions`` and the predicted velocity, masked by the action
  mask) is computed inside the pi0.5 model forward (``third_party/openpi/.../ftp1_pytorch.py``).
* :func:`code_ce_loss` (code loss) and :func:`contact_state_loss` (L_contact) are defined with the heads they train in
  :mod:`tacinno.models.policy.condition_tokens` and re-exported here.
"""
from tacinno.models.policy.condition_tokens import code_ce_loss, contact_state_loss, tactile_aux_loss

__all__ = ["code_ce_loss", "contact_state_loss", "tactile_aux_loss"]
