"""TacInno-VLA: pi0.5 with tactile condition tokens, trained with the TacInno code loss.

Structure (sizes of the released models)
----------------------------------------
* Prefix (PaliGemma, Gemma 2B, width 2048, 18 layers): the head and wrist camera images (224 x 224, SigLIP So400m/14,
  256 tokens each; Put Bottle uses the head camera only) and the tokenized language prompt. Bidirectional attention
  inside the prefix; the prefix does not attend to the expert sequence.
* State: the 120-D unified state vector (7 arm joints at 9:16, gripper at 44) is Fourier-encoded, averaged and,
  concatenated with the denoising-time embedding, conditions every action-expert layer through adaRMSNorm.
* Action expert (Gemma 300M, width 1024, 18 layers): ``[proprio x H | tactile x 60 | code queries x 4 | actions x 32]``,
  see :mod:`tacinno.models.policy.condition_tokens` for the condition tokens and their attention mask.
* Output: 32-step chunk of the 120-D action (arm joints relative to the joints at inference time, gripper absolute),
  z-score normalized per task; trained by flow matching (velocity MSE), sampled with 10 Euler steps.

Loss: L = L_flow + contact_loss_weight * L_contact + code_loss_weight * L_code. The code loss is the only signal that
asks the model to predict the tactile innovation code (the part of the future tactile change that the images and the
executed actions cannot explain); the code queries are not read by the action tokens, so the code reaches the actions
only through shared parameters (prefix, tactile encoder, expert). Whether the policy has learned to use tactile input
is tested at deployment by the tactile-off test (replace every tactile frame with a no-contact frame).

The model class is the pi0.5 implementation of the vendored FTP-1 / openpi code (``third_party/openpi``) with the
TacInno hooks; parameter names are those of the released checkpoints.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

from openpi.models_pytorch.ftp1_model_config import FTP1_ACTION_HORIZON, FTP1ModelConfig
from openpi.models_pytorch.ftp1_pytorch import FTP1Pytorch

from tacinno.registry import MODELS

__all__ = ["TacInnoVLA", "build_model_config", "model_config_to_dict"]

# 120-D unified FTP-1 action / state layout: 2 x (9-D wrist pose + 7 arm joints + 32 hand joints) + 9-D head + 15 reserved
ACTION_DIM = 2 * (9 + 7 + 32) + 9 + 15

_EXPERT_UPDATE_NAMES = {"proprio": "prop", "prop": "prop", "tactile": "tac", "tac": "tac", "code": "code"}


def build_model_config(
    *,
    paligemma_variant: str = "gemma_2b",
    action_expert_variant: str = "gemma_300m",
    action_horizon: int = FTP1_ACTION_HORIZON,
    max_token_len: int = 150,
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    proprio_history: int = 0,
    tactile_history: Sequence[int] = (8, 4),
    tactile_tokens_per_frame: int = 10,
    use_tactile: bool = True,
    use_code_queries: bool = True,
    num_code_slots: int = 4,
    code_head: bool = True,
    code_to_action: bool = False,
    expert_updates: Sequence[str] = ("tactile", "code"),
    gate_init: float = 0.1,
    code_loss_weight: float = 0.119,
    contact_loss_weight: float = 0.467,
    contact_head: bool = True,
    assets_dir: str | None = None,
) -> FTP1ModelConfig:
    """Translate the TacInno model settings into the vendored pi0.5 model config.

    Args:
        proprio_history: H proprio tokens (joints of frames t-H+1 .. t); 0 disables them.
        tactile_history: frame offsets of the earlier tactile observations (t-8, t-4); () = frame t only.
        tactile_tokens_per_frame: tokens per gel and frame (1 summary, 9 pooled patches, 10 both, 48 all patches).
        use_code_queries / num_code_slots / code_head: 4 learned code queries and the code head (code loss).
        code_to_action: let the other expert tokens attend to the code queries (ablation; False in TacInno-VLA).
        expert_updates: condition groups updated by the expert layers ("proprio", "tactile", "code").
        gate_init: initial value of the per-group gate scalar g (tokens are scaled by tanh(g)).
        code_loss_weight / contact_loss_weight: lambda_code and lambda_contact (0 removes the loss, keeps the head).
    """
    if num_code_slots != 4:
        raise ValueError("the TacInno tokenizer has 4 code slots")
    active = []
    for g in expert_updates:
        if g not in _EXPERT_UPDATE_NAMES:
            raise ValueError(f"expert_updates entries must be proprio / tactile / code, got {g!r}")
        active.append(_EXPERT_UPDATE_NAMES[g])
    return FTP1ModelConfig(
        dtype=dtype,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        attn_implementation=attn_implementation,
        action_dim=ACTION_DIM,
        action_horizon=int(action_horizon),
        max_token_len=int(max_token_len),
        discrete_state_input=False,
        state_input_mode="adarms",
        use_tactile_input=False,  # the FTP-1 tactile branch is off; tactile input enters as condition tokens
        disable_history=True,
        tactile_input_config_file="",
        assets_dir=assets_dir,
        cond_proprio_hist=int(proprio_history),
        cond_tactile=bool(use_tactile),
        cond_code=bool(use_code_queries),
        cond_gate_init=float(gate_init),
        cond_code_zero=True,
        cond_code_per_slot=False,
        cond_tac_aux=bool(contact_head and use_tactile),
        cond_tac_aux_weight=float(contact_loss_weight),
        cond_tac_hist=",".join(str(int(o)) for o in tactile_history) if use_tactile else "",
        cond_active=",".join(active),
        cond_code_to_action=bool(code_to_action),
        cond_tac_tokens=int(tactile_tokens_per_frame),
        cond_code_head=bool(code_head and use_code_queries),
        cond_code_ce_weight=float(code_loss_weight),
    )


def model_config_to_dict(config: FTP1ModelConfig) -> dict[str, Any]:
    """JSON-serializable view of the vendored model config (the format of ``model_config.json``)."""
    d = dataclasses.asdict(config)
    for k in ("paligemma_variant", "action_expert_variant", "tactile_expert_variant"):
        d[k] = str(d[k])
    return d


@MODELS.register_module()
class TacInnoVLA(FTP1Pytorch):
    """TacInno-VLA policy (see the module docstring).

    Build from TacInno settings, ``MODELS.build(dict(type="TacInnoVLA", proprio_history=0, ...))``, or from a vendored
    model config, ``TacInnoVLA(model_config=cfg)``. ``forward(observation, actions)`` returns
    ``{"loss", "flow_loss", "aux_loss", "code_ce", ...}``; ``sample_actions(device, observation)`` returns the action
    chunk (B, 32, 120) in normalized units.
    """

    def __init__(self, model_config: FTP1ModelConfig | Mapping[str, Any] | None = None, **settings: Any):
        if model_config is None:
            model_config = build_model_config(**settings)
        elif isinstance(model_config, Mapping):
            model_config = FTP1ModelConfig(**dict(model_config))
        elif settings:
            raise ValueError("pass either model_config or TacInno settings, not both")
        super().__init__(model_config)
