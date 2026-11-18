"""TacInno tokenizer: labels every demonstration frame t with a discrete tactile innovation code.

The code is defined by what it must explain. A forward-dynamics decoder has to predict the future tactile change
``X_{t+5k} - X_t`` (k = 1, 2, 3) from the current tactile observation ``X_t``, the camera images and the actions that
are executed until t+15, plus the code. Because images and actions are given to both the encoder and the decoder,
the 4 x 4-bit code only has to carry the part of the tactile change that images and actions cannot explain (the
tactile innovation). This property has no direct supervision; it follows from the bottleneck and is checked by the
code-shuffle test (reconstruction error with codes permuted within a task, ``val_gap``).

Inputs per anchor frame t (window frames t + offsets, default offsets (0, 5, 10, 15)):
  fields   (B, F, 2, 3, 96, 128)  tactile observation of both gels at each window frame (see tactile_fields)
  vis      (B, F, 2, 16, 768)     frozen DINOv2-base features of head and wrist cameras at each window frame,
                                  4 x 4 average-pooled patch tokens
  actions  (B, 15, 8)             executed relative joint targets joint[t+i, :8] - joint[t, :8], i = 1..15
                                  (7 arm joints + gripper), standardised per (i, joint)

Encoder (inverse dynamics, :class:`InnovationEncoder`): tokens per window frame f
  96 tactile patch tokens (2 gels x 6 x 8 patches of 16 x 16; shared conv patch embedding + patch-position and gel
     embeddings), 32 image tokens (LayerNorm + Linear 768 -> 384 + per-token position embedding), 5 action tokens
     (frame t+5 gets actions t+1..t+5, t+10 gets t+6..t+10, t+15 gets t+11..t+15; MLP 8 -> 384 -> 384 + slot
     embedding; frame t has 5 learned empty-action tokens) and the same 4 learned query tokens. 6 spatio-temporal
     blocks (width 384, 6 heads; spatial attention within a frame, causal temporal attention across frames at each
     token position, MLP). No temporal position embedding: frame order comes from the causal mask. The 4 query outputs
     of the last frame -> LayerNorm -> Linear 384 -> 32 = z (B, 4, 32).
Quantizer (:class:`PerSlotVectorQuantizer`): one 16 x 32 codebook per slot -> 4 codes in {0..15} (16 bit).
Decoder (forward dynamics, :class:`ForwardDynamicsDecoder`): frame t holds the 96 patch tokens of the known X_t;
  frames t+5, t+10, t+15 hold 96 "to-reconstruct" tokens (one learned vector + patch position + gel embedding);
  every frame also holds its image tokens, its action segment and the 4 code vectors (Linear 32 -> 384 + slot
  embedding). 6 spatio-temporal blocks; the to-reconstruct tokens -> LayerNorm -> Linear 384 -> 16 x 16 x 3 ->
  predicted X_{t+5k} - X_t. The decoder never receives a tactile observation after t.
Loss: contact-weighted L1 of the predicted change (weight 1 + 4 on pixels in contact at t or t+5k, the three horizons
  weighted equally) + codebook loss + 0.25 x commitment loss.

``context`` selects which conditioning both encoder and decoder receive: ``'both'`` (TacInno), ``'image'``
(image-only conditioning), ``'action'`` (actions only), ``'none'`` (w/o conditioning: the code then describes the
full future tactile change). ``offsets=(0, 15)`` gives the two-frame code ablation (one future frame that receives
all 15 action tokens). All variants keep every parameter so checkpoints share one layout.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from tacinno.registry import MODELS

from .layers import SpatioTemporalBlock, unpatchify
from .losses import reconstruction_loss
from .quantizer import PerSlotVectorQuantizer

__all__ = ["InnovationEncoder", "ForwardDynamicsDecoder", "TacInnoTokenizer", "CONTEXTS"]

CONTEXTS = ("both", "image", "action", "none")


class _Geometry:
    """Token counts derived from the config (kept in one place for encoder and decoder)."""

    def __init__(self, offsets: Sequence[int], field_size=(96, 128), patch_size=16, num_gels=2, num_views=2,
                 vis_tokens_per_view=16, num_slots=4):
        self.offsets = tuple(int(o) for o in offsets)
        assert self.offsets[0] == 0 and len(self.offsets) >= 2, offsets
        self.num_frames = len(self.offsets)
        self.num_future = self.num_frames - 1
        self.horizon = self.offsets[-1]
        assert self.horizon % self.num_future == 0, (offsets, 'actions must split evenly over the future frames')
        self.grid_hw = (field_size[0] // patch_size, field_size[1] // patch_size)
        self.n_patch = self.grid_hw[0] * self.grid_hw[1]          # patches per gel (48)
        self.n_tac = num_gels * self.n_patch                      # tactile tokens per frame (96)
        self.n_vis = num_views * vis_tokens_per_view              # image tokens per frame (32)
        self.n_act = self.horizon // self.num_future              # action tokens per future frame (5)
        self.n_q = num_slots


class InnovationEncoder(nn.Module):
    """Inverse-dynamics encoder: (fields, vis, actions) of a window -> continuous code vectors z (B, slots, code_dim)."""

    def __init__(self, geo: _Geometry, embed_dim=384, num_heads=6, depth=6, code_dim=32, context='both',
                 field_channels=3, patch_size=16, num_gels=2, vis_dim=768, action_dim=8, field_size=(96, 128)):
        super().__init__()
        D = embed_dim
        self.geo, self.context, self.D = geo, context, D
        self.num_gels, self.field_channels, self.field_size = num_gels, field_channels, field_size
        self.vis_dim, self.action_dim = vis_dim, action_dim
        self.tac_proj = nn.Conv2d(field_channels, D, patch_size, patch_size)
        self.tac_pos = nn.Parameter(torch.randn(geo.n_patch, D) * 0.02)
        self.tac_gel = nn.Parameter(torch.randn(num_gels, D) * 0.02)
        self.vis_norm, self.vis_proj = nn.LayerNorm(vis_dim), nn.Linear(vis_dim, D)
        self.vis_pos = nn.Parameter(torch.randn(geo.n_vis, D) * 0.02)
        self.cmd_mlp = nn.Sequential(nn.Linear(action_dim, D), nn.GELU(), nn.Linear(D, D))
        self.cmd_pos = nn.Parameter(torch.randn(geo.n_act, D) * 0.02)
        self.cmd_empty = nn.Parameter(torch.randn(geo.n_act, D) * 0.02)
        self.queries = nn.Parameter(torch.randn(geo.n_q, D) * 0.02)
        self.blocks = nn.ModuleList([SpatioTemporalBlock(D, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(D)
        self.to_code = nn.Linear(D, code_dim)

    def tokens(self, fields, vis, actions):
        """fields (B, F, gels, C, H, W); vis (B, F, views, 16, vis_dim); actions (B, horizon, action_dim)
        -> (B, F, tokens, D)."""
        g, D = self.geo, self.D
        B, Fr = fields.shape[:2]
        tac = self.tac_proj(fields.reshape(B * Fr * self.num_gels, self.field_channels, *self.field_size))
        tac = tac.flatten(2).transpose(1, 2)                                                    # (B*F*gels, 48, D)
        tac = tac.reshape(B, Fr, self.num_gels, g.n_patch, D) + self.tac_pos + self.tac_gel[:, None]
        tac = tac.reshape(B, Fr, g.n_tac, D)
        if self.context == 'none':
            return torch.cat([tac, self.queries.expand(B, Fr, g.n_q, D)], 2)
        q = self.queries.expand(B, Fr, g.n_q, D)
        if self.context == 'action':
            c = self.cmd_mlp(actions.reshape(B, g.num_future, g.n_act, self.action_dim)) + self.cmd_pos
            return torch.cat([tac, torch.cat([self.cmd_empty.expand(B, 1, g.n_act, D), c], 1), q], 2)
        v = self.vis_proj(self.vis_norm(vis.reshape(B, Fr, g.n_vis, self.vis_dim).float())) + self.vis_pos
        if self.context == 'image':
            return torch.cat([tac, v, q], 2)
        c = self.cmd_mlp(actions.reshape(B, g.num_future, g.n_act, self.action_dim)) + self.cmd_pos
        c = torch.cat([self.cmd_empty.expand(B, 1, g.n_act, D), c], 1)                          # frame t: empty
        return torch.cat([tac, v, c, q], 2)

    def forward(self, fields, vis, actions):
        x = self.tokens(fields, vis, actions)
        for blk in self.blocks:
            x = blk(x)
        return self.to_code(self.norm(x[:, -1, -self.geo.n_q:]))                                # (B, slots, code_dim)


class ForwardDynamicsDecoder(nn.Module):
    """Forward-dynamics decoder: (X_t, code vectors, vis, actions) -> predicted X_{t+offset} - X_t for every future
    window frame, (B, F - 1, gels, C, H, W). Its only tactile input is X_t."""

    def __init__(self, geo: _Geometry, embed_dim=384, num_heads=6, depth=6, code_dim=32, context='both',
                 field_channels=3, patch_size=16, num_gels=2, vis_dim=768, action_dim=8, field_size=(96, 128)):
        super().__init__()
        D = embed_dim
        self.geo, self.context, self.D = geo, context, D
        self.num_gels, self.field_channels, self.field_size = num_gels, field_channels, field_size
        self.patch_size, self.vis_dim, self.action_dim = patch_size, vis_dim, action_dim
        self.tac_proj = nn.Conv2d(field_channels, D, patch_size, patch_size)
        self.tac_pos = nn.Parameter(torch.randn(geo.n_patch, D) * 0.02)
        self.tac_gel = nn.Parameter(torch.randn(num_gels, D) * 0.02)
        self.slot_tok = nn.Parameter(torch.randn(D) * 0.02)
        self.vis_norm, self.vis_proj = nn.LayerNorm(vis_dim), nn.Linear(vis_dim, D)
        self.vis_pos = nn.Parameter(torch.randn(geo.n_vis, D) * 0.02)
        self.cmd_mlp = nn.Sequential(nn.Linear(action_dim, D), nn.GELU(), nn.Linear(D, D))
        self.cmd_pos = nn.Parameter(torch.randn(geo.n_act, D) * 0.02)
        self.cmd_empty = nn.Parameter(torch.randn(geo.n_act, D) * 0.02)
        self.code_in = nn.Linear(code_dim, D)
        self.code_slot = nn.Parameter(torch.randn(geo.n_q, D) * 0.02)
        self.blocks = nn.ModuleList([SpatioTemporalBlock(D, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, patch_size * patch_size * field_channels)

    def grid(self):
        return (self.tac_pos[None] + self.tac_gel[:, None]).reshape(self.geo.n_tac, self.D)

    def tokens(self, x_t, zq, vis, actions):
        """x_t (B, gels, C, H, W); zq (B, slots, code_dim); vis (B, F, views, 16, vis_dim); actions (B, horizon, 8)."""
        g, D = self.geo, self.D
        B = x_t.shape[0]
        tac0 = self.tac_proj(x_t.reshape(B * self.num_gels, self.field_channels, *self.field_size))
        tac0 = tac0.flatten(2).transpose(1, 2).reshape(B, g.n_tac, D) + self.grid()
        slots = (self.slot_tok + self.grid())[None, None].expand(B, g.num_future, g.n_tac, D)
        tac = torch.cat([tac0[:, None], slots], 1)                                              # (B, F, 96, D)
        q = (self.code_in(zq) + self.code_slot)[:, None].expand(B, g.num_frames, g.n_q, D)
        if self.context == 'none':
            return torch.cat([tac, q], 2)
        if self.context == 'action':
            c = self.cmd_mlp(actions.reshape(B, g.num_future, g.n_act, self.action_dim)) + self.cmd_pos
            return torch.cat([tac, torch.cat([self.cmd_empty.expand(B, 1, g.n_act, D), c], 1), q], 2)
        v = self.vis_proj(self.vis_norm(vis.reshape(B, g.num_frames, g.n_vis, self.vis_dim).float())) + self.vis_pos
        if self.context == 'image':
            return torch.cat([tac, v, q], 2)
        c = self.cmd_mlp(actions.reshape(B, g.num_future, g.n_act, self.action_dim)) + self.cmd_pos
        c = torch.cat([self.cmd_empty.expand(B, 1, g.n_act, D), c], 1)
        return torch.cat([tac, v, c, q], 2)

    def forward(self, x_t, zq, vis, actions):
        g = self.geo
        x = self.tokens(x_t, zq, vis, actions)
        for blk in self.blocks:
            x = blk(x)
        h = self.head(self.norm(x[:, 1:, :g.n_tac]))                                            # (B, F-1, 96, 768)
        B = h.shape[0]
        out = unpatchify(h.reshape(B * g.num_future, g.n_tac, h.shape[-1]), self.num_gels, self.field_channels,
                         g.grid_hw, self.patch_size)
        return out.reshape(B, g.num_future, self.num_gels, self.field_channels, *self.field_size)


@MODELS.register_module()
class TacInnoTokenizer(nn.Module):
    """Encoder + per-slot vector quantizer + forward-dynamics decoder (see the module docstring).

    The submodule names ``enc`` / ``vq`` / ``dec`` and every parameter name match the released checkpoints, which load
    with ``strict=True``.
    """

    def __init__(self, offsets: Sequence[int] = (0, 5, 10, 15), context: str = 'both', embed_dim: int = 384,
                 num_heads: int = 6, enc_depth: int = 6, dec_depth: int = 6, num_slots: int = 4,
                 codebook_size: int = 16, code_dim: int = 32, field_size: Tuple[int, int] = (96, 128),
                 patch_size: int = 16, field_channels: int = 3, num_gels: int = 2, num_views: int = 2,
                 vis_tokens_per_view: int = 16, vis_dim: int = 768, action_dim: int = 8,
                 target: str = 'cumulative', contact_weight: float = 4.0, commit_weight: float = 0.25,
                 amp: bool = True):
        super().__init__()
        if context not in CONTEXTS:
            raise ValueError(f"context must be one of {CONTEXTS}, got {context!r}")
        geo = _Geometry(offsets, tuple(field_size), patch_size, num_gels, num_views, vis_tokens_per_view, num_slots)
        kw = dict(embed_dim=embed_dim, num_heads=num_heads, code_dim=code_dim, context=context,
                  field_channels=field_channels, patch_size=patch_size, num_gels=num_gels, vis_dim=vis_dim,
                  action_dim=action_dim, field_size=tuple(field_size))
        self.geo, self.context, self.target = geo, context, target
        self.contact_weight, self.commit_weight, self.amp = contact_weight, commit_weight, amp
        self.enc, self.vq = InnovationEncoder(geo, depth=enc_depth, **kw), PerSlotVectorQuantizer(num_slots, codebook_size, code_dim)
        self.dec = ForwardDynamicsDecoder(geo, depth=dec_depth, **kw)

    # ------------------------------------------------------------------ properties
    @property
    def offsets(self) -> Tuple[int, ...]:
        return self.geo.offsets

    @property
    def horizon(self) -> int:
        return self.geo.horizon

    @property
    def uses_images(self) -> bool:
        return self.context in ('both', 'image')

    @property
    def uses_actions(self) -> bool:
        return self.context in ('both', 'action')

    def autocast(self, device: torch.device):
        """bf16 autocast context used for the encoder and decoder (enabled on CUDA when ``amp``)."""
        device = torch.device(device)
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=self.amp and device.type == 'cuda')

    # ------------------------------------------------------------------ inference
    def encode(self, fields: torch.Tensor, vis: Optional[torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        """-> continuous code vectors z (B, slots, code_dim), fp32 (encoder runs in bf16 autocast on CUDA)."""
        with self.autocast(fields.device):
            z = self.enc(fields, vis, actions)
        return z.float()

    def quantize(self, z: torch.Tensor):
        """z (B, slots, code_dim) fp32 -> (z_q, codes (B, slots), codebook loss, commitment loss)."""
        return self.vq(z)

    def decode(self, x_t, zq, vis, actions) -> torch.Tensor:
        """-> predicted tactile change for every future window frame (B, F - 1, gels, C, H, W), fp32."""
        with self.autocast(x_t.device):
            pred = self.dec(x_t, zq, vis, actions)
        return pred.float()

    @torch.no_grad()
    def predict_codes(self, fields, vis, actions) -> torch.Tensor:
        """Tactile innovation code of each window: (B, slots) int64 in [0, codebook_size)."""
        return self.quantize(self.encode(fields, vis, actions))[1]

    # ------------------------------------------------------------------ training
    def loss_reconstruction(self, pred: torch.Tensor, fields: torch.Tensor) -> torch.Tensor:
        """Per-sample, per-horizon contact-weighted L1 (B, F - 1)."""
        return reconstruction_loss(pred, fields, self.target, self.contact_weight)

    def forward_train(self, fields, vis, actions, decoder_vis=None, decoder_actions=None) -> Dict[str, torch.Tensor]:
        """One training forward pass. Initialises the codebook from the first batch and counts code usage.

        Returns dict(loss, rec, cb, commit, z (detached), codes)."""
        z = self.encode(fields, vis, actions)
        if not bool(self.vq.cb_initialised):
            self.vq.init_from(z.detach())
        zq, codes, l_cb, l_commit = self.quantize(z)
        self.vq.count(codes)
        pred = self.decode(fields[:, 0], zq, vis if decoder_vis is None else decoder_vis,
                           actions if decoder_actions is None else decoder_actions)
        l_rec = self.loss_reconstruction(pred, fields).mean()
        loss = l_rec + l_cb + self.commit_weight * l_commit
        return dict(loss=loss, rec=l_rec, cb=l_cb, commit=l_commit, z=z.detach(), codes=codes)

    @torch.no_grad()
    def forward_eval(self, fields, vis, actions, perm: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Code-shuffle test on one batch: per-horizon reconstruction loss with the true codes (``true``), with codes
        permuted by ``perm`` (``shuffled``, normally within a task) and of predicting no change (``copy``)."""
        z = self.encode(fields, vis, actions)
        zq, codes, _, _ = self.quantize(z)
        pred = self.decode(fields[:, 0], zq, vis, actions)
        pred_s = self.decode(fields[:, 0], zq[perm], vis, actions)
        return dict(codes=codes, true=self.loss_reconstruction(pred, fields),
                    shuffled=self.loss_reconstruction(pred_s, fields),
                    copy=self.loss_reconstruction(torch.zeros_like(pred), fields))
