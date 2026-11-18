"""Condition tokens of TacInno-VLA and its two auxiliary training losses.

TacInno-VLA is pi0.5 (PaliGemma prefix + Gemma-300M action expert, width W = 1024, 18 layers) with extra *condition
tokens* placed in the action-expert sequence in front of the 32 noisy action tokens::

    expert sequence = [ proprio x H | tactile x 2*n*(F+1) | code queries x 4 | noisy actions x 32 ]

Inputs (per sample, built by :mod:`tacinno.datasets.condition_data` in training and :mod:`tacinno.apis.inference`
at deployment):

* ``proprio``    (H, 8) normalized joints (7 arm joints + gripper) of frames t-H+1 .. t. Each frame goes through
  Linear 8->W, SiLU, Linear W->W, plus a learned time embedding (H, W). H = 8 or 0 (recipe dependent).
* ``field_hist`` (F, 2, 3, 96, 128) and ``field`` (2, 3, 96, 128): the tactile observations of frames t-8, t-4 and t
  (F = 2 history frames). Channel 0 is the indentation field, channels 1-2 the marker tangential displacement field
  (x and y, divided by sigma_u), for the left and the right gel. Every frame and gel goes through the
  :class:`TactileFieldEncoder` (Conv 16x16 / stride 16 -> 6 x 8 = 48 patch tokens of width 384 + learned position and
  gel embeddings, one learned summary query, 2 pre-norm Transformer encoder layers with 6 heads, MLP 4x, full
  attention). Per gel and frame it outputs n = 10 tokens: the summary-query output and the 48 patch outputs
  average-pooled to 3 x 3, all through LayerNorm and Linear 384->W, plus a learned token-index embedding (10, W) and a
  learned frame embedding (F+1, W). With F = 2 this gives 60 tactile tokens.
* code queries: 4 learned query vectors (``code_proj.bias + code_slot``; the content input is zero). They carry no
  input information; they exist to be read out by the code head.

Every group is multiplied by tanh(g), g a learned scalar per group (initial value ``gate_init``).

Attention (built in the vendored pi0.5 model, see ``third_party/openpi/MODIFICATIONS.md``): condition tokens attend to
the prefix (images + prompt) and to each other, never to the action tokens; action tokens attend to the prefix, the
proprio and tactile tokens and all action tokens. With the code head enabled, no token other than the code queries
attends to the code queries (``code_to_action=False``), so the code information can reach the actions only through
shared parameters. The denoising-time adaRMS modulation acts only on the 32 action tokens. The tactile tokens and the
code queries are updated by every expert layer (:func:`open_residual`); the proprio tokens are not (each layer only
projects their input embedding to K/V).

Auxiliary losses (training only; the heads are not used for acting):

* code loss :func:`code_ce_loss`: the code queries' last-layer expert outputs -> LayerNorm(W) -> per-slot Linear W->16;
  cross-entropy against the TacInno tokenizer's code of frame t (4 slots, 16 entries each), mean over slots and valid
  frames (frames whose t+15 lies outside the episode have no target).
* contact-state loss :func:`tactile_aux_loss` (L_contact): per gel and tactile frame, the encoder's summary (384,
  before the gate) -> Linear 384->256, GELU, Linear 256->15 -> 14 standardized contact properties (MSE, masked by their
  validity) + one contact logit (BCE against the contact label).

Total loss: L = L_flow + lambda_contact * L_contact + lambda_code * L_code (``contact_loss_weight``, ``code_loss_weight``).
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["TactileFieldEncoder", "CondTokenEncoder", "tactile_aux_loss", "contact_state_loss", "code_ce_loss",
           "open_residual"]


class TactileFieldEncoder(nn.Module):
    """Encodes the tactile observation of both gels of one frame into condition tokens (see the module docstring)."""

    def __init__(self, width: int, d: int = 384, layers: int = 2, heads: int = 6, patch: int = 16,
                 hw: tuple[int, int] = (96, 128), tokens: int = 1):
        super().__init__()
        self.patch = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
        self.grid = (hw[0] // patch, hw[1] // patch)
        n = self.grid[0] * self.grid[1]
        if tokens not in (1, 9, 10, n):
            raise ValueError(f"tactile tokens per gel must be 1, 9, 10 or {n}, got {tokens}")
        self.tokens = int(tokens)
        if self.tokens > 1:  # learned embedding of the token index (summary + pooled patches)
            self.tok_pos = nn.Parameter(torch.randn(self.tokens, width) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.gel = nn.Parameter(torch.randn(2, 1, d) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, width)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """field (B, 2, 3, 96, 128) -> tokens (B, 2 * tokens, W), gel-major. Keeps the per-gel summaries (B, 2, 384)
        in ``self.last_summary`` for the contact-state head."""
        B = field.shape[0]
        x = self.patch(field.reshape(B * 2, 3, *field.shape[-2:]).float())       # (2B, d, 6, 8)
        x = x.flatten(2).transpose(1, 2) + self.pos                               # (2B, 48, d)
        x = x.reshape(B, 2, -1, x.shape[-1]) + self.gel[None]                     # (B, 2, 48, d)
        x = x.reshape(B * 2, -1, x.shape[-1])
        x = torch.cat([self.query.expand(B * 2, -1, -1), x], 1)
        h = self.enc(x)
        x = self.norm(h[:, 0])                                                    # (2B, d) summary per gel
        self.last_summary = x.reshape(B, 2, -1)
        if self.tokens == 1:
            return self.out(x).reshape(B, 2, -1)                                  # (B, 2, W)
        p = h[:, 1:]                                                              # (2B, 48, d) patch outputs
        if self.tokens in (9, 10):
            g = p.transpose(1, 2).reshape(B * 2, -1, *self.grid)                  # (2B, d, 6, 8)
            p = nn.functional.adaptive_avg_pool2d(g, (3, 3)).flatten(2).transpose(1, 2)  # (2B, 9, d)
        p = self.norm(p)
        if self.tokens == 10:
            p = torch.cat([x[:, None], p], 1)                                     # (2B, 10, d) summary first
        return (self.out(p) + self.tok_pos).reshape(B, 2 * self.tokens, -1)       # (B, 2n, W) gel-major


class CondTokenEncoder(nn.Module):
    """Builds all condition tokens (proprio, tactile, code queries) and holds the two auxiliary heads.

    Parameter / buffer names are those of the released checkpoints (``cond_encoder.*``); do not rename them.
    ``code_zero=True`` (TacInno-VLA) makes the code tokens pure learned queries; ``code_zero=False`` feeds externally
    predicted code probabilities through a frozen codebook (kept only for compatibility, not used by TacInno-VLA).
    """

    def __init__(self, width: int, proprio_hist: int = 0, use_tactile: bool = False, use_code: bool = False,
                 code_slots: int = 4, code_vocab: int = 16, code_dim: int = 32, gate_init: float = 0.0,
                 code_zero: bool = False, code_per_slot: bool = False, tac_aux: bool = False,
                 tac_hist: tuple[int, ...] = (), code_head: bool = False, tac_tokens: int = 1):
        super().__init__()
        self.proprio_hist, self.use_tactile, self.use_code = int(proprio_hist), bool(use_tactile), bool(use_code)
        self.code_zero = bool(code_zero)
        if self.proprio_hist > 0:
            self.prop_mlp = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width))
            self.prop_time = nn.Parameter(torch.randn(self.proprio_hist, width) * 0.02)
            self.g_prop = nn.Parameter(torch.full((), float(gate_init)))
        self.tac_hist = tuple(int(o) for o in tac_hist) if use_tactile else ()
        if self.use_tactile:
            self.tac = TactileFieldEncoder(width, tokens=tac_tokens)
            self.g_tac = nn.Parameter(torch.full((), float(gate_init)))
            if self.tac_hist:  # frame embedding of (t - o for o in tac_hist) and t, oldest first
                self.tac_time = nn.Parameter(torch.randn(len(self.tac_hist) + 1, 1, width) * 0.02)
        # contact-state head on the encoder summary (384, before the gate); its loss reaches only the tactile encoder
        # and this head
        self.tac_aux = bool(tac_aux) and self.use_tactile
        if self.tac_aux:
            self.aux_head = nn.Sequential(nn.Linear(384, 256), nn.GELU(), nn.Linear(256, 14 + 1))
        if self.use_code:
            shape = (code_slots, code_vocab, code_dim) if code_per_slot else (code_vocab, code_dim)
            self.register_buffer("codebook", torch.zeros(*shape))  # only read when code_zero is False
            self.code_proj = nn.Linear(code_dim, width)
            self.code_slot = nn.Parameter(torch.randn(code_slots, width) * 0.02)
            self.g_code = nn.Parameter(torch.full((), float(gate_init)))
        self.code_head = bool(code_head) and self.use_code
        if self.code_head:
            self.code_head_norm = nn.LayerNorm(width)
            self.code_head_w = nn.Parameter(torch.randn(code_slots, width, code_vocab) * width ** -0.5)
            self.code_head_b = nn.Parameter(torch.zeros(code_slots, code_vocab))

    @property
    def enabled(self) -> bool:
        return self.proprio_hist > 0 or self.use_tactile or self.use_code

    def num_tactile_tokens(self) -> int:
        return 2 * self.tac.tokens * (len(self.tac_hist) + 1) if self.use_tactile else 0

    def num_tokens(self) -> int:
        return self.proprio_hist + self.num_tactile_tokens() + (self.code_slot.shape[0] if self.use_code else 0)

    def spans(self) -> dict[str, tuple[int, int]]:
        """Positions [start, end) of each group among the condition tokens (= in the expert sequence, which starts
        with them); disabled groups have an empty span."""
        a = self.proprio_hist
        b = a + self.num_tactile_tokens()
        return {"prop": (0, a), "tac": (a, b), "code": (b, b + (self.code_slot.shape[0] if self.use_code else 0))}

    def code_span(self) -> tuple[int, int]:
        return self.spans()["code"]

    def code_logits(self, h: torch.Tensor) -> torch.Tensor:
        """h (B, 4, W) expert outputs at the code queries -> (B, 4, 16) logits, one linear map per slot."""
        return torch.einsum("bsw,swk->bsk", self.code_head_norm(h.float()), self.code_head_w) + self.code_head_b

    def gates(self) -> dict[str, float]:
        out = {}
        for k in ("g_prop", "g_tac", "g_code"):
            if hasattr(self, k):
                out[k] = float(torch.tanh(getattr(self, k).detach().float()))
        return out

    def forward(self, cond: dict) -> torch.Tensor:
        """cond dict (see the module docstring) -> condition tokens (B, num_tokens(), W)."""
        toks = []
        if self.proprio_hist > 0:
            p = cond["proprio"].float()
            if p.shape[1] != self.proprio_hist:
                raise ValueError(f"proprio history length {p.shape[1]} != {self.proprio_hist}")
            toks.append(torch.tanh(self.g_prop) * (self.prop_mlp(p) + self.prop_time[None]))
        if self.use_tactile:
            if self.tac_hist:
                f = torch.cat([cond["field_hist"], cond["field"][:, None]], 1)       # (B, F+1, 2, 3, 96, 128)
                B, F = f.shape[:2]
                t = self.tac(f.reshape(B * F, *f.shape[2:]))                          # (B (F+1), 2 n, W)
                t = t.reshape(B, F, t.shape[1], -1) + self.tac_time[None]
                self.tac.last_summary_all = self.tac.last_summary.reshape(B, F, 2, -1)  # frames oldest first
                self.tac.last_summary = self.tac.last_summary_all[:, -1]
                toks.append(torch.tanh(self.g_tac) * t.reshape(B, -1, t.shape[-1]))
            else:
                toks.append(torch.tanh(self.g_tac) * self.tac(cond["field"]))
        if self.use_code:
            if self.code_zero:
                B = toks[0].shape[0] if toks else next(iter(cond.values())).shape[0]
                emb = torch.zeros(B, self.code_slot.shape[0], self.codebook.shape[-1], device=self.code_slot.device)
            else:
                pr = cond["code_prob"].float()                                      # (B, 4, 16)
                if self.codebook.dim() == 3:
                    emb = torch.einsum("bik,ikd->bid", pr, self.codebook)          # (B, 4, 32) per-slot books
                else:
                    emb = pr @ self.codebook                                        # (B, 4, 32) shared book
            toks.append(torch.tanh(self.g_code) * (self.code_proj(emb) + self.code_slot[None]))
        return torch.cat(toks, 1)


def tactile_aux_loss(enc: CondTokenEncoder, cond: dict) -> tuple[torch.Tensor, dict]:
    """Contact-state loss L_contact: MSE on the 14 standardized contact properties (masked by their validity) + BCE on
    the contact label, both gels; with stacked targets (aux_m (B, F+1, 2, 14)) every tactile frame is scored against
    its own targets. Uses the summaries of the last forward of ``enc.tac``."""
    if cond["aux_m"].dim() == 4:
        pred = enc.aux_head(enc.tac.last_summary_all.float())                          # (B, F+1, 2, 15)
    else:
        pred = enc.aux_head(enc.tac.last_summary.float())                              # (B, 2, 15)
    m, mv, ec = cond["aux_m"].float(), cond["aux_mv"].float(), cond["aux_ec"].float()
    mse = (((pred[..., :14] - m) ** 2).mean(-1) * mv).sum() / mv.sum().clamp(min=1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred[..., 14], ec)
    return mse + bce, {"aux_mse": mse.detach(), "aux_bce": bce.detach()}


contact_state_loss = tactile_aux_loss  # paper name


def code_ce_loss(enc: CondTokenEncoder, h_code: torch.Tensor, cond: dict) -> tuple[torch.Tensor, dict]:
    """Code loss L_code: cross-entropy of the code head against the target code of frame t (code_tgt (B, 4) int,
    code_tv (B,) validity), mean over the 4 slots and the valid frames; also returns the top-1 accuracy."""
    logits = enc.code_logits(h_code)                                                    # (B, 4, 16)
    tgt, tv = cond["code_tgt"].long(), cond["code_tv"].float()
    ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tgt.clamp(min=0).flatten(), reduction="none")
    ce = (ce.view(tgt.shape).mean(-1) * tv).sum() / tv.sum().clamp(min=1)
    acc = ((logits.argmax(-1) == tgt).float().mean(-1) * tv).sum() / tv.sum().clamp(min=1)
    return ce, {"code_ce": ce.detach(), "code_acc": acc.detach(), "code_valid": tv.mean().detach()}


def _open_forward(self, x, cond=None, active_tail_tokens=None):
    normed, gate = type(self)._base_forward(self, x, cond, active_tail_tokens=active_tail_tokens)
    if gate is None or x.ndim != 3 or x.shape[1] < max(e for _, e in self.open_spans):
        return normed, gate
    pos = torch.arange(x.shape[1], device=x.device)[None, :, None]
    m = torch.zeros_like(pos, dtype=torch.bool)
    for s, e in self.open_spans:
        m = m | ((pos >= s) & (pos < e))
    return normed, torch.where(m, torch.ones_like(gate), gate)


_OPEN_CLASSES: dict[type, type] = {}


def open_residual(norms, spans: list[tuple[int, int]]) -> None:
    """Let the condition tokens at expert positions [s, e) be updated by the expert layers.

    The pi0.5 action expert masks its adaRMS modulation to the last ``action_horizon`` tokens, which returns modulation
    0 and residual gate 0 for every other position, so condition tokens would never change across layers. For the
    given spans this sets the residual gate to 1 (scale and shift stay 0, i.e. a plain RMSNorm), in every norm of
    ``norms`` (input and post-attention norms of the expert layers). Implemented by swapping the module class (same
    parameters and state-dict keys)."""
    for m in norms:
        base = type(m)
        if base not in _OPEN_CLASSES:
            _OPEN_CLASSES[base] = type("Open" + base.__name__, (base,), {"forward": _open_forward,
                                                                         "_base_forward": base.forward})
        m.__class__ = _OPEN_CLASSES[base]
        m.open_spans = [(int(s), int(e)) for s, e in spans if e > s]
