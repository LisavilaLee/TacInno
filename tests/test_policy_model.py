"""CPU tests of the TacInno-VLA model on a small backbone.

The backbone is the pi0.5 model of training at a small size: language model width 2048 (the SigLIP projection is fixed
to 2048) with 4 layers, action expert "dummy" (width 64, 4 layers), float32, eager attention; the SigLIP tower is the
full one. The condition tokens are those of the released recipe (tactile t-8, t-4, t x 10 tokens per gel, 4 code
queries, tactile and code tokens updated by the expert, code queries hidden from every other token).
"""
from __future__ import annotations

import dataclasses
import os
import tempfile

import pytest
import torch

torch.set_num_threads(min(16, os.cpu_count() or 1))

_gemma = pytest.importorskip("openpi.models.gemma")
from openpi.models.model import Observation  # noqa: E402

from tacinno.registry import MODELS  # noqa: E402

B, H = 2, 32
_orig_get_config = _gemma.get_config


def _test_get_config(variant):
    if variant == "test_vlm":
        return _gemma.Config(width=2048, depth=4, mlp_dim=256, num_heads=8, num_kv_heads=1, head_dim=16)
    return _orig_get_config(variant)


_gemma.get_config = _test_get_config
SMALL = dict(type="TacInnoVLA", paligemma_variant="test_vlm", action_expert_variant="dummy", dtype="float32",
             action_horizon=H)


def make(seed=0, **kw):
    import tacinno.models.policy.tacinno_vla  # noqa: F401  (registers TacInnoVLA)
    torch.manual_seed(seed)
    return MODELS.build({**SMALL, **kw}).eval()


def make_inputs(seed=1, proprio=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    cond = {"field": r(B, 2, 3, 96, 128), "field_hist": r(B, 2, 2, 3, 96, 128),
            "code_tgt": torch.randint(0, 16, (B, 4), generator=g), "code_tv": torch.tensor([1.0, 0.0]),
            "aux_m": r(B, 3, 2, 14), "aux_mv": torch.ones(B, 3, 2), "aux_ec": (r(B, 3, 2) > 0).float()}
    if proprio:
        cond["proprio"] = r(B, proprio, 8)
    mask = torch.ones(B, 20, dtype=torch.bool)
    mask[1, -5:] = False
    obs = Observation(images={"camera_ego_rgb_0": r(B, 224, 224, 3).clamp(-1, 1),
                              "right_wrist_camera_rgb_0": r(B, 224, 224, 3).clamp(-1, 1)},
                      image_masks={"camera_ego_rgb_0": torch.ones(B, dtype=torch.bool),
                                   "right_wrist_camera_rgb_0": torch.ones(B, dtype=torch.bool)},
                      state=r(B, 1, 120), tokenized_prompt=torch.randint(0, 1000, (B, 20), generator=g),
                      tokenized_prompt_mask=mask, cond=cond)
    return obs, r(B, H, 120), r(B, H, 120), torch.rand(B, generator=g) * 0.9 + 0.05


def _store(box):
    """forward hook keeping the first output of the expert's final norm (returns None: the output is unchanged)"""
    def hook(mod, inp, out):
        box.setdefault("final", out[0] if isinstance(out, tuple) else out)
    return hook


def deploy_obs(obs):
    return dataclasses.replace(obs, cond={k: v for k, v in obs.cond.items() if k in ("proprio", "field", "field_hist")})


def fwd(model, obs, actions, noise, time):
    torch.manual_seed(7)  # the training preprocessing augments images with the torch RNG
    return model(obs, actions, noise=noise, time=time)


def sample(model, obs, noise):
    with torch.no_grad():
        return model.sample_actions("cpu", deploy_obs(obs), noise=noise.clone(), num_steps=3)


@pytest.fixture(scope="module")
def model():
    return make()


@pytest.fixture(scope="module")
def batch():
    return make_inputs()


def test_token_layout(model):
    enc = model.cond_encoder
    assert enc.num_tactile_tokens() == 2 * 10 * 3
    assert enc.num_tokens() == 60 + 4
    assert enc.spans() == {"prop": (0, 0), "tac": (0, 60), "code": (60, 64)}
    m8 = make(proprio_history=8)
    assert m8.cond_encoder.spans() == {"prop": (0, 8), "tac": (8, 68), "code": (68, 72)}


def test_loss_composition(model, batch):
    obs, actions, noise, time = batch
    with torch.no_grad():
        out = fwd(model, obs, actions, noise, time)
    cfg = model.config
    expect = out["flow_loss"] + cfg.cond_tac_aux_weight * out["aux_loss"] + cfg.cond_code_ce_weight * out["code_ce"]
    assert torch.allclose(out["loss"], expect, rtol=1e-6, atol=1e-7)
    assert {"aux_mse", "aux_bce", "code_acc", "code_valid"} <= set(out)


def test_actions_do_not_read_code_queries_or_heads(model, batch):
    obs, _, noise, _ = batch
    a0 = sample(model, obs, noise)
    m = make()
    m.load_state_dict(model.state_dict())
    with torch.no_grad():  # change the code-query content and both auxiliary heads
        m.cond_encoder.code_slot.add_(1.0)
        m.cond_encoder.code_head_w.mul_(-3.0)
        for p in m.cond_encoder.aux_head.parameters():
            p.add_(0.5)
    a1 = sample(m, obs, noise)
    assert torch.equal(a0, a1)


def test_code_queries_change_actions_when_read(batch):
    obs, _, noise, _ = batch
    m0 = make(code_to_action=True)
    m1 = make(code_to_action=True)
    m1.load_state_dict(m0.state_dict())
    with torch.no_grad():
        m1.cond_encoder.code_slot.add_(1.0)
    assert not torch.equal(sample(m0, obs, noise), sample(m1, obs, noise))


def test_gradient_routes(batch):
    obs, actions, noise, time = batch
    m = make().train()

    from tacinno.models.policy.condition_tokens import code_ce_loss, tactile_aux_loss

    enc = m.cond_encoder
    # code loss: reaches the tactile encoder and the code head, not the action projections / time MLP / state encoder
    m.zero_grad(set_to_none=True)
    hook = {}
    h = m.paligemma_with_expert.gemma_expert.model.norm.register_forward_hook(_store(hook))
    torch.manual_seed(7)
    m(obs, actions, noise=noise, time=time)
    h.remove()
    c0, c1 = enc.code_span()
    ce, _ = code_ce_loss(enc, hook["final"][:, c0:c1], obs.cond)
    ce.backward()
    nz = lambda mod: any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())  # noqa: E731
    assert nz(enc.tac) and enc.code_head_w.grad is not None
    for mod in (m.action_in_proj, m.action_out_proj, m.time_mlp_in, m.time_mlp_out, m.state_encoder):
        assert not nz(mod)
    # contact-state loss: only the tactile encoder and its head
    m.zero_grad(set_to_none=True)
    torch.manual_seed(7)
    m(obs, actions, noise=noise, time=time)
    aux, _ = tactile_aux_loss(enc, obs.cond)
    aux.backward()
    assert nz(enc.tac) and nz(enc.aux_head)
    for mod in (m.action_in_proj, m.paligemma_with_expert, m.state_encoder):
        assert not nz(mod)


def test_condition_tokens_independent_of_noise_and_time(model, batch):
    obs, actions, noise, time = batch
    outs = []
    for k in range(2):
        cap = {}
        h = model.paligemma_with_expert.gemma_expert.model.norm.register_forward_hook(_store(cap))
        with torch.no_grad():
            fwd(model, obs, actions, noise * (k + 1), time * 0.5 if k else time)
        h.remove()
        outs.append(cap["final"][:, :model.cond_encoder.num_tokens()])
    assert torch.equal(outs[0], outs[1])


def test_checkpoint_round_trip(model, batch):
    import safetensors.torch

    obs, _, noise, _ = batch
    with tempfile.TemporaryDirectory() as d:
        safetensors.torch.save_model(model, os.path.join(d, "model.safetensors"))
        m = make(seed=123)
        safetensors.torch.load_model(m, os.path.join(d, "model.safetensors"))
    assert torch.equal(sample(model, obs, noise), sample(m, obs, noise))
