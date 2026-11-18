"""CPU unit tests of the TacInno tokenizer (tiny shapes)."""
import json
import os.path as osp

import numpy as np
import pytest
import torch

from tacinno.config import Config
from tacinno.datasets import tactile_fields as TF
from tacinno.datasets.tokenizer_dataset import DEFAULT_STATS_FILE, UniVTACTokenizerData, parse_episodes
from tacinno.models.tokenizer import (PerSlotVectorQuantizer, SpatioTemporalBlock, TacInnoTokenizer,
                                      reconstruction_loss, unpatchify)
from tacinno.models.tokenizer.training import cosine_lambda, param_groups, perplexity, within_group_perm
from tacinno.registry import MODELS

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
TINY = dict(embed_dim=32, num_heads=2, enc_depth=1, dec_depth=1, field_size=(32, 48), patch_size=16,
            vis_tokens_per_view=2, vis_dim=8, amp=False)


def tiny_inputs(model, B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    Fr, (H, W) = len(model.offsets), model.enc.field_size
    fields = torch.randn(B, Fr, 2, 3, H, W, generator=g)
    vis = torch.randn(B, Fr, 2, 2, 8, generator=g)
    actions = torch.randn(B, model.horizon, 8, generator=g)
    return fields, vis, actions


@pytest.mark.parametrize('context', ['both', 'image', 'action', 'none'])
@pytest.mark.parametrize('offsets', [(0, 5, 10, 15), (0, 15)])
def test_forward_shapes(context, offsets):
    torch.manual_seed(0)
    model = TacInnoTokenizer(offsets=offsets, context=context, **TINY)
    fields, vis, actions = tiny_inputs(model, B=16)  # the codebook is initialised from >= 16 encoder outputs
    z = model.encode(fields, vis, actions)
    assert z.shape == (16, 4, 32)
    zq, codes, l_cb, l_commit = model.quantize(z)
    assert codes.shape == (16, 4) and codes.dtype == torch.long and int(codes.max()) < 16
    pred = model.decode(fields[:, 0], zq, vis, actions)
    assert pred.shape == (16, len(offsets) - 1, 2, 3, 32, 48)
    out = model.forward_train(fields, vis, actions)
    assert torch.isfinite(out['loss']) and bool(model.vq.cb_initialised)
    out['loss'].backward()
    assert model.enc.to_code.weight.grad is not None  # straight-through gradient reaches the encoder


def test_context_ignores_unused_inputs():
    torch.manual_seed(0)
    model = TacInnoTokenizer(context='none', **TINY).eval()
    fields, vis, actions = tiny_inputs(model)
    z1 = model.encode(fields, vis, actions)
    z2 = model.encode(fields, vis * 0 + 5, actions * 0 - 3)
    assert torch.equal(z1, z2)
    model = TacInnoTokenizer(context='image', **TINY).eval()
    assert torch.equal(model.encode(fields, vis, actions), model.encode(fields, vis, actions + 1))
    model = TacInnoTokenizer(context='action', **TINY).eval()
    assert torch.equal(model.encode(fields, vis, actions), model.encode(fields, vis + 1, actions))


def test_temporal_attention_is_causal():
    torch.manual_seed(0)
    blk = SpatioTemporalBlock(dim=16, num_heads=2).eval()
    x = torch.randn(2, 4, 5, 16)
    x2 = x.clone()
    x2[:, 2:] += torch.randn_like(x2[:, 2:])  # change frames 2 and 3
    y, y2 = blk(x), blk(x2)
    assert torch.allclose(y[:, :2], y2[:, :2], atol=1e-6)  # frames 0 and 1 do not see later frames
    assert not torch.allclose(y[:, 2:], y2[:, 2:])


def test_decoder_sees_no_future_tactile():
    torch.manual_seed(0)
    model = TacInnoTokenizer(**TINY).eval()
    fields, vis, actions = tiny_inputs(model)
    zq = model.quantize(model.encode(fields, vis, actions))[0]
    pred = model.decode(fields[:, 0], zq, vis, actions)
    fields2 = fields.clone()
    fields2[:, 1:] = 0
    assert torch.equal(pred, model.decode(fields2[:, 0], zq, vis, actions))


def test_vq_nearest_and_reset():
    torch.manual_seed(0)
    vq = PerSlotVectorQuantizer(num_slots=3, codebook_size=5, code_dim=4)
    z = torch.randn(7, 3, 4, requires_grad=True)
    zq, idx, l_cb, l_commit = vq(z)
    brute = torch.cdist(z.detach().transpose(0, 1), vq.codebook.detach()).argmin(-1).transpose(0, 1)
    assert torch.equal(idx, brute)
    assert torch.allclose(zq, vq.lookup(idx))
    zq.sum().backward()
    assert torch.allclose(z.grad, torch.ones_like(z))  # straight-through
    vq.count(idx)
    n_dead = int((vq.usage == 0).sum())
    assert vq.reset_dead(z.detach()) == n_dead and float(vq.usage.sum()) == 0


def test_unpatchify_inverts_patch_order():
    x = torch.randn(2, 2, 3, 32, 48)
    # token order (gel, row, col), per-token order (channel, patch_row, patch_col)
    p = x.reshape(2, 2, 3, 2, 16, 3, 16).permute(0, 1, 3, 5, 2, 4, 6).reshape(2, 2 * 2 * 3, 3 * 16 * 16)
    assert torch.equal(unpatchify(p, 2, 3, (2, 3), 16), x)


def test_reconstruction_loss():
    fields = torch.zeros(2, 4, 2, 3, 8, 8)
    fields[:, 1:, :, 1] = 1.0                       # constant change of 1 in channel 1 at every future frame
    pred = torch.zeros(2, 3, 2, 3, 8, 8)
    loss = reconstruction_loss(pred, fields)        # |0 - 1| on one of three channels everywhere -> 1/3
    assert loss.shape == (2, 3) and torch.allclose(loss, torch.full((2, 3), 1 / 3))
    fields[:, :, :, 0, :4] = 1.0                    # contact (indentation 1 > threshold) on half the rows, no change
    loss_w = reconstruction_loss(pred, fields)      # the contact weights do not change a uniform error
    assert torch.allclose(loss_w, torch.full((2, 3), 1 / 3))
    pred[:, :, :, 1, :4] = 1.0                      # exact on the contact half: error only on the weight-1 half
    assert torch.allclose(reconstruction_loss(pred, fields), torch.full((2, 3), (4 * 8 * 1) / (3 * (4 * 8 * 5 + 4 * 8))))


def test_param_groups_and_schedule():
    model = TacInnoTokenizer(**TINY)
    dec, nodec = param_groups(model, 0.05)
    names = {id(p): n for n, p in model.named_parameters()}
    assert all(not names[id(p)].endswith(('pos', 'gel', 'queries', 'slot')) for p in dec['params'])
    assert any(names[id(p)] == 'vq.codebook' for p in nodec['params'])
    f = cosine_lambda(100, 10)
    assert abs(f(0) - 0.1) < 1e-9 and f(9) > 0.95 and abs(f(100) - 0.1) < 1e-9


def test_code_statistics():
    codes = torch.tensor([[0, 1], [0, 2], [1, 3], [1, 0]])
    pp, used, per_slot = perplexity(codes, n_codes=4)
    assert used == 4 and abs(per_slot[0] - 2.0) < 1e-5 and abs(per_slot[1] - 4.0) < 1e-5
    groups = torch.tensor([0, 0, 1, 1, 1])
    perm = within_group_perm(groups, torch.Generator().manual_seed(0))
    assert torch.equal(groups[perm], groups) and sorted(perm.tolist()) == list(range(5))


def test_configs_build_and_checkpoint_layout():
    names = ['', '_wo-conditioning', '_image-only-conditioning', '_action-only-conditioning', '_two-frame-code']
    for n in names:
        cfg = Config.fromfile(osp.join(ROOT, 'configs', 'tokenizer', f'tacinno-tokenizer_univtac-6task{n}.py'))
        model = MODELS.build(cfg.model)
        keys = set(model.state_dict())
        # parameter names of the released checkpoints
        for k in ('enc.tac_proj.weight', 'enc.blocks.5.t_attn.qkv.weight', 'enc.cmd_empty', 'enc.to_code.bias',
                  'vq.codebook', 'vq.usage', 'vq.cb_initialised', 'dec.slot_tok', 'dec.code_slot', 'dec.head.weight'):
            assert k in keys, (n, k)
        assert model.enc.cmd_pos.shape == ((15, 384) if n == '_two-frame-code' else (5, 384))
    cfg = Config.fromfile(osp.join(ROOT, 'configs', 'tokenizer', 'tacinno-tokenizer_univtac-6task.py'))
    assert sum(p.numel() for p in MODELS.build(cfg.model).parameters()) > 20_000_000


def test_parse_episodes():
    assert parse_episodes('0-2,5') == [0, 1, 2, 5] and parse_episodes(3) == [3] and parse_episodes([4, 6]) == [4, 6]


def _write_episode(root, feat_root, task, ep, T, stats, rng):
    rest = np.stack([np.asarray(stats['rest_native'][g], np.float32) for g in range(2)])  # (2, 63, 2) native order
    cur = rest[None] + rng.normal(0, 0.5, (T, 2, 63, 2)).astype(np.float32)
    mk = np.stack([np.broadcast_to(rest[None], cur.shape), cur], 2)                         # (T, 2, 2, 63, 2)
    (root / task).mkdir(parents=True, exist_ok=True)
    (feat_root / task).mkdir(parents=True, exist_ok=True)
    np.savez(root / task / f'{ep:04d}.npz', w96=rng.random((T, 2, 96, 128)).astype(np.float16), mk=mk, rest=rest,
             valid=np.ones((T, 2, 63), bool), m=np.zeros((T, 2, 14), np.float32), m_valid=np.ones((T, 2), bool),
             e_c=np.zeros((T, 2), bool), joint=rng.normal(size=(T, 9)).astype(np.float32), step=np.arange(T))
    np.savez(feat_root / task / f'{ep:04d}.npz', patch=rng.normal(size=(T, 2, 16, 768)).astype(np.float16))


def test_dataset_windows(tmp_path):
    stats = json.load(open(DEFAULT_STATS_FILE))
    rng = np.random.default_rng(0)
    tac, feat = tmp_path / 'tactile', tmp_path / 'dino'
    for task, T in (('lift_can', 20), ('insert_hole', 18)):
        for ep in (0, 1):
            _write_episode(tac, feat, task, ep, T, stats, rng)
    data = UniVTACTokenizerData(str(tac), str(feat), tasks=['lift_can', 'insert_hole'], episodes='0-1', device='cpu')
    assert data.n_frames == 2 * 20 + 2 * 18
    anc = data.anchors(15)
    assert len(anc) == 2 * 5 + 2 * 3                     # T - 15 anchors per episode
    X = data.fields_at(anc[:4], (0, 5, 10, 15))
    assert X.shape == (4, 4, 2, 3, 96, 128) and torch.isfinite(X).all()
    assert data.vis_at(anc[:4], (0, 5, 10, 15)).shape == (4, 4, 2, 16, 768)
    a = data.actions(anc[:4], 15)
    assert a.shape == (4, 15, 8)
    i = int(anc[1])
    assert torch.allclose(a[1, 2], data.Q[i + 3] - data.Q[i])
    mean, std = data.action_stats(anc, 15)
    assert mean.shape == (15, 8) and (std > 0).all()
    cpu_store = UniVTACTokenizerData(str(tac), str(feat), tasks=['lift_can', 'insert_hole'], episodes='0-1',
                                     device='cpu', storage_device='cpu', verbose=False)
    assert torch.equal(cpu_store.fields_at(anc[:4], (0, 15)), data.fields_at(anc[:4], (0, 15)))


def test_deploy_frame_matches_episode_processing():
    stats = json.load(open(DEFAULT_STATS_FILE))
    rng = np.random.default_rng(1)
    depth = np.full((240, 320), TF.NO_CONTACT, np.float32)
    depth[100:140, 150:200] = 27.5                       # an indentation of 1 mm
    for gel in (0, 1):
        rest = np.asarray(stats['rest_native'][gel], np.float32)
        native = rest.copy()
        if gel == 1:                                     # deploy_frame takes the raw (unmirrored) sensor frame
            native[:, 0] = TF.W_IMG - native[:, 0]
        cur = native + rng.normal(0, 0.3, native.shape).astype(np.float32)
        marker = np.stack([native, cur])
        out = TF.deploy_frame(depth, marker, gel, stats)
        assert out['w96'].shape == (96, 128) and out['d'].shape == (7, 9, 2) and out['valid'].all()
        assert out['e_c'] and abs(out['w96'].max() - 1 / 3) < 1e-5
        samp = TF.grid_sampling_coords(torch.tensor(stats['rest_grid']).reshape(7, 9, 2), 'cpu')
        X = TF.build_fields(torch.from_numpy(out['w96'])[None], torch.from_numpy(out['d'])[None],
                            torch.from_numpy(out['valid'])[None], stats['sigma_u'], samp)
        assert X.shape == (1, 3, 96, 128) and torch.isfinite(X).all()
