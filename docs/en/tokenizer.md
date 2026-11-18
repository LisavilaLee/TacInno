# TacInno tokenizer

The TacInno tokenizer labels every demonstration frame with a discrete **tactile innovation code**. The code is the
part of the upcoming tactile change that the camera images and the executed actions cannot explain. TacInno-VLA
uses these codes only as a training target, so the tokenizer is not needed at deployment.

This page describes what the tokenizer computes, how it is trained and how its codes become policy targets.
Code: [`tacinno/models/tokenizer/`](../../tacinno/models/tokenizer). Configs: [`configs/tokenizer/`](../../configs/tokenizer).

## Inputs

For an anchor frame *t*, the tokenizer looks at a window of recorded frames *t + o* with offsets *o* = (0, 5, 10, 15).

| Input | Shape | Source and preprocessing |
|---|---|---|
| Tactile observation `fields` | (B, 4, 2, 3, 96, 128) | For each window frame and both gels: the indentation field `w = clip(28.5 - depth, 0, 6) / 3`, area-resampled from 240 x 320 to 96 x 128, and the marker tangential displacement field `(u_x, u_y) / sigma_u`. The second field is built from the 7 x 9 markers: missing markers are in-painted, then the grid is bicubically interpolated to the same 96 x 128 grid. The right gel is mirrored so both gels share one frame. See [`tactile_fields.py`](../../tacinno/datasets/tactile_fields.py). |
| Image features `vis` | (B, 4, 2, 16, 768) | Head and wrist RGB at each window frame, resized to 392 x 224. They go through frozen DINOv2-base, and the 16 x 28 last-layer patch tokens are average-pooled to 4 x 4. |
| Actions `actions` | (B, 15, 8) | Executed relative joint targets `joint[t+i, :8] - joint[t, :8]` for *i* = 1..15 (7 arm joints and the gripper opening). They are standardised per (*i*, joint) with statistics of the training anchors (`action_stats.pt`). |

## Architecture

**Encoder (inverse dynamics).** Each window frame contributes 137 tokens of width 384:

- 96 tactile patch tokens: 2 gels x 6 x 8 patches of 16 x 16. They come from one shared Conv2d(3, 384, 16, stride 16) plus a learned patch-position embedding and a gel embedding.
- 32 image tokens: 2 views x 16 pooled DINOv2 tokens, passed through LayerNorm and Linear(768, 384), plus a learned per-token embedding.
- 5 action tokens. Frame *t+5* gets actions *t+1..t+5*, frame *t+10* gets *t+6..t+10* and frame *t+15* gets *t+11..t+15*. Each action goes through an MLP (8 → 384 → 384, GELU) and gets a slot embedding. Frame *t* gets 5 learned empty-action tokens instead.
- 4 learned query tokens, the same 4 in every frame.

Six spatio-temporal blocks (pre-LN, residual) process the tokens. Each block has three sub-layers:

1. Spatial self-attention among the 137 tokens of a frame (6 heads).
2. Temporal self-attention across the 4 frames at each token position, with a causal mask so that frame *f* attends only to frames ≤ *f*.
3. An MLP (384 → 1536 → 384, GELU).

There is no temporal position embedding; the causal mask fixes the frame order. The 4 query outputs of the last frame go through LayerNorm and Linear(384, 32), giving `z` with shape (B, 4, 32).

**Quantizer.** There is one codebook of 16 x 32 per slot. Each slot takes its Euclidean nearest entry, and gradients pass straight through. The code is therefore 4 integers in [0, 16), 16 bit per frame. The codebook is initialised from encoder outputs at the first step, and entries unused for 2000 steps are re-initialised from encoder outputs.

**Decoder (forward dynamics).** It uses the same block type, 6 blocks, width 384.

- Frame *t* holds the 96 patch tokens of the known observation `X_t`.
- Frames *t+5*, *t+10* and *t+15* each hold 96 to-reconstruct tokens: one learned vector plus the patch-position and gel embeddings.
- Every frame also holds its 32 image tokens, its 5 action tokens and the 4 code vectors (Linear(32, 384) plus a slot embedding).

The to-reconstruct tokens go through LayerNorm and Linear(384, 768), and are un-patched to the predicted change `X_{t+5k} - X_t` for *k* = 1, 2, 3. The decoder never receives a tactile observation after *t*.

## Loss and what the code learns

`L = L_rec + L_codebook + 0.25 L_commit`.

`L_rec` is the L1 error of the predicted change, averaged over gels, channels and pixels, with the three horizons weighted equally. Pixels in contact at either frame of a pair (indentation above the 28.3 mm contact threshold) get weight 1 + 4.

Images and actions reach both the encoder and the decoder. The code therefore has no reason to carry what they already explain, and its 16-bit capacity is spent on the remainder. **There is no direct supervision of this property.** It follows from the bottleneck and is verified by the code-shuffle test, which training logs every `val_interval` steps:

- `val_true_h`: reconstruction error with the true codes, per horizon.
- `val_shuf_h`: the same error with codes permuted among frames of the same task.
- `val_copy_h`: error of predicting no change.
- `val_gap`: `mean(val_shuf_h) - mean(val_true_h)`, how much the decoder relies on the code.
- `slot_ppl`: per-slot code perplexity (at most 16).

The released tokenizer reaches `val_gap` 0.077 on validation demonstrations 90-99 (true 0.040 / 0.049 / 0.053 against shuffled 0.105 / 0.125 / 0.143 per horizon), with slot perplexities 10.2 to 13.9.

## Ablations

Each ablation changes only the model config (all variants share one parameter layout):

| Config | Change | Meaning |
|---|---|---|
| `tacinno-tokenizer_univtac-6task.py` | none | TacInno tokenizer |
| `..._wo-conditioning.py` | `context='none'` | No images or actions on either side. The code describes the full future tactile change. |
| `..._image-only-conditioning.py` | `context='image'` | Images but no actions. |
| `..._action-only-conditioning.py` | `context='action'` | Actions but no images. |
| `..._two-frame-code.py` | `offsets=(0, 15)` | Two frames, *t* and *t+15*. The future frame gets all 15 action tokens and the decoder predicts `X_{t+15} - X_t` only. |

## Data preparation

```bash
# 1. tactile inputs (one npz per episode) from the UniVTAC HDF5 demonstrations
python tools/data/export_tactile.py --src data/univtac/raw --out data/univtac/tactile
# 2. frozen DINOv2-base image features
python tools/data/extract_image_features.py --src data/univtac/raw --out data/univtac/dino --model facebook/dinov2-base
# 3. (optional) tactile normalisation statistics; a copy computed on our data is bundled in
#    tacinno/datasets/stats/univtac_tactile_stats.json and used by default
python tools/data/build_tactile_stats.py --data data/univtac/tactile --out data/univtac/tactile/tactile_stats.json
```

## Training

```bash
python tools/train_tokenizer.py configs/tokenizer/tacinno-tokenizer_univtac-6task.py --work-dir work_dirs/tokenizer
```

This trains for 12k steps with batch 256 on one GPU. The optimizer is AdamW with lr 3e-4, betas (0.9, 0.95) and weight decay 0.05, using 1k warm-up steps and cosine decay to 0.1x. Mixed precision is bf16.

- **Resume.** Re-running the same command continues from `work_dir/ckpt_latest.pt`. Model, optimizer, scheduler, step and every RNG state are restored, so a resumed run continues exactly like an uninterrupted one.
- **Output.** `model_<step>.pt` holds the weights. `action_stats.pt` holds the action standardisation that export needs.
- **Overrides.** Change any entry with `--cfg-options`, for example `train_cfg.batch_size=64 train_dataset.storage_device=cpu`.

## Exporting codes and policy targets

```bash
python tools/export_codes.py configs/tokenizer/tacinno-tokenizer_univtac-6task.py \
    work_dirs/tokenizer/model_012000.pt --out data/univtac/codes --episodes 0-99
python tools/data/make_code_targets.py data/univtac/codes data/univtac/cond
```

`export_codes.py` writes `<task>/<ep:04d>.npz` with `codes` (T, 4) int16 and `valid` (T,). Frames whose window does not fit in the episode (the last 15 frames) get -1. The export assembles the tactile observations inside the bf16 autocast region, the same way the run that produced the released codes did. With the released checkpoint it reproduces those codes bit for bit. `make_code_targets.py` aligns the codes with the frames of the policy training data in `data/univtac/cond/<split>/<task>/`, producing `code_tgt.npy` and `code_tv.npy` (see [policy.md](policy.md)).

The released tokenizer checkpoint (`model_012000.pt` with `cmd_stats.pt` next to it) loads with `strict=True` into the config above.
