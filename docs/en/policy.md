# TacInno-VLA: policy

TacInno-VLA is the pi0.5 policy trained with TacInno. It reads the cameras, the language prompt, the proprioceptive
state and the tactile observations of three frames, and outputs a 32-step action chunk. During training a code head
predicts the tactile innovation code of the TacInno tokenizer (see [tokenizer.md](tokenizer.md)); the head is not used
for acting.

Code: [`tacinno/models/policy`](../../tacinno/models/policy) (condition tokens, losses, model builder),
[`tacinno/engine`](../../tacinno/engine) (trainer), [`tacinno/apis`](../../tacinno/apis) (inference and server); the
pi0.5 backbone is vendored in [`third_party/openpi`](../../third_party/openpi).

## Inputs

| Input | Source | Shape / preprocessing | Enters |
|---|---|---|---|
| head and wrist images | simulator cameras (real robot: third-person and wrist) | uint8 RGB, resized to 224 x 224, scaled to [-1, 1] | prefix (SigLIP) |
| language prompt | task instruction | PaliGemma tokenizer, at most 150 tokens with the images | prefix |
| state | 7 arm joints + gripper opening at time t | written into the 120-D unified state (arm 9:16, gripper 44), z-score per task | adaRMSNorm condition of the action expert |
| proprio history (optional, `cond_proprio_hist`) | joints of frames t-7 .. t | (8, 8), normalized with the training mean / std; frames before the episode start repeat the first | 8 condition tokens |
| tactile observations | frames t-8, t-4, t, both gels | (3, 2, 3, 96, 128): indentation field and marker tangential displacement field (x, y) / sigma_u | 60 condition tokens |

Frames are control steps (one control step = one recorded data frame in UniVTAC); histories are clipped at the episode
start. The tactile observation is built by [`tacinno/datasets/tactile_fields.py`](../../tacinno/datasets/tactile_fields.py)
from the gel depth image and the marker positions, identically for training (stored as float16) and deployment
(rounded through float16).

## Structure

* **Prefix** (PaliGemma: SigLIP So400m/14 + Gemma 2B, width 2048, 18 layers): 256 tokens per image + the prompt tokens,
  bidirectional attention among themselves; the prefix does not attend to the expert sequence.
* **Action expert** (Gemma 300M, width 1024, 18 layers, 8 query heads, 1 KV head, head dim 256), sequence
  `[proprio x H | tactile x 60 | code queries x 4 | noisy actions x 32]`. Each expert layer attends jointly over the prefix
  keys / values and the expert sequence.
  * proprio tokens: per frame Linear 8->1024, SiLU, Linear 1024->1024, plus a learned time embedding (8, 1024).
  * tactile tokens: per frame and gel, Conv 16x16 / stride 16 on the 3 x 96 x 128 observation -> 48 patch tokens of
    width 384 + learned position and gel embeddings; a learned summary query is prepended; 2 pre-norm Transformer
    encoder layers (6 heads, MLP 1536, full attention). The summary output and the 48 patch outputs average-pooled to
    3 x 3 give 10 tokens -> LayerNorm -> Linear 384->1024, plus a token-index embedding (10, 1024) and a frame
    embedding (3, 1024). 3 frames x 2 gels x 10 = 60 tokens.
  * code queries: 4 learned vectors (no input content).
  * each group is scaled by tanh(g) with a learned scalar g (initial 0.1).
  * noisy actions: Linear 120->1024 of the interpolated chunk `x_tau = tau * noise + (1 - tau) * actions`.
* **Attention mask** (expert sequence): condition tokens attend to the prefix and to each other, never to the action
  tokens; action tokens attend to the prefix, the proprio and tactile tokens and all action tokens; no token other
  than the code queries attends to the code queries.
* **Updates inside the expert**: the denoising-time adaRMS modulation acts only on the 32 action tokens. The tactile
  tokens and the code queries are updated by every expert layer (plain residual, RMSNorm without modulation); the
  proprio tokens are not (each layer only projects their input embedding to keys and values).
* **Output**: the last-layer expert outputs of the 32 action tokens -> Linear 1024->120 = flow-matching velocity.
  At inference 10 Euler steps from Gaussian noise; the prefix is computed once per decision and its keys and values
  are reused in every step. The chunk holds arm joint targets relative to the joints at inference time and the
  absolute gripper opening; deployment executes the first rows of the chunk and re-plans (`replan` in the deploy
  config).

## Losses

`L = L_flow + lambda_contact * L_contact + lambda_code * L_code`

| Term | Prediction | Target | Reaches |
|---|---|---|---|
| L_flow | velocity of the 32 x 120 chunk | `noise - actions` (MSE, action mask) | all parameters except the two heads and the code queries |
| L_code | code queries' last-layer outputs -> LayerNorm -> per-slot Linear 1024->16 | TacInno tokenizer code of frame t (4 slots x 16 entries, encodes the tactile change t -> t+15 that the images and the executed actions cannot explain); cross-entropy, frames without t+15 inside the episode masked | code head, code queries, expert, prefix language model, tactile encoder (through the code queries' attention) |
| L_contact | per gel and tactile frame: summary (384) -> Linear 384->256, GELU, Linear 256->15 | 14 standardized contact properties (MSE, masked by validity) + contact label (BCE) | tactile encoder and this head only |

lambda_code and lambda_contact are calibrated once per data set: a 200-step run with both weights 1, then each weight
is set so that its term is 15 % of L_flow averaged over steps 100-199 (UniVTAC: 0.119 / 0.467; real robot:
0.118 / 0.406).

**What has a direct supervision signal and what does not.** The code head and the contact-state head are supervised
directly. Whether the action tokens use the tactile tokens is not supervised: the action tokens never attend to the
code queries, so the code loss can only shape the shared parameters (tactile encoder, expert, language model), and the
actions are trained only by L_flow. This is verified at test time with the tactile-off test (replace every tactile
frame by a no-contact frame, `tools/serve_policy.py --tactile-off`) and by comparing with the variant without the
code loss (`code_loss_weight=0`).

## Training settings

All configs: global batch 64, AdamW (b1 0.9, b2 0.95, eps 1e-8, weight decay 1e-10), gradient clipping 1.0,
bfloat16, torch.compile, seed 42; demonstrations 0-49 of each UniVTAC task (real robot: the first 45 of each task).
The config of every released checkpoint is in
[`configs/tacinno_vla/`](../../configs/tacinno_vla) (real robot: [`configs/real_robot/`](../../configs/real_robot)), and
the released weights are the EMA weights (decay 0.99) of the last step.

## Training

```bash
# one node, N GPUs (the global batch stays 64)
bash tools/dist_train.sh configs/tacinno_vla/tacinno-vla_univtac-insert-tube.py 2
# override any key
python tools/train_policy.py CONFIG --cfg-options schedule.num_steps=1000 data.data_root=/path/to/univtac
```

A run writes `work_dirs/<config>/<step>/` (model, optimizer, EMA, every rank's RNG states, configs and normalization
statistics) and resumes automatically from the newest complete one (same step, optimizer, RNG states and data
position; the next update is step s + 1). The last step also writes the deployable `ema_<step>/`.
`runtime.keep_last` and `runtime.pre_delete` bound the disk use (about 35 GB per resume checkpoint).

## Inference and serving

```python
from tacinno.apis import init_policy
from tacinno.datasets import stats_path
policy = init_policy("checkpoints/sim/lift_can", task="lift_can", cond_stats=stats_path("univtac_cond_stats"))
chunk = policy.infer(head=rgb_head, wrist=rgb_wrist, qpos=q8, qhist=q8_history,
                     tac_frames=[frame_t8, frame_t4, frame_t], noise_seed=0)   # (32, 8): arm deltas, gripper
```

```bash
python tools/serve_policy.py checkpoints/sim/lift_can --task lift_can:8000
# a policy trained on your own data: pass its statistics
python tools/serve_policy.py work_dirs/<config>/ema_<step> --task lift_can:8000 --config CONFIG \
    --cond-stats data/univtac/cond/cond_stats.json
```

The server protocol is described in [`tacinno/apis/server.py`](../../tacinno/apis/server.py); the UniVTAC evaluation
client is described in [evaluation.md](evaluation.md).

## Limitations

* The code queries are kept in the inference sequence (removing them would shift the rotary positions of the action
  tokens); they cost 4 tokens and do not change the actions.
* The policy assumes the tactile frames are aligned with the control steps at which the demonstrations were recorded.
