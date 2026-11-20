# Evaluation in UniVTAC

TacInno-VLA is evaluated with the benchmark's own evaluator, `scripts/eval_policy.py` of
[UniVTAC](https://github.com/univtac/UniVTAC). The evaluator instantiates the TacInno-VLA client
(`tacinno/evaluation/univtac_client.py`), which queries a policy server over a websocket.

```
 UniVTAC (Isaac Sim)                                   GPU machine
 ┌──────────────────────────────┐   observation       ┌───────────────────────────┐
 │ scripts/eval_policy.py        │ ──────────────────▶ │ tools/serve_policy.py      │
 │   policy/TacInnoVLA (client)  │ ◀────────────────── │   TacInno-VLA (one task)   │
 └──────────────────────────────┘   action chunk      └───────────────────────────┘
```

## Protocol of the paper

| Setting | Value |
|:--|:--|
| Seeds | from 1000000, the same for every policy (100 valid episodes per task) |
| Action limit | twice the benchmark default for every policy (`UNIVTAC_STEP_LIM_SCALE=2`), because the demonstrations are longer than those for which the default limits were set |
| Errors | episodes interrupted by simulator errors are discarded and evaluation continues with the next seed; an error of the policy server or an invalid action chunk makes the client hold the robot still, and the episode fails |
| Deployment | the deploy configs that `install` writes |
| Paired comparisons | exact two-sided McNemar test on the seeds both policies completed |

## Step by step

**1. Install the client** into the UniVTAC checkout (once):

```bash
python tools/eval_univtac.py install --univtac-root /path/to/UniVTAC
```

This writes `policy/TacInnoVLA/` (client, `__init__.py`, `deploy_<task>.yml` and `deploy_<task>_tactile_off.yml`
for the six tasks) and applies `tools/univtac/eval_policy_protocol.patch`. The patch keeps the episode loop, seeds
and success checks of the benchmark; it only adds a per-episode record (`episodes.json` in the run directory), the
optional action-limit scale and a headless switch (`UNIVTAC_LIVESTREAM=0`).

**2. Start the policy server** for the task on a GPU machine:

```bash
python tools/serve_policy.py checkpoints/sim/lift_can --task lift_can:8000 --bind 0.0.0.0
```

The server binds to `127.0.0.1` unless `--bind` is given. The joint and tactile statistics of the released models
are bundled in `tacinno/datasets/stats/` and used by default; `--cond-stats` and `--tactile-stats` select other
files. On connect the server reports the task, the checkpoint and the tactile history it expects; the client checks
them before the first episode.

**3. Evaluate** 100 episodes with twice the default action limit:

```bash
python tools/eval_univtac.py run --univtac-root /path/to/UniVTAC --task lift_can --host <server> --port 8000
```

`--episodes`, `--start-seed` and `--time-limit-scale` change the protocol; arguments after `--` are passed to
`eval_policy.py`. Results go to `eval_result/TacInnoVLA/<task>/deploy_<task>/<time>/` of the UniVTAC checkout
(`episodes.json`, logs, and the benchmark's videos).

**4. Summarize.** Several run directories of the same policy can be merged (for example when the seeds are split over
several simulator processes). The success rate is computed over the first 100 valid episodes from seed 1000000:

```bash
python tools/eval_univtac.py summarize /path/to/UniVTAC/eval_result/TacInnoVLA/lift_can/deploy_lift_can
# 100 valid episodes (seeds 1000000-1000099): success 88/100 = 88.0%
```

## Tactile-off test

The tactile-off test replaces the tactile input with a fixed no-contact frame at test time while every other input
stays live, so any change of the success rate is attributable to tactile input.

```bash
python tools/eval_univtac.py run --univtac-root /path/to/UniVTAC --task lift_can --host <server> --port 8000 --tactile-off
python tools/analysis/mcnemar.py \
    --a /path/to/UniVTAC/eval_result/TacInnoVLA/lift_can/deploy_lift_can \
    --b /path/to/UniVTAC/eval_result/TacInnoVLA/lift_can/deploy_lift_can_tactile_off
```

By default the client freezes the first tactile frame of each episode, which has no contact for tasks that start with
an open gripper. For a task that starts in contact, record a no-contact frame (`l_depth`, `l_marker`, `r_depth`,
`r_marker` arrays of one step) into an `.npz` and set `no_contact_frame:` in `deploy_<task>_tactile_off.yml`.

## Client reference

The client is self-contained (NumPy, OpenCV, PyTorch, `websockets`, `msgpack`) so that it runs inside the simulator
environment without installing `tacinno`. Its module docstring documents the wire protocol. The main classes:

| Class | Role |
|:--|:--|
| `PolicyServerClient` | websocket connection, metadata checks (task, optional `expect_checkpoint`), reconnects |
| `TacInnoVLAPolicy` | UniVTAC policy: keeps the joint and tactile histories, executes action chunks |
| `TactileOff` | tactile-off test wrapper |
| `Policy(args)` | factory called by the evaluator with the deploy config |

The deploy config keys are `host`, `port` (overridable with `TACINNO_HOST` and `TACINNO_PORT`), `replan`,
`expect_checkpoint`, `tactile_off` and `no_contact_frame`.

## Statistics in Python

```python
from tacinno.evaluation import compare, first_results, first_valid_seeds, success_rate

ours = first_results(["eval_result/TacInnoVLA/lift_can/deploy_lift_can"])
off = first_results(["eval_result/TacInnoVLA/lift_can/deploy_lift_can_tactile_off"])
s_ours, s_off = first_valid_seeds(ours), first_valid_seeds(off)  # first 100 valid episodes from seed 1000000
print(success_rate(ours, s_ours))  # {'success': ..., 'valid': ..., 'rate': ..., 'missing': [...]}
print(compare(ours, off, sorted(set(s_ours) & set(s_off))))  # paired seeds, discordant counts, exact McNemar p
```
