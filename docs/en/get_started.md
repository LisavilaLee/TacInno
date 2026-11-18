# Get started

TacInno uses two Python environments:

| Environment | Used for | Main packages |
|:--|:--|:--|
| `tacinno` | tokenizer and policy training, code export, the policy server | Python 3.11, PyTorch 2.7, transformers 4.53 (patched by openpi) |
| UniVTAC | closed-loop evaluation in simulation (Isaac Sim) | the benchmark's own environment, plus `websockets` and `msgpack` for the client |

The two only talk over a websocket, so the simulator and the policy can run on different machines.

## 1. Training and serving environment

```bash
git clone https://github.com/LisavilaLee/TacInno.git
cd TacInno

conda create -n tacinno python=3.11 -y
conda activate tacinno
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e third_party/openpi_client
pip install -e third_party/openpi   # pi0.5 in PyTorch (vendored, trimmed)
pip install -e .
```

The PyTorch implementation of $\pi_{0.5}$ in openpi replaces a few files of `transformers` (PaliGemma, Gemma and
SigLIP with the changes openpi needs). Copy them over the installed package once:

```bash
cp -r third_party/openpi/src/openpi/models_pytorch/transformers_replace/* \
      "$(python -c 'import transformers, os; print(os.path.dirname(transformers.__file__))')/"
```

Check the installation:

```bash
python -m pytest tests -q
```

### Pretrained components

| Component | Used by | Source |
|:--|:--|:--|
| $\pi_{0.5}$ base checkpoint (PyTorch) | TacInno-VLA initialization | [openpi](https://github.com/Physical-Intelligence/openpi) (convert the JAX checkpoint with openpi's conversion script) |
| PaliGemma SentencePiece tokenizer | language instruction | openpi / big_vision |
| `facebook/dinov2-base` | image features of the TacInno tokenizer | [Hugging Face](https://huggingface.co/facebook/dinov2-base) |

* The $\pi_{0.5}$ base weights (a directory with `model.safetensors`) are read from `pi05_base` in
  `configs/_base_/policy_runtime.py` (default `checkpoints/pi05_base_pytorch/pi05`); override it with
  `--cfg-options pi05_base=/path/to/pi05 init_weights.path=/path/to/pi05`. Serving a released TacInno-VLA checkpoint
  does not need them.
* The PaliGemma tokenizer (`paligemma_tokenizer.model`) is downloaded by openpi from `gs://big_vision/` on first use
  and cached in `$OPENPI_DATA_HOME` (default `~/.cache/openpi`). On a machine without access, copy the file to
  `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model` beforehand.
* The DINOv2 features of the tokenizer are extracted once with `tools/data/extract_image_features.py`
  ([data_preparation.md](data_preparation.md)).

## 2. Simulation environment (evaluation)

Install [UniVTAC](https://github.com/univtac/UniVTAC) following its instructions, then add the two packages the
TacInno-VLA client needs to that environment:

```bash
pip install websockets msgpack
```

and install the client into the checkout (copies `tacinno/evaluation/univtac_client.py`, writes the deploy configs and
patches the evaluator to record per-episode results):

```bash
python tools/eval_univtac.py install --univtac-root /path/to/UniVTAC
```

See [evaluation.md](evaluation.md) for running the evaluation.

## 3. Hardware used for the paper

| Stage | Hardware |
|:--|:--|
| Tokenizer training | one GPU (any recent GPU with 24 GB) |
| TacInno-VLA training | GPUs with at least 80 GB each (full fine-tuning of $\pi_{0.5}$) |
| Policy server | one GPU with at least 10 GB (bf16 inference) |
| UniVTAC evaluation | one RTX-class GPU per simulator process |

## 4. Configs and tools

Every tool takes a config file and optional overrides, as in MMDetection:

```bash
python tools/train_tokenizer.py configs/tokenizer/tacinno-tokenizer_univtac-6task.py --work-dir work_dirs/tokenizer
python tools/train_policy.py configs/tacinno_vla/tacinno-vla_pi05_20k_univtac-insert-tube.py \
    --cfg-options data.data_root=/path/to/data
```

Configs are Python files that inherit from `configs/_base_/` through `_base_ = [...]`; dicts are merged recursively and
`dict(_delete_=True, ...)` replaces a base dict. The resolved config is saved to the work directory of every run.
