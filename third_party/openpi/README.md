# Vendored pi0.5 (openpi via FTP-1)

This directory holds the PyTorch pi0.5 implementation that TacInno-VLA is built on: the PyTorch port of
[openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence, Apache-2.0) as extended by
[FTP-1](https://github.com/michaelyuancb/ftp1-policy) (Apache-2.0), trimmed to the modules that TacInno-VLA training
and inference use. `third_party/openpi_client` is the openpi websocket client package (Apache-2.0), used for the
policy-server wire format.

TacInno-specific code lives in `tacinno/`; this copy only contains the hooks that connect the two, listed in
[MODIFICATIONS.md](MODIFICATIONS.md).

Install (after the repository requirements):

```bash
pip install -e third_party/openpi_client -e third_party/openpi
# openpi patches transformers 4.53.2 (adaRMSNorm, KV-cache handling for the action expert):
cp -r third_party/openpi/src/openpi/models_pytorch/transformers_replace/* \
      "$(python -c 'import os, transformers; print(os.path.dirname(transformers.__file__))')/"
```
