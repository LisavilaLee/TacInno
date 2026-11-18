"""Frozen DINOv2-base features of the head and wrist camera images of every demonstration frame (tokenizer input).

    python tools/data/extract_image_features.py --src data/univtac/raw --out data/univtac/dino \
        --model facebook/dinov2-base [--tasks ...] [--episodes 0-99]

Input ``<src>/<task>/hdf5/<ep>.hdf5``: ``observation/{head,wrist}/rgb`` (JPEG bytes, 480 x 270). Each image is resized to
392 x 224 (bicubic), ImageNet-normalised and passed through DINOv2-base (no registers, frozen, fp16); the 16 x 28
patch tokens of the last layer are average-pooled to 4 x 4 = 16 tokens. Output ``<out>/<task>/<ep:04d>.npz``:
``patch`` (T, 2, 16, 768) fp16 and ``cls`` (T, 2, 768) fp16, views ordered (head, wrist).
``--model`` is a Hugging Face model id or a local snapshot directory.
"""
from __future__ import annotations

import argparse
import io
import os
import os.path as osp
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from tacinno.datasets.tokenizer_dataset import parse_episodes  # noqa: E402

TASKS = ['insert_hole', 'insert_tube', 'lift_bottle', 'lift_can', 'pull_out_key', 'put_bottle_in_shelf']
INPUT_WH = (392, 224)
POOL = (4, 4)


def main():
    import h5py
    from PIL import Image
    from transformers import Dinov2Model

    ap = argparse.ArgumentParser(description='DINOv2 image features for the TacInno tokenizer')
    ap.add_argument('--src', required=True, help='root with <task>/hdf5/<ep>.hdf5')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='facebook/dinov2-base')
    ap.add_argument('--tasks', nargs='+', default=TASKS)
    ap.add_argument('--episodes', default='0-99')
    ap.add_argument('--batch-size', type=int, default=256)
    a = ap.parse_args()

    dev = 'cuda'
    model = Dinov2Model.from_pretrained(a.model).to(dev).half().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    pool = ThreadPoolExecutor(8)

    def decode(b):
        return np.asarray(Image.open(io.BytesIO(bytes(b))).convert('RGB').resize(INPUT_WH, Image.BICUBIC))

    @torch.no_grad()
    def feats(imgs):
        x = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float().div(255)
        x = ((x - mean) / std).half()
        h = model(pixel_values=x).last_hidden_state                        # (B, 1 + 16 * 28, 768)
        gh, gw = INPUT_WH[1] // 14, INPUT_WH[0] // 14
        cls, p = h[:, 0], h[:, 1:].reshape(-1, gh, gw, h.shape[-1]).permute(0, 3, 1, 2)
        p = F.adaptive_avg_pool2d(p.float(), POOL).flatten(2).transpose(1, 2)  # (B, 16, 768)
        return p.half().cpu().numpy(), cls.half().cpu().numpy()

    t0, n_img = time.time(), 0
    for task in a.tasks:
        os.makedirs(osp.join(a.out, task), exist_ok=True)
        for ep in parse_episodes(a.episodes):
            dst = osp.join(a.out, task, f'{ep:04d}.npz')
            if osp.exists(dst):
                continue
            with h5py.File(osp.join(a.src, task, 'hdf5', f'{ep}.hdf5'), 'r') as f:
                raw = [f['observation/head/rgb'][()], f['observation/wrist/rgb'][()]]
            T = len(raw[0])
            P = np.zeros((T, 2, POOL[0] * POOL[1], 768), np.float16)
            C = np.zeros((T, 2, 768), np.float16)
            for v in range(2):
                imgs = list(pool.map(decode, raw[v]))
                for s in range(0, T, a.batch_size):
                    p, c = feats(imgs[s:s + a.batch_size])
                    P[s:s + a.batch_size, v] = p
                    C[s:s + a.batch_size, v] = c
            np.savez(dst + '.tmp.npz', patch=P, cls=C)
            os.replace(dst + '.tmp.npz', dst)
            n_img += 2 * T
            print(f'{task} {ep:04d} T={T} imgs={n_img} {n_img / (time.time() - t0):.0f} img/s', flush=True)
    print('DONE', round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
