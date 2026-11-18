# 12k steps, batch 256, AdamW (betas 0.9 / 0.95), lr 3e-4 with 1k warm-up and cosine decay to 0.1x, wd 0.05.
train_cfg = dict(
    max_steps=12000,
    batch_size=256,
    seed=0,
    log_interval=100,
    val_interval=1000,
    n_val=4096,                    # validation anchors for the code-shuffle test
    ckpt_interval=500,             # ckpt_latest.pt (resumable)
    keep_interval=5000,            # model_<step>.pt
    codebook_reset_interval=2000,  # re-initialise unused codebook entries
)
optim_cfg = dict(lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95), warmup=1000, lr_floor=0.1, grad_clip=1.0)
