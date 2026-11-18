# Joint six-task stage: 17000 steps, global batch 64, AdamW (b2 0.95, weight decay 1e-10, clip 1.0),
# warm-up 500 steps to 5e-5 then cosine to 5e-6; no weight EMA.
schedule = dict(
    num_steps=17000, batch_size=64, warmup_steps=500, peak_lr=5e-5, end_lr=5e-6,
    optimizer=dict(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_grad_norm=1.0),
    ema_decay=0.0, ema_dtype="auto",
)
