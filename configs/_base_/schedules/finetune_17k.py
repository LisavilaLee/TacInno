# Per-task continuation of a joint model: 17000 steps, fresh AdamW, warm-up 200 steps to 5e-6 then constant,
# weight EMA 0.99 (the released weights are the EMA weights of the last step, ema_16999/).
schedule = dict(
    num_steps=17000, batch_size=64, warmup_steps=200, peak_lr=5e-6, end_lr=5e-6,
    optimizer=dict(b1=0.9, b2=0.95, eps=1e-8, weight_decay=1e-10, clip_grad_norm=1.0),
    ema_decay=0.99, ema_dtype="auto",
)
