# TacInno tokenizer: spatio-temporal inverse-dynamics encoder over the window t, t+5, t+10, t+15, per-slot VQ
# (4 slots x 16 codes = 16 bit) and a forward-dynamics decoder that predicts X_{t+5k} - X_t from X_t, images,
# actions and the code. See docs/en/tokenizer.md.
model = dict(
    type='TacInnoTokenizer',
    offsets=(0, 5, 10, 15),   # window frames relative to t; actions t+1 .. t+offsets[-1]
    context='both',           # conditioning of encoder and decoder: 'both' | 'image' | 'action' | 'none'
    embed_dim=384,
    num_heads=6,
    enc_depth=6,
    dec_depth=6,
    num_slots=4,
    codebook_size=16,
    code_dim=32,
    field_size=(96, 128),
    patch_size=16,
    field_channels=3,
    num_gels=2,
    num_views=2,
    vis_tokens_per_view=16,
    vis_dim=768,
    action_dim=8,
    target='cumulative',      # X_{t+5k} - X_t
    contact_weight=4.0,       # L1 weight 1 + 4 on pixels in contact at either frame
    commit_weight=0.25,
    amp=True,                 # bf16 autocast for encoder and decoder on CUDA
)
