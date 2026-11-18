runtime = dict(
    seed=42,
    precision="bfloat16",
    use_torch_compile=True,
    gradient_checkpointing=False,
    log_interval=50,
    save_interval=5000,     # checkpoints <work_dir>/<step>/ (model, optimizer, EMA, RNG states: exact resume)
    keep_period=100000,     # also keep every multiple of this step
    keep_last=1,            # newest resume checkpoints kept on disk
    pre_delete=False,       # remove the previous checkpoint before writing the next one (saves disk)
    compile_cache_dir=None, # torch.compile cache base directory (default $TMPDIR)
)
# pi0.5 base weights converted to PyTorch (openpi), directory containing model.safetensors
pi05_base = "checkpoints/pi05_base_pytorch/pi05"
init_weights = dict(type="pi05_base", path=pi05_base)
