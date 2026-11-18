# UniVTAC demonstrations for the TacInno tokenizer: six tasks, 100 demonstrations each,
# episodes 0-89 for training and 90-99 for validation.
# Prepare with tools/data/export_tactile.py (tactile/) and tools/data/extract_image_features.py (dino/).
data_root = 'data/univtac'
tasks = ('insert_hole', 'insert_tube', 'lift_bottle', 'lift_can', 'pull_out_key', 'put_bottle_in_shelf')

train_dataset = dict(
    type='UniVTACTokenizerData',
    tactile_root=data_root + '/tactile',
    feature_root=data_root + '/dino',
    stats_file=None,          # None: the bundled tacinno/datasets/stats/univtac_tactile_stats.json
    tasks=tasks,
    episodes='0-89',
    with_vision=None,         # None: load image features only if the model is conditioned on images
    storage_device=None,      # 'cpu' keeps the frames in host memory (identical results, less GPU memory)
)
val_dataset = dict(train_dataset, episodes='90-99')
