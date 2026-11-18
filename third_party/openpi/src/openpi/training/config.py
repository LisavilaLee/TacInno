"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from re import S
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.models_pytorch.ftp1_model_config import FTP1ModelConfig
from openpi.models_pytorch.ftp1_model_config import (
    FTP1_SINGLE_ARM_ACTION_REP_DIM,
    FTP1_SINGLE_HAND_NUM_TACTILE_AREAS,
    FTP1_ACTION_HORIZON,
    FTP1_RESERVED_ACTION_DIM,
)

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Normalization type: 'quantile', 'zscore'
    norm_type: str = "quantile"

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05 | _model.ModelType.FTP1:
                # assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            norm_type="quantile" if model_config.model_type != ModelType.PI0 else "zscore",
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # If true and pytorch_weight_path is set, also load the optimizer state from the same
    # checkpoint directory. global_step is NOT restored (scheduler resets from step 0).
    # This gives mode "load weights + optimizer, but fresh scheduler".
    load_optimizer_from_weight_path: bool = False

    # If true, use partial (name+shape matching) loading for model weights from pytorch_weight_path.
    # Only keys whose names AND shapes match between checkpoint and current model are loaded;
    # mismatched or missing keys are silently skipped. Default off (normal strict=False loading).
    allow_partial_load: bool = False

    # If true, parameters restored from pytorch_weight_path use reduced LR during warmup only.
    # After warmup, all parameter groups use the same LR schedule.
    reduce_loaded_params_lr_warmup_only: bool = False

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    # Whether to enable torch.compile for model optimization (PyTorch 2.0+).
    # This can improve training speed but may have compatibility issues with some models.
    use_torch_compile: bool = False

    # Whether to enable gradient checkpointing in PyTorch training loops (if supported by the model).
    # This trades extra compute for lower memory usage. Can be toggled per experiment via CLI.
    gradient_checkpointing_enable: bool = True

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of workers to use for the validation data loader. If None, defaults to num_workers.
    # (Useful when val is I/O heavy and you want different parallelism than training.)
    val_num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # Step interval for pure-domain train grad-norm probes (for wandb TrainDomainGradNorm/*).
    # <= 0 disables the probe.
    domain_grad_probe_interval: int = 0
    # How often (in steps) to validate the model.
    val_interval: int = 1000
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If true, will use the validation dataset for training.
    use_val_dataset: bool = False
    val_ratio: float = 0.01
    # If true, will create a train/val split from the dataset. It's used only when compute norm stats.
    create_train_val_split: bool = False

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.domain_grad_probe_interval < 0:
            raise ValueError("domain_grad_probe_interval must be >= 0.")
        if self.val_num_workers is None:
            object.__setattr__(self, "val_num_workers", self.num_workers)


@dataclasses.dataclass(frozen=True)
class TransformMultiZarrTransDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Prepare data for policy training
        # For Zarr dataset, the data transforms & normalization are conducted inside the dataset.
        data_transforms = _transforms.Group(
            inputs=[],
            outputs=[],
        )
        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            assets_dirs=assets_dirs,
        )

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or MultiZarrDataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            assets_dirs=assets_dirs,
        )


@dataclasses.dataclass(frozen=True)
class MultiZarrDataConfig(DataConfig):
    # See the note in FTP1TrainConfig for the definition of the input & output.

    assets_dirs: pathlib.Path = tyro.MISSING
    dataset_config_path: str = ""  # Path to dataset_config.json file

    single_arm_action_rep_dim: int = FTP1_SINGLE_ARM_ACTION_REP_DIM  #  defined in ftp1_model_config.py
    single_hand_num_tactile_areas: int = FTP1_SINGLE_HAND_NUM_TACTILE_AREAS  #  defined in ftp1_model_config.py
    reserved_action_dim: int = FTP1_RESERVED_ACTION_DIM  #  defined in ftp1_model_config.py
    state_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    image_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    tactile_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    disable_history: bool = True
    action_down_sample_steps: int = tyro.MISSING
    proprioception_pose_rep: str = "relative"  # 'abs' or 'relative'
    action_pose_rep: str = "relative"  # 'abs' or 'relative'
    proprioception_joint_rep: str = "abs"  # 'abs' or 'relative'
    action_joint_rep: str = "mix"  # 'abs', 'relative', or 'mix' (mix: only hand slot 28 absolute)
    use_domain_condition: bool = False  # if 'True', will add domain condition in the instruction prompt.
    independent_norm_mode: str = "all"  # 'none', 'joint' or 'all'
    norm_time_dim: bool = False
    norm_type: str = "quantile"  # 'quantile', 'zscore'
    norm_sample_ratio: float = 1.0
    norm_batch_size: int = 32
    norm_num_workers: int = 16
    # Image-type tactile channel-stat sampling budgets used when computing normalization stats.
    norm_image_tactile_mode: Literal["channel_wise", "image_norm"] = "channel_wise"
    norm_image_channel_pool_max_size: int = 2_000_000
    norm_image_channel_batch_samples: int = 0
    contact_detection_threshold_k: float = 3.0
    non_tactile_dropout_ratio: float = 0.0
    reuse_norm_repo_id: str = ""

    use_val_dataset: bool = True
    val_ratio: float = 0.01
    create_train_val_split: bool = False
    bind_train_val_split_to_dataset_config: bool = False
    seed: int = 42

    # TacInno condition data (tacinno.datasets.ConditionData); off when cond_root or cond_keys is empty
    cond_root: str = ""
    cond_keys: tuple[str, ...] = ()
    cond_proprio_hist: int = 8
    cond_tac_hist: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class FTP1TrainConfig(TrainConfig):
    repo_id: str = tyro.MISSING

    # Path to dataset_config.json file that contains dataset paths and sample ratios
    dataset_config_path: str = ""
    check_norm_params_snapshot: bool = True

    # ============================ INPUT & OUTPUT DEFINITION ============================
    # action/state definintion: left-arm-dim + right-arm-dim + head-dim
    # left-arm-dim  = 9dof-wrist-dim + hand-dim  (self.single_arm_action_rep_dim)
    # right-arm-dim = 9dof-wrist-dim + hand-dim  (self.single_arm_action_rep_dim)
    # 9dof-head-dim = 9dof-wrist-dim             (9_dim)
    # total-dim = 2 * self.single_arm_action_rep_dim + 9
    single_arm_action_rep_dim: int = FTP1_SINGLE_ARM_ACTION_REP_DIM  # defined in ftp1_model_config.py
    single_hand_num_tactile_areas: int = FTP1_SINGLE_HAND_NUM_TACTILE_AREAS  # defined in ftp1_model_config.py
    reserved_action_dim: int = FTP1_RESERVED_ACTION_DIM  # defined in ftp1_model_config.py
    state_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    image_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    tactile_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    disable_history: bool = True
    action_down_sample_steps: int = tyro.MISSING
    # 1. we define proprioception_pose in relative action-space because the absolute space is not unified due to coordinate & camera-parameters definition.
    # 2. when disable_history=True, all history lists must stay empty and the data path is restricted to current-frame inputs.
    # 3. we define action_pose in relative action-space since this is how human works. It also unified the pose definition for different embodiments.
    # 4. we define priprioception_joint in absolute action-space since this is how human proprioception works.
    # 5. we define action_joint in relative action-space since this is how human finger-action works.
    proprioception_pose_rep: str = "relative"  # 'abs' or 'relative'
    action_pose_rep: str = "relative"  # 'abs' or 'relative'
    proprioception_joint_rep: str = "abs"  # 'abs' or 'relative'
    action_joint_rep: str = "relative"      # 'abs', 'relative', or 'mix' (mix: only hand slot 28 absolute)
    use_domain_condition: bool = False  # if 'True', will add domain condition in the instruction prompt.
    # ============================ Normalization Settings ============================
    independent_norm_mode: str = "all"  # 'none', 'joint' or 'all'
    # independent_norm_mode:
    # 1. none: only do independent normalization for tactile.
    # 2. joint: do independent normalization for tactile & hand joints.
    # 3. all: do independent normalization for tactile & hand joints & wrist pose.
    norm_time_dim: bool = False
    norm_type: str = "quantile"  # 'quantile', 'zscore'
    norm_sample_ratio: float = 1.0
    norm_batch_size: int = 32
    norm_num_workers: int = 16
    # Image-type tactile channel-stat sampling budgets used when computing normalization stats.
    norm_image_tactile_mode: Literal["channel_wise", "image_norm"] = "channel_wise"
    norm_image_channel_pool_max_size: int = 2_000_000
    norm_image_channel_batch_samples: int = 0
    contact_detection_threshold_k: float = 3.0
    skip_contact_detection: bool = True
    reuse_norm_repo_id: str = ""

    # =========================== Training Dropout Settings ===========================
    non_tactile_dropout_ratio: float = 0.0
    # =========================== Validation Settings ===========================
    use_val_dataset: bool = True
    val_ratio: float = 0.01

    # =========================== Data Statistics Settings ===========================
    data_sample_ratio: float = 1.0  # Ratio of validation samples to collect for data statistics (0.0-1.0)
    norm_commit: str = ""  # Optional commit identifier for file naming (e.g., "v1", "exp1"). If not empty, files will be named as {norm_type}_{norm_commit}_{field_name}
    create_train_val_split: bool = False
    bind_train_val_split_to_dataset_config: bool = False
    seed: int = 42

    # the data config, defined in __post_init__
    data: DataConfigFactory = dataclasses.field(init=False)

    # If true, set the `decay_step` to the number of training steps.
    lr_decay_till_end: bool = True

    # TacInno condition data (see MultiZarrDataConfig)
    cond_root: str = ""
    cond_keys: tuple[str, ...] = ()
    cond_proprio_hist: int = 8
    cond_tac_hist: tuple[int, ...] = ()

    def _build_data_factory(self) -> DataConfigFactory:
        """(Re)build the DataConfigFactory from the current FTP1TrainConfig fields.

        Note: `data` is derived from other fields (e.g. `dataset_config_path`). Some CLI
        override paths update those fields without re-running `__post_init__`, so we
        also rebuild `data` in `finalize_config()`.
        """
        return TransformMultiZarrTransDataConfig(
            repo_id=self.repo_id,
            base_config=MultiZarrDataConfig(
                dataset_config_path=self.dataset_config_path,
                single_arm_action_rep_dim=self.single_arm_action_rep_dim,
                single_hand_num_tactile_areas=self.single_hand_num_tactile_areas,
                reserved_action_dim=self.reserved_action_dim,
                state_down_sample_steps=self.state_down_sample_steps,
                image_down_sample_steps=self.image_down_sample_steps,
                tactile_down_sample_steps=self.tactile_down_sample_steps,
                disable_history=self.disable_history,
                action_down_sample_steps=self.action_down_sample_steps,
                proprioception_pose_rep=self.proprioception_pose_rep,
                action_pose_rep=self.action_pose_rep,
                proprioception_joint_rep=self.proprioception_joint_rep,
                action_joint_rep=self.action_joint_rep,
                use_domain_condition=self.use_domain_condition,
                independent_norm_mode=self.independent_norm_mode,
                norm_time_dim=self.norm_time_dim,
                norm_type=self.norm_type,
                norm_sample_ratio=self.norm_sample_ratio,
                norm_batch_size=self.norm_batch_size,
                norm_num_workers=self.norm_num_workers,
                norm_image_tactile_mode=self.norm_image_tactile_mode,
                norm_image_channel_pool_max_size=self.norm_image_channel_pool_max_size,
                norm_image_channel_batch_samples=self.norm_image_channel_batch_samples,
                contact_detection_threshold_k=self.contact_detection_threshold_k,
                reuse_norm_repo_id=self.reuse_norm_repo_id,
                non_tactile_dropout_ratio=self.non_tactile_dropout_ratio,
                use_val_dataset=self.use_val_dataset,
                val_ratio=self.val_ratio,
                create_train_val_split=self.create_train_val_split,
                bind_train_val_split_to_dataset_config=self.bind_train_val_split_to_dataset_config,
                seed=self.seed,
                cond_root=self.cond_root,
                cond_keys=tuple(self.cond_keys),
                cond_proprio_hist=self.cond_proprio_hist,
                cond_tac_hist=tuple(self.cond_tac_hist),
            ),
        )

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "data", self._build_data_factory())
        if self.lr_decay_till_end:
            assert isinstance(self.lr_schedule, _optimizer.CosineDecaySchedule), (
                "Only CosineDecaySchedule is supported for lr_decay_till_end"
            )
            object.__setattr__(
                self, "lr_schedule", dataclasses.replace(self.lr_schedule, decay_steps=self.num_train_steps)
            )
        model_updates = {}
        calculated_action_dim = 2 * self.single_arm_action_rep_dim + 9 + self.reserved_action_dim
        if self.model.action_dim != calculated_action_dim:
            model_updates["action_dim"] = calculated_action_dim
        if isinstance(self.model, FTP1ModelConfig) and self.model.disable_history != self.disable_history:
            model_updates["disable_history"] = self.disable_history
        if model_updates:
            object.__setattr__(self, "model", dataclasses.replace(self.model, **model_updates))
        self.model.set_single_hand_num_tactile_areas(self.single_hand_num_tactile_areas)

    def _validate_removed_training_modes(self) -> None:
        """Reject legacy FTP1 training-mode fields removed from the master stack.

        The March 2026 expert-only cleanup removed several FTP1-specific training toggles.
        Current CLI paths no longer expose them, but older programmatic config construction may
        still inject them onto the dataclass instance. Catch those cases with a helpful error.
        """
        removed_fields = {
            "tactile_mae_mask_ratio": "tactile MAE pretraining is no longer supported on `master`.",
            "train_vlm_branch": "partial branch-freeze training is no longer supported on `master`.",
            "train_action_branch": "partial branch-freeze training is no longer supported on `master`.",
            "train_tactile_branch": "partial branch-freeze training is no longer supported on `master`.",
        }
        legacy_fields = {name: getattr(self, name) for name in removed_fields if name in vars(self)}
        if not legacy_fields:
            return

        details = ", ".join(f"{name}={value!r}" for name, value in sorted(legacy_fields.items()))
        guidance = " ".join(removed_fields[name] for name in sorted(legacy_fields))
        raise ValueError(
            "Legacy FTP1 training-mode fields are no longer supported by the expert-only `master` stack: "
            f"{details}. {guidance}"
        )

    def finalize_config(self):
        """Finalize the config after command line arguments are applied.
        This sets all computed/derived values that depend on command line arguments.
        """
        self._validate_removed_training_modes()
        # Rebuild derived data factory after CLI overrides (e.g. dataset_config_path).
        object.__setattr__(self, "data", self._build_data_factory())

        # Prepare model updates
        model_updates = {}
        calculated_action_dim = 2 * self.single_arm_action_rep_dim + 9 + self.reserved_action_dim
        if self.model.action_dim != calculated_action_dim:
            model_updates["action_dim"] = calculated_action_dim
        if isinstance(self.model, FTP1ModelConfig) and self.model.disable_history != self.disable_history:
            model_updates["disable_history"] = self.disable_history
        # Set assets_dir from assets_base_dir if not already set
        if isinstance(self.model, FTP1ModelConfig) and self.model.assets_dir is None:
            model_updates["assets_dir"] = self.assets_base_dir
        # Apply model updates
        if model_updates:
            new_model = dataclasses.replace(self.model, **model_updates)
            object.__setattr__(self, "model", new_model)
        # Call finalize_config on model config if it has one
        if isinstance(self.model, FTP1ModelConfig):
            self.model.finalize_config(assets_dir=self.assets_base_dir)


# Keys that must match between norm compute and training (stored in norm folder as norm_params_snapshot.json).
NORM_PARAMS_SNAPSHOT_KEYS = (
    "repo_id",
    "dataset_config_path",
    "action_down_sample_steps",
    "state_down_sample_steps",
    "image_down_sample_steps",
    "tactile_down_sample_steps",
    "disable_history",
    "proprioception_pose_rep",
    "action_pose_rep",
    "proprioception_joint_rep",
    "action_joint_rep",
    "norm_type",
    "independent_norm_mode",
    "norm_time_dim",
    "norm_image_tactile_mode",
    "norm_image_channel_pool_max_size",
    "norm_image_channel_batch_samples",
    "contact_detection_threshold_k",
)


def get_norm_params_snapshot(config: "TrainConfig") -> dict | None:
    """Build a JSON-serializable dict of params that must match between compute_norm_stats and training.

    Used to save in norm folder (compute_norm_stats) and to validate at train start (zarr_train_ftp1_pytorch).
    Returns None for non-FTP1 configs.
    """
    if not isinstance(config, FTP1TrainConfig):
        return None
    out = {}
    for key in NORM_PARAMS_SNAPSHOT_KEYS:
        if not hasattr(config, key):
            continue
        val = getattr(config, key)
        if isinstance(val, pathlib.Path):
            val = str(val)
        out[key] = val
    return out


def check_norm_params_snapshot_matches(
    saved: dict, current: dict, *, ignored_keys: set[str] | None = None
) -> list[str]:
    """Compare saved vs current norm params; return list of diff messages. Empty if match."""
    ignored_keys = ignored_keys or set()
    diffs = []
    keys = set(saved.keys()) | set(current.keys())
    for k in sorted(keys):
        if k in ignored_keys:
            continue
        s = saved.get(k)
        c = current.get(k)
        if s != c:
            diffs.append(f"  {k}: saved={s!r} vs current={c!r}")
    return diffs


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    FTP1TrainConfig(
        name="ftp1",
        model=FTP1ModelConfig(action_horizon=FTP1_ACTION_HORIZON),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
