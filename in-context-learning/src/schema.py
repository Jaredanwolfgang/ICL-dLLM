from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Optional

import yaml


TASK_LIST = {
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
}

MODEL_FAMILIES = {"gpt2", "lstm", "diffusion_encoder", "diffusion_decoder"}
DATA_SOURCES = {"gaussian"}


def _require_keys(mapping: Dict[str, Any], required: Iterable[str]) -> None:
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")


def _to_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer for '{key}', got {value!r}") from exc


def _to_float(value: Any, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected float for '{key}', got {value!r}") from exc


@dataclass
class CurriculumRange:
    start: int
    end: int
    inc: int
    interval: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumRange":
        _require_keys(data, ("start", "end", "inc", "interval"))
        return cls(
            start=_to_int(data["start"], "curriculum.start"),
            end=_to_int(data["end"], "curriculum.end"),
            inc=_to_int(data["inc"], "curriculum.inc"),
            interval=_to_int(data["interval"], "curriculum.interval"),
        )


@dataclass
class CurriculumConfig:
    dims: CurriculumRange
    points: CurriculumRange

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumConfig":
        _require_keys(data, ("dims", "points"))
        if not isinstance(data["dims"], dict) or not isinstance(data["points"], dict):
            raise ValueError("Curriculum 'dims' and 'points' must be dictionaries.")
        return cls(
            dims=CurriculumRange.from_dict(data["dims"]),
            points=CurriculumRange.from_dict(data["points"]),
        )


@dataclass
class ModelConfig:
    family: str
    n_positions: int
    n_dims: int
    n_embd: int
    n_layer: int
    n_head: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        _require_keys(
            data, ("family", "n_positions", "n_dims", "n_embd", "n_layer", "n_head")
        )
        family = str(data["family"])
        if family not in MODEL_FAMILIES:
            raise ValueError(
                f"Model family '{family}' is not supported. "
                f"Expected one of {sorted(MODEL_FAMILIES)}."
            )
        return cls(
            family=family,
            n_positions=_to_int(data["n_positions"], "model.n_positions"),
            n_dims=_to_int(data["n_dims"], "model.n_dims"),
            n_embd=_to_int(data["n_embd"], "model.n_embd"),
            n_layer=_to_int(data["n_layer"], "model.n_layer"),
            n_head=_to_int(data["n_head"], "model.n_head"),
        )


@dataclass
class TrainingConfig:
    task: str
    task_kwargs: Dict[str, Any] = field(default_factory=dict)
    num_tasks: Optional[int] = None
    num_training_examples: Optional[int] = None
    data: str = "gaussian"
    batch_size: int = 64
    learning_rate: float = 3e-4
    train_steps: int = 1000
    save_every_steps: int = 1000
    keep_every_steps: int = -1
    resume_id: Optional[str] = None
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        _require_keys(data, ("task", "task_kwargs", "curriculum"))
        task = str(data["task"])
        if task not in TASK_LIST:
            raise ValueError(
                f"Training task '{task}' is not supported. "
                f"Expected one of {sorted(TASK_LIST)}."
            )
        task_kwargs = data.get("task_kwargs") or {}
        if not isinstance(task_kwargs, dict):
            raise ValueError("training.task_kwargs must be a dictionary.")
        data_source = str(data.get("data", "gaussian"))
        if data_source not in DATA_SOURCES:
            raise ValueError(
                f"Data source '{data_source}' is not supported. "
                f"Expected one of {sorted(DATA_SOURCES)}."
            )
        curriculum_cfg = CurriculumConfig.from_dict(data["curriculum"])
        num_tasks = data.get("num_tasks")
        num_training_examples = data.get("num_training_examples")
        return cls(
            task=task,
            task_kwargs=task_kwargs,
            num_tasks=None if num_tasks is None else _to_int(num_tasks, "training.num_tasks"),
            num_training_examples=(
                None
                if num_training_examples is None
                else _to_int(num_training_examples, "training.num_training_examples")
            ),
            data=data_source,
            batch_size=_to_int(data.get("batch_size", 64), "training.batch_size"),
            learning_rate=_to_float(
                data.get("learning_rate", 3e-4), "training.learning_rate"
            ),
            train_steps=_to_int(data.get("train_steps", 1000), "training.train_steps"),
            save_every_steps=_to_int(
                data.get("save_every_steps", 1000), "training.save_every_steps"
            ),
            keep_every_steps=_to_int(
                data.get("keep_every_steps", -1), "training.keep_every_steps"
            ),
            resume_id=data.get("resume_id"),
            curriculum=curriculum_cfg,
        )


@dataclass
class WandbConfig:
    enabled: bool = True
    project: Optional[str] = "in-context-training"
    entity: Optional[str] = None
    notes: str = ""
    name: Optional[str] = None
    log_every_steps: int = 10

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "WandbConfig":
        if data is None:
            data = {}
        enabled = bool(data.get("enabled", True))
        project = data.get("project", "in-context-training")
        if enabled and not project:
            raise ValueError(
                "W&B project name must be provided when wandb.enabled is true."
            )
        entity = data.get("entity")
        entity = None if entity in (None, "", "null") else str(entity)
        return cls(
            enabled=enabled,
            project=None if project is None else str(project),
            entity=entity,
            notes=str(data.get("notes", "")),
            name=data.get("name"),
            log_every_steps=_to_int(
                data.get("log_every_steps", 10), "wandb.log_every_steps"
            ),
        )


@dataclass
class Config:
    out_dir: str
    model: ModelConfig
    training: TrainingConfig
    wandb: WandbConfig
    test_run: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        _require_keys(data, ("out_dir", "model", "training", "wandb"))
        model = ModelConfig.from_dict(data["model"])
        training = TrainingConfig.from_dict(data["training"])
        wandb_cfg = WandbConfig.from_dict(data.get("wandb"))
        test_run = bool(data.get("test_run", False))
        return cls(
            out_dir=str(data["out_dir"]),
            model=model,
            training=training,
            wandb=wandb_cfg,
            test_run=test_run,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Use key=value syntax.")
        path, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        keys = path.split(".")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(
                    f"Cannot apply override '{override}': '{key}' is missing or not a mapping."
                )
            target = target[key]
        target[keys[-1]] = value


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Config:
    with open(path, "r", encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream) or {}
    if overrides:
        _apply_overrides(raw_config, overrides)
    return Config.from_dict(raw_config)
