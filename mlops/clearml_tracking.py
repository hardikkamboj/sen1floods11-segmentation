"""ClearML experiment tracking helpers.

Three functions to drop into existing training notebooks/scripts:

    init_clearml_task(model_name, hparams)   # at the top of training
    log_epoch(epoch, metrics)                # inside the training loop
    register_model(weights_path, metadata)   # at the end of training

ClearML auto-magics handle most of the rest — Tasks transparently capture
stdout, plots (matplotlib), git state, package versions, hardware info,
and (for PyTorch) automatically log scalars logged via tensorboard / wandb
if those are present. We only need to be explicit about the things ClearML
can't infer: model name, hyperparameter dict, and final artifact location.

Example
-------
    >>> from mlops.clearml_tracking import init_clearml_task, log_epoch, register_model
    >>>
    >>> task = init_clearml_task(
    ...     model_name="segformer-mit-b2",
    ...     hparams={"lr": 6e-5, "batch_size": 16, "loss": "dice+bce"},
    ... )
    >>> for epoch in range(EPOCHS):
    ...     train_loss, train_iou = train_one_epoch(...)
    ...     val_loss,   val_iou   = validate(...)
    ...     log_epoch(epoch, {"train_loss": train_loss, "val_loss": val_loss,
    ...                       "train_iou": train_iou, "val_iou": val_iou})
    >>> register_model("/path/to/best.pt", {"test_iou": 0.668, "bolivia_iou": 0.683})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clearml import Task


# Module-level handle so notebook cells can call log_epoch / register_model
# without threading the task object through every call site.
_TASK: Task | None = None


def init_clearml_task(
    model_name: str,
    hparams: dict[str, Any],
    project: str = "Sen1Floods11/Training",
) -> Task:
    """Create a ClearML Task that captures the current run.

    Tasks auto-capture: git commit/diff, package list, console output,
    matplotlib figures, and any tensorboard/wandb scalars. We additionally
    register the hyperparameters explicitly so they show up in the
    Hyperparameters tab and become searchable across runs.
    """
    global _TASK
    _TASK = Task.init(
        project_name=project,
        task_name=model_name,
        task_type=Task.TaskTypes.training,
        # Defer auto-tracking of pytorch checkpoints until we explicitly
        # register the best one — otherwise every epoch's checkpoint gets
        # uploaded which blows up storage.
        auto_connect_frameworks={"pytorch": False, "matplotlib": True},
    )
    _TASK.connect(hparams, name="hparams")
    return _TASK


def log_epoch(epoch: int, metrics: dict[str, float]) -> None:
    """Log per-epoch scalars. Each (title, series) pair becomes a curve
    in the ClearML scalars dashboard."""
    if _TASK is None:
        return
    logger = _TASK.get_logger()
    for k, v in metrics.items():
        # Group train_* and val_* metrics into the same plot for visual diff.
        # e.g. ("train_iou", 0.55) -> title="iou", series="train"
        if "_" in k:
            split, name = k.split("_", 1)
            if split in ("train", "val", "test"):
                logger.report_scalar(title=name, series=split, value=v, iteration=epoch)
                continue
        logger.report_scalar(title=k, series="value", value=v, iteration=epoch)


def register_model(
    weights_path: str | Path,
    metadata: dict[str, Any] | None = None,
    name: str = "best_model",
) -> None:
    """Upload the best checkpoint as a ClearML OutputModel and tag it with
    final test/bolivia metrics so the model registry is queryable."""
    if _TASK is None:
        return
    weights_path = str(weights_path)
    _TASK.update_output_model(
        model_path=weights_path,
        name=name,
        tags=["sen1floods11", "flood-segmentation"],
    )
    if metadata:
        # Extra metadata becomes searchable on the ClearML model registry.
        _TASK.connect(metadata, name="final_metrics")
