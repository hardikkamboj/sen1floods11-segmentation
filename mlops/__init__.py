"""MLOps scaffolding for the Sen1Floods11 cascaded flood-segmentation pipeline.

Modules
-------
clearml_tracking            Light wrapper for ClearML experiment tracking
                            (used by ``train_segformer.py``).
calibrate_ambiguity_band    Derives the empirical ambiguity band (Figure A).
benchmark_cascade           Cascade efficiency benchmark (Figure B + system table).
cascaded_inference_pipeline ClearML Pipeline running cascade on a full scene.
train_segformer             Reproducible, ClearML-tracked SegFormer training.
"""
