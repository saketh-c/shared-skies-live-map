"""Shared loader: import the untouched production training module
(pipeline/03_train_enhanced.py) from the staged repo. The module-level code
runs on import (feature-list assembly, AQ-feature gating on the cell parquets),
exactly as when the script is executed directly.
"""
import importlib.util
import os
import sys

REPO = os.path.expanduser("~/scratch/livemap_retrain/repo")


def load_training_module():
    path = os.path.join(REPO, "pipeline", "03_train_enhanced.py")
    sys.path.insert(0, os.path.join(REPO, "pipeline"))
    spec = importlib.util.spec_from_file_location("train_enhanced", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
