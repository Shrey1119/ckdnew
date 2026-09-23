"""Prediction Service Layer.

Unified business logic layer for clinical tabular predictions,
CT scan image classifications, multimodal fusion, and model comparisons.
"""

from app.services.predict_service import (
    get_image_prediction_probabilities,
    run_tabular_prediction,
    run_image_prediction,
    run_fusion_prediction,
    run_multimodal_prediction,
    get_model_comparison_metrics,
    run_optimized_prediction
)

__all__ = [
    "get_image_prediction_probabilities",
    "run_tabular_prediction",
    "run_image_prediction",
    "run_fusion_prediction",
    "run_multimodal_prediction",
    "get_model_comparison_metrics",
    "run_optimized_prediction"
]
