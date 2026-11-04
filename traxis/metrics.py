"""Material-loss metrics on a shared physical pixel grid."""

import math

import numpy as np

from traxis.config import Plate, positive


def resample_cell_map(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Area-preserving cell averages for integer-factor aligned grids.

    Upsampling replicates piecewise-constant predictions. Downsampling takes
    block means. This avoids crediting finer display pixels as extra resolution.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or min(values.shape) < 1:
        raise ValueError("map must be a nonempty finite 2D array")
    if len(shape) != 2 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("target shape must be two positive integers")
    result = values.copy()
    for axis, target in enumerate(shape):
        source = result.shape[axis]
        if target >= source and target % source == 0:
            result = np.repeat(result, target // source, axis=axis)
        elif source > target and source % target == 0:
            if axis == 0:
                result = result.reshape(target, source // target, result.shape[1]).mean(axis=1)
            else:
                result = result.reshape(result.shape[0], target, source // target).mean(axis=2)
        else:
            raise ValueError("grids must be aligned integer refinements")
    return result


def depth_metrics(prediction_mm: np.ndarray, truth_mm: np.ndarray, plate: Plate,
                   threshold_mm: float = 0.1) -> dict:
    positive("defect threshold", threshold_mm)
    truth = np.asarray(truth_mm, dtype=float)
    pred = resample_cell_map(prediction_mm, truth.shape)
    if truth.ndim != 2 or not np.isfinite(truth).all():
        raise ValueError("truth must be a finite 2D grid")
    for a in (truth, pred):
        if np.any(a < 0) or np.any(a > plate.thickness_m * 1000):
            raise ValueError("depth must be between zero and nominal thickness")
    error = pred - truth
    truth_mask, pred_mask = truth >= threshold_mm, pred >= threshold_mm
    region = truth > 0
    n_true, n_pred = int(truth_mask.sum()), int(pred_mask.sum())
    intersection, union = int((truth_mask & pred_mask).sum()), int((truth_mask | pred_mask).sum())
    area = plate.width_m * plate.height_m * 1e6 / truth.size

    def centroid(mask):
        if not mask.any():
            return None
        yy, xx = np.nonzero(mask)
        return [(float(xx.mean()) + 0.5) / truth.shape[1] * plate.width_m * 1000,
                (float(yy.mean()) + 0.5) / truth.shape[0] * plate.height_m * 1000]

    true_xy, pred_xy = centroid(truth_mask), centroid(pred_mask)
    distance = math.dist(true_xy, pred_xy) if true_xy is not None and pred_xy is not None else None
    true_volume, pred_volume = float(truth.sum() * area), float(pred.sum() * area)
    return {"mae_mm": float(abs(error).mean()), "rmse_mm": float(np.sqrt((error**2).mean())),
            "defect_region_mae_mm": float(abs(error[region]).mean()) if region.any() else None,
            "intact_region_mae_mm": float(abs(error[~region]).mean()) if (~region).any() else None,
            "dice": 2 * intersection / (n_true + n_pred) if n_true + n_pred else 1.0,
            "iou": intersection / union if union else 1.0,
            "centroid_error_mm": distance, "truth_centroid_mm": true_xy, "predicted_centroid_mm": pred_xy,
            "missed_defect": bool(n_true and not n_pred), "truth_intact": not bool(region.any()),
            "footprint_area_error_mm2": (n_pred - n_true) * area,
            "equivalent_diameter_error_mm": 2 * (math.sqrt(n_pred * area / math.pi) - math.sqrt(n_true * area / math.pi)),
            "false_positive_area_mm2": int((pred_mask & ~truth_mask).sum()) * area,
            "peak_depth_error_mm": float(pred.max() - truth.max()),
            "true_region_mean_depth_bias_mm": float(error[region].mean()) if region.any() else None,
            "true_loss_volume_mm3": true_volume, "predicted_loss_volume_mm3": pred_volume,
            "loss_volume_error_mm3": pred_volume - true_volume,
            "relative_loss_volume_error": (pred_volume - true_volume) / true_volume if true_volume else None,
            "threshold_mm": threshold_mm, "reference_shape": list(truth.shape)}


def ensemble_statistics(predictions_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(predictions_mm, dtype=float)
    if values.ndim != 3 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite independent member maps")
    return values.mean(axis=0), values.std(axis=0, ddof=1)


def uncertainty_diagnostics(mean_mm: np.ndarray, std_mm: np.ndarray, truth_mm: np.ndarray) -> dict:
    mean, std, truth = [np.asarray(a, dtype=float) for a in (mean_mm, std_mm, truth_mm)]
    if mean.ndim != 2 or mean.shape != std.shape or mean.shape != truth.shape:
        raise ValueError("uncertainty, prediction and truth grids must match")
    if not all(np.isfinite(a).all() for a in (mean, std, truth)) or np.any(std < 0):
        raise ValueError("maps must be finite and spread nonnegative")
    error = abs(mean - truth)
    correlation = float(np.corrcoef(std.ravel(), error.ravel())[0, 1]) if std.std() > 0 and error.std() > 0 else None
    return {"mean_spread_mm": float(std.mean()), "spread_error_correlation": correlation,
            "coverage_1sigma": float((error <= std).mean()),
            "coverage_1_96sigma": float((error <= 1.96 * std).mean()),
            "zero_spread_fraction": float((std == 0).mean()),
            "note": "Ensemble spread is not calibrated confidence; Gaussian coverage is diagnostic only."}
