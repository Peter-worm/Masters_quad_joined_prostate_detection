"""Test-set metrics and visualizations for segmentation experiments."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage
from skimage.measure import marching_cubes

from src.config import ExperimentConfig
from src.data.dataset import ProstateLesionDataset
from src.models.architectures import AnisotropicUNet3D

LOGGER = logging.getLogger("src.utils.evaluation")


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1))


def segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    """Calculate case-level overlap, volume, and surface metrics."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    dice = (2.0 * intersection) / (prediction.sum() + target.sum()) if prediction.any() or target.any() else 1.0
    precision = intersection / prediction.sum() if prediction.any() else float(target.sum() == 0)
    recall = intersection / target.sum() if target.any() else float(prediction.sum() == 0)
    target_volume = target.sum() * float(np.prod(spacing))
    predicted_volume = prediction.sum() * float(np.prod(spacing))

    pred_surface, target_surface = _surface(prediction), _surface(target)
    if pred_surface.any() and target_surface.any():
        target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
        pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
        pred_to_target = target_distance[pred_surface]
        target_to_pred = pred_distance[target_surface]
        all_distances = np.concatenate((pred_to_target, target_to_pred))
        hd95 = float(np.percentile(all_distances, 95))
        assd = float(all_distances.mean())
    elif prediction.any() or target.any():
        hd95 = float("inf")
        assd = float("inf")
    else:
        hd95 = assd = 0.0

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "hd95_mm": hd95,
        "assd_mm": assd,
        "target_volume_mm3": float(target_volume),
        "prediction_volume_mm3": float(predicted_volume),
        "relative_volume_error": float((predicted_volume - target_volume) / target_volume)
        if target_volume
        else float(predicted_volume > 0),
        "false_positive_voxels": float(np.logical_and(prediction, ~target).sum()),
        "false_negative_voxels": float(np.logical_and(target, ~prediction).sum()),
        "ground_truth_has_lesion": int(target.any()),
        "prediction_has_lesion": int(prediction.any()),
    }


def patient_detection_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Calculate patient-level lesion detection metrics from case rows."""
    true_positive = sum(
        row["ground_truth_has_lesion"] == 1 and row["prediction_has_lesion"] == 1
        for row in rows
    )
    false_positive = sum(
        row["ground_truth_has_lesion"] == 0 and row["prediction_has_lesion"] == 1
        for row in rows
    )
    true_negative = sum(
        row["ground_truth_has_lesion"] == 0 and row["prediction_has_lesion"] == 0
        for row in rows
    )
    false_negative = sum(
        row["ground_truth_has_lesion"] == 1 and row["prediction_has_lesion"] == 0
        for row in rows
    )
    positive_cases = true_positive + false_negative
    negative_cases = true_negative + false_positive
    total_cases = len(rows)
    return {
        "total_patients": total_cases,
        "lesion_positive_patients": positive_cases,
        "lesion_negative_patients": negative_cases,
        "true_positive_patients": true_positive,
        "false_positive_patients": false_positive,
        "true_negative_patients": true_negative,
        "false_negative_patients": false_negative,
        "patient_sensitivity": true_positive / positive_cases if positive_cases else float("nan"),
        "patient_specificity": true_negative / negative_cases if negative_cases else float("nan"),
        "patient_accuracy": (true_positive + true_negative) / total_cases
        if total_cases
        else float("nan"),
    }


def _load_model(config: ExperimentConfig, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = AnisotropicUNet3D(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_filters=config.base_filters,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def _prediction(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(sample.unsqueeze(0).to(device))
    return (torch.sigmoid(logits[0, 0]).cpu().numpy() > threshold)


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        return np.zeros_like(image, dtype=float)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def show_prediction_viewer(
    config: ExperimentConfig,
    checkpoint: Path,
    test_files: Iterable[Path],
    device: torch.device,
) -> None:
    """Display an interactive multi-slice viewer for test-set predictions.

    The model is run once when the viewer is created. Each displayed slice
    contains the input T2 image, ground truth, prediction, and an overlay.
    Ground truth is green and prediction is red; overlapping regions appear
    yellow.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as error:
        raise ImportError(
            "The interactive viewer requires ipywidgets and IPython. "
            "Install the notebook dependencies before calling show_prediction_viewer."
        ) from error

    model = _load_model(config, checkpoint, device)
    dataset = ProstateLesionDataset(list(test_files), is_train=False)
    cases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]] = {}
    for index, file_path in enumerate(dataset.pickle_files):
        inputs, target_tensor = dataset[index]
        target = target_tensor[0].numpy().astype(bool)
        prediction = _prediction(model, inputs, device, config.evaluation_threshold)
        metrics = segmentation_metrics(prediction, target, config.target_spacing)
        cases[Path(file_path).stem] = (
            inputs.numpy(),
            target,
            prediction,
            metrics,
        )

    if not cases:
        raise ValueError("No test volumes were provided to show_prediction_viewer.")

    orientations = {"Axial": 0, "Coronal": 1, "Sagittal": 2}
    background_channels = {"ADC": 0, "DWI": 1, "T2": 2, "Anatomy": 3}
    patient_dropdown = widgets.Dropdown(
        options=sorted(cases),
        description="Case:",
        layout=widgets.Layout(width="300px"),
    )
    orientation_dropdown = widgets.Dropdown(
        options=list(orientations),
        value="Axial",
        description="View:",
    )
    background_dropdown = widgets.Dropdown(
        options=list(background_channels),
        value="T2",
        description="Background:",
    )
    slice_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=cases[patient_dropdown.value][0][
            background_channels[background_dropdown.value]
        ].shape[0]
        - 1,
        step=1,
        description="Slice:",
        continuous_update=False,
        layout=widgets.Layout(width="450px"),
    )
    overlay_focus = widgets.Checkbox(
        value=False,
        description="Focus on overlay",
        indent=False,
    )
    output = widgets.Output()

    def update_slice_range(*_change: object) -> None:
        axis = orientations[orientation_dropdown.value]
        channel = background_channels[background_dropdown.value]
        slice_slider.max = cases[patient_dropdown.value][0][channel].shape[axis] - 1
        slice_slider.value = min(slice_slider.value, slice_slider.max)

    def render(*_change: object) -> None:
        inputs, target, prediction, metrics = cases[patient_dropdown.value]
        axis = orientations[orientation_dropdown.value]
        image = inputs[background_channels[background_dropdown.value]]
        image_slice = np.take(image, slice_slider.value, axis=axis)
        target_slice = np.take(target, slice_slider.value, axis=axis)
        prediction_slice = np.take(prediction, slice_slider.value, axis=axis)
        overlay = np.zeros((*target_slice.shape, 3), dtype=float)
        overlay[target_slice, 1] = 1.0
        overlay[prediction_slice, 0] = 1.0

        with output:
            output.clear_output(wait=True)
            if overlay_focus.value:
                figure, axis_plot = plt.subplots(figsize=(8, 8))
                axis_plot.imshow(_normalize_for_display(image_slice), cmap="gray")
                axis_plot.imshow(overlay, alpha=0.7)
                axis_plot.set_title("Ground truth + prediction overlay")
                axis_plot.axis("off")
            else:
                figure, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(_normalize_for_display(image_slice), cmap="gray")
                axes[1].imshow(target_slice, cmap="Greens", vmin=0, vmax=1)
                axes[2].imshow(prediction_slice, cmap="Reds", vmin=0, vmax=1)
                axes[3].imshow(_normalize_for_display(image_slice), cmap="gray")
                axes[3].imshow(overlay, alpha=0.55)
                for axis_plot, title in zip(
                    axes,
                    (
                        f"Input {background_dropdown.value}",
                        "Ground truth",
                        "Prediction",
                        "Overlay",
                    ),
                ):
                    axis_plot.set_title(title)
                    axis_plot.axis("off")
            figure.suptitle(
                f"{patient_dropdown.value} | {orientation_dropdown.value} "
                f"slice {slice_slider.value} | Dice {metrics['dice']:.3f}"
            )
            figure.tight_layout()
            display(figure)
            plt.close(figure)

    patient_dropdown.observe(update_slice_range, names="value")
    orientation_dropdown.observe(update_slice_range, names="value")
    patient_dropdown.observe(render, names="value")
    orientation_dropdown.observe(render, names="value")
    background_dropdown.observe(render, names="value")
    slice_slider.observe(render, names="value")
    overlay_focus.observe(render, names="value")
    update_slice_range()
    display(
        widgets.VBox(
            [
                widgets.HBox(
                    [patient_dropdown, orientation_dropdown, background_dropdown, overlay_focus]
                ),
                slice_slider,
                output,
            ]
        )
    )
    render()


def show_prediction_3d_viewer(
    config: ExperimentConfig,
    checkpoint: Path,
    test_files: Iterable[Path],
    device: torch.device,
) -> None:
    """Display ground-truth and predicted lesions as interactive 3D meshes."""
    try:
        import ipywidgets as widgets
        import plotly.graph_objects as go
        from IPython.display import display
        from preprocessing.core.visualization_lib import mask_to_plotly_mesh
    except ImportError as error:
        raise ImportError(
            "The 3D viewer requires ipywidgets, plotly, and scikit-image."
        ) from error

    model = _load_model(config, checkpoint, device)
    dataset = ProstateLesionDataset(list(test_files), is_train=False)
    cases: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, file_path in enumerate(dataset.pickle_files):
        inputs, target_tensor = dataset[index]
        target = target_tensor[0].numpy().astype(bool)
        prediction = _prediction(model, inputs, device, config.evaluation_threshold)
        cases[Path(file_path).stem] = (target, prediction)

    if not cases:
        raise ValueError("No test volumes were provided to show_prediction_3d_viewer.")

    case_dropdown = widgets.Dropdown(
        options=sorted(cases),
        description="Case:",
        layout=widgets.Layout(width="300px"),
    )
    axial_scale = widgets.FloatSlider(
        value=1.0,
        min=1.0,
        max=5.0,
        step=0.25,
        description="Axial scale:",
        continuous_update=False,
        readout_format=".2f",
        layout=widgets.Layout(width="400px"),
    )
    show_ground_truth = widgets.Checkbox(
        value=True,
        description="Ground truth",
        indent=False,
    )
    show_prediction = widgets.Checkbox(
        value=True,
        description="Prediction",
        indent=False,
    )
    output = widgets.Output()

    def render(*_change: object) -> None:
        target, prediction = cases[case_dropdown.value]
        axis_scale = (axial_scale.value, 1.0, 1.0)
        array_spacing = (
            config.target_spacing[2],
            config.target_spacing[1],
            config.target_spacing[0],
        )
        candidate_traces = (
            (
                mask_to_plotly_mesh(
                    target,
                    array_spacing,
                    color="green",
                    name="Ground truth",
                    opacity=0.55,
                    axis_scale=axis_scale,
                ),
                show_ground_truth.value,
            ),
            (
                mask_to_plotly_mesh(
                    prediction,
                    array_spacing,
                    color="red",
                    name="Prediction",
                    opacity=0.55,
                    axis_scale=axis_scale,
                ),
                show_prediction.value,
            ),
        )
        traces = [
            trace
            for trace, visible in candidate_traces
            if trace is not None and visible
        ]
        figure = go.Figure(data=traces)
        figure.update_layout(
            title=f"{case_dropdown.value} | 3D lesion comparison",
            width=900,
            height=700,
            scene=dict(
                aspectmode="data",
                xaxis_title="Sagittal (mm)",
                yaxis_title="Coronal (mm)",
                zaxis_title="Axial (mm)",
            ),
            legend=dict(
                title="Lesion masks",
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
        with output:
            output.clear_output(wait=True)
            display(figure)

    case_dropdown.observe(render, names="value")
    axial_scale.observe(render, names="value")
    show_ground_truth.observe(render, names="value")
    show_prediction.observe(render, names="value")
    display(
        widgets.VBox(
            [
                widgets.HBox([case_dropdown, axial_scale]),
                widgets.HBox([show_ground_truth, show_prediction]),
                output,
            ]
        )
    )
    render()


def _save_case_figure(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    case_id: str,
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    views = ((0, "axial"), (1, "coronal"), (2, "sagittal"))
    figure, axes = plt.subplots(3, 4, figsize=(14, 10))
    for row, (axis, name) in enumerate(views):
        extent = np.any(target | prediction, axis=tuple(i for i in range(3) if i != axis))
        slice_index = int(np.where(extent)[0][len(np.where(extent)[0]) // 2]) if extent.any() else image.shape[axis] // 2
        image_slice = np.take(image, slice_index, axis=axis)
        target_slice = np.take(target, slice_index, axis=axis)
        prediction_slice = np.take(prediction, slice_index, axis=axis)
        overlay = np.zeros((*target_slice.shape, 3), dtype=float)
        overlay[target_slice, 1] = 1.0
        overlay[prediction_slice, 0] = 1.0
        axes[row, 0].imshow(image_slice, cmap="gray")
        axes[row, 1].imshow(target_slice, cmap="Greens", vmin=0, vmax=1)
        axes[row, 2].imshow(prediction_slice, cmap="Reds", vmin=0, vmax=1)
        axes[row, 3].imshow(image_slice, cmap="gray")
        axes[row, 3].imshow(overlay, alpha=0.55)
        axes[row, 0].set_ylabel(name)
        for column in range(4):
            axes[row, column].axis("off")
    for axis, title in zip(axes[0], ("Input T2", "Ground truth", "Prediction", "Overlay")):
        axis.set_title(title)
    figure.suptitle(f"{case_id} | Dice {metrics['dice']:.3f} | HD95 {metrics['hd95_mm']:.2f} mm")
    figure.tight_layout()
    figure.savefig(output_dir / f"{case_id}_views.png", dpi=180)
    plt.close(figure)


def _save_surface_figure(
    target: np.ndarray,
    prediction: np.ndarray,
    case_id: str,
    spacing: tuple[float, float, float],
    output_dir: Path,
) -> None:
    if not prediction.any() or not target.any():
        return
    vertices, faces, _, _ = marching_cubes(prediction.astype(np.float32), level=0.5)
    target_surface = _surface(target)
    target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    errors = target_distance[
        tuple(np.clip(np.rint(vertices[:, axis]).astype(int), 0, target.shape[axis] - 1) for axis in range(3))
    ]
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    mesh = axis.plot_trisurf(
        vertices[:, 2] * spacing[2],
        vertices[:, 1] * spacing[1],
        vertices[:, 0] * spacing[0],
        triangles=faces,
        cmap="turbo",
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    mesh.set_array(errors)
    mesh.set_clim(0, max(float(np.percentile(errors, 95)), 1.0))
    figure.colorbar(mesh, ax=axis, label="Distance to ground-truth surface (mm)")
    axis.set(xlabel="Sagittal (mm)", ylabel="Coronal (mm)", zlabel="Axial (mm)")
    axis.set_title(f"{case_id} predicted surface error")
    figure.tight_layout()
    figure.savefig(output_dir / f"{case_id}_surface_error.png", dpi=180)
    plt.close(figure)


def _save_summary_plot(rows: list[dict[str, float]], output_dir: Path) -> None:
    names = ("dice", "hd95_mm", "assd_mm", "precision", "recall")
    figure, axes = plt.subplots(1, len(names), figsize=(16, 4))
    for axis, name in zip(axes, names):
        values = [row[name] for row in rows if np.isfinite(row[name])]
        axis.boxplot(values, tick_labels=[name])
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "test_metric_distributions.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    target = [row["target_volume_mm3"] for row in rows]
    prediction = [row["prediction_volume_mm3"] for row in rows]
    axis.scatter(target, prediction, alpha=0.8)
    limit = max(target + prediction) if target else 1.0
    axis.plot((0, limit), (0, limit), "--", color="black")
    axis.set(xlabel="Ground-truth volume (mm3)", ylabel="Predicted volume (mm3)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "volume_agreement.png", dpi=180)
    plt.close(figure)


def generate_evaluation_artifacts(
    config: ExperimentConfig,
    checkpoint: Path,
    test_files: Iterable[Path],
    device: torch.device,
) -> Path:
    """Evaluate each test volume and write CSV metrics plus diagnostic figures."""
    output_dir = config.evaluation_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(config, checkpoint, device)
    dataset = ProstateLesionDataset(list(test_files), is_train=False)
    rows: list[dict[str, float]] = []
    cases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for index, file_path in enumerate(dataset.pickle_files):
        inputs, target_tensor = dataset[index]
        target = target_tensor[0].numpy().astype(bool)
        prediction = _prediction(model, inputs, device, config.evaluation_threshold)
        metrics = segmentation_metrics(prediction, target, config.target_spacing)
        row = {"case_id": Path(file_path).stem, **metrics}
        rows.append(row)
        cases[row["case_id"]] = (inputs[2].numpy(), target, prediction)

    csv_path = output_dir / "test_case_metrics.csv"
    fieldnames = list(rows[0]) if rows else ["case_id"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        detection_path = output_dir / "patient_detection_metrics.csv"
        detection_metrics = patient_detection_metrics(rows)
        with detection_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("metric", "value"))
            writer.writerows(detection_metrics.items())
        ranked = sorted(rows, key=lambda row: row["dice"])
        selected = {ranked[0]["case_id"], ranked[len(ranked) // 2]["case_id"], ranked[-1]["case_id"]}
        for row in rows:
            if row["case_id"] in selected:
                image, target, prediction = cases[row["case_id"]]
                _save_case_figure(image, target, prediction, row["case_id"], row, output_dir)
                _save_surface_figure(
                    target,
                    prediction,
                    row["case_id"],
                    config.target_spacing,
                    output_dir,
                )
        _save_summary_plot(rows, output_dir)
    LOGGER.info("Saved case-level test metrics and figures to %s", output_dir)
    return csv_path
