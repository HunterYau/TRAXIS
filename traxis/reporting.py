"""Headless scientific figures from immutable, completed replay results."""

import csv
import html
import io
import os
from pathlib import Path
import time

import numpy as np

from traxis.evaluation_artifacts import load_case, load_evaluation
from traxis.metrics import resample_cell_map
from traxis.resources import ResourceGuard
from traxis.storage import atomic_write, canonical_hash, sha256_file, write_json


def _matched_runs(reports: list[dict]):
    """Require common data/protocol before drawing comparisons in one figure."""
    fields = ("dataset_fingerprint", "split", "budgets", "initial_actions", "threshold_mm")
    first = reports[0]["plan"]
    actions = {}
    for report in reports:
        if any(report["plan"][key] != first[key] for key in fields):
            raise ValueError("comparison runs require the same dataset, split, budgets, initial actions and threshold")
        for episode in report["episodes"]:
            if episode["policy"]["kind"] not in ("fixed", "random"):
                continue
            for row in episode["rows"]:
                key = (episode["specimen_id"], canonical_hash(episode["policy"]), row["records"])
                if actions.setdefault(key, row["action_indices"]) != row["action_indices"]:
                    raise ValueError("matched fixed/random comparisons used different observation sets")


def _figure_api(cache: Path):
    # Matplotlib otherwise writes its font cache outside the project on Windows.
    cache.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(cache.resolve())
    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError("install requirements-plots.txt for headless reports") from exc
    finally:
        if previous is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous
    return matplotlib, Figure, FigureCanvasAgg


def _new_figure(api, size):
    _, Figure, Canvas = api
    figure = Figure(figsize=size, layout="constrained")
    Canvas(figure)
    return figure


def _save_figure(figure, output: Path, stem: str, guard):
    paths = []
    try:
        for suffix in ("png", "svg"):
            if guard:
                guard.check()
            path = output / f"{stem}.{suffix}"
            atomic_write(path, lambda stream: figure.savefig(stream, format=suffix, dpi=160))
            paths.append(path.name)
    finally:
        figure.clear()
    return paths


def _predictor_label(descriptor: dict) -> str:
    if descriptor["kind"] == "das":
        return "DAS (calibrated)"
    config = descriptor["config"]
    name = "A" if config["architecture"] == "direct" else "B"
    label = f"{name}, {config['output_size']} px, extra {config['extra_decoder_blocks']}"
    if descriptor["kind"] == "ensemble":
        label += f", {len(descriptor['members'])} members"
    elif descriptor.get("seed") is not None:
        label += f", seed {descriptor['seed']}"
    return label


def _curves(api, reports, labels, output, guard):
    series_count = sum(len({(r["predictor"], r["policy"]) for r in report["summary"]}) for report in reports)
    if series_count > 24:
        raise ValueError("at most 24 predictor/policy curves can share one report")
    figure = _new_figure(api, (12, 8 + .16 * series_count))
    layout = figure.add_gridspec(4, 2, height_ratios=(1, 1, max(.16, .055 * ((series_count + 1) // 2)), .14))
    axes = [figure.add_subplot(layout[row, col]) for row in range(2) for col in range(2)]
    metrics = (("mae_mm", "Whole-plate MAE (mm)"), ("defect_region_mae_mm", "Defect-region MAE (mm)"),
               ("dice", "Footprint Dice"), ("absolute_excess_mae_mm", "MAE minus dense-reference MAE (mm)"))
    handles, legends = [], []
    series_index = 0
    for report, label in zip(reports, labels):
        series = {}
        for row in report["summary"]:
            series.setdefault((row["predictor"], row["policy"]), []).append(row)
        for (predictor, policy), rows in sorted(series.items()):
            rows.sort(key=lambda r: r["records"])
            name = f"{label}: {_predictor_label(report['predictors'][predictor])}; {policy}"
            handle = None
            color = api[0].colormaps["tab20"](series_index % 20)
            style = "-" if series_index < 20 else "--"
            series_index += 1
            for axis, (metric, ylabel) in zip(axes, metrics):
                selected = [row for row in rows if row["metrics"][metric]["mean"] is not None]
                if not selected:
                    continue
                y = [r["metrics"][metric]["mean"] for r in selected]
                sd = [r["metrics"][metric]["between_parent_std"] for r in selected]
                line = axis.errorbar([r["records"] for r in selected], y,
                                     yerr=[v if v is not None else np.nan for v in sd] if any(v is not None for v in sd) else None,
                                     marker="o", markersize=3, capsize=2, linewidth=1.2, label=name, color=color, linestyle=style)
                handle = handle or line
            if handle is not None:
                handles.append(handle)
                legends.append(name)
    for axis, (_, ylabel) in zip(axes, metrics):
        axis.set(xlabel="Acquired records", ylabel=ylabel)
        axis.grid(alpha=.2)
    lower, upper = axes[2].get_ylim()
    axes[2].set_ylim(min(0, lower), max(1, upper))
    axes[3].axhline(0, color="0.35", linewidth=.7, linestyle="--")
    debug = any(r["debug_only"] for r in reports)
    figure.suptitle(("DEBUG FIXTURES — " if debug else "") + "Frozen offline replay comparisons", fontsize=13)
    legend_axis = figure.add_subplot(layout[2, :])
    legend_axis.set_axis_off()
    legend_axis.legend(handles, legends, loc="center", fontsize=7, ncols=2)
    note_axis = figure.add_subplot(layout[3, :])
    note_axis.set_axis_off()
    note_axis.text(.5, .5, "Means across independent parents; whiskers are between-parent SD, not confidence intervals.\n"
                          "Scans/random seeds are averaged within parents. Dense predictions are model references, not truth.",
                   fontsize=8, ha="center", va="center")
    return _save_figure(figure, output, "error-versus-records", guard)


def _surface_grid(values, limit=64):
    shape = tuple(max(n for n in range(1, min(limit, size) + 1) if size % n == 0) for size in values.shape)
    return resample_cell_map(values, shape)


def _maps(api, plate, maps, title, output, stem, guard):
    nominal = plate.thickness_m * 1000
    truth = maps["truth"]
    prediction = resample_cell_map(maps["prediction"], truth.shape)
    panels = [(truth, "Known loss", "viridis", "Loss depth (mm)"),
              (prediction, "Predicted loss", "viridis", "Loss depth (mm)"),
              (abs(prediction - truth), "Absolute error", "magma", "Absolute error (mm)")]
    if "std" in maps:
        panels.append((resample_cell_map(maps["std"], truth.shape), "Ensemble SD (uncalibrated)", "cividis", "Ensemble SD (mm)"))
    figure = _new_figure(api, (3.7 * len(panels), 4.4))
    axes = figure.subplots(1, len(panels))
    extent = (0, plate.width_m * 1000, 0, plate.height_m * 1000)
    for axis, (values, name, cmap, unit) in zip(axes, panels):
        # Nearest-cell display preserves piecewise-constant prediction resolution.
        vmax = max(nominal, float(values.max())) if name.startswith("Ensemble") else nominal
        im = axis.imshow(values, extent=extent, origin="lower", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
        axis.set(title=name, xlabel="x (mm)", ylabel="y (mm)")
        figure.colorbar(im, ax=axis, orientation="horizontal", shrink=.9, label=unit)
    figure.suptitle(title, fontsize=11)
    figure.supxlabel("Depth is missing aluminum; remaining thickness = nominal thickness − loss depth. Shared fixed depth/error scales.", fontsize=8)
    paths = _save_figure(figure, output, stem + "-maps", guard)
    figure = _new_figure(api, (11, 4.7))
    for index, (values, name) in enumerate(((truth, "Known loss"), (maps["prediction"], "Predicted loss")), 1):
        values = _surface_grid(values)
        y, x = np.meshgrid((np.arange(values.shape[0]) + .5) / values.shape[0] * extent[3],
                           (np.arange(values.shape[1]) + .5) / values.shape[1] * extent[1], indexing="ij")
        axis = figure.add_subplot(1, 2, index, projection="3d")
        axis.plot_surface(x, y, values, cmap="viridis", vmin=0, vmax=nominal,
                          rcount=values.shape[0], ccount=values.shape[1], linewidth=0, antialiased=False)
        axis.set(title=name, xlabel="x (mm)", ylabel="y (mm)", zlabel="Loss depth (mm)",
                 xlim=extent[:2], ylim=extent[2:], zlim=(0, nominal))
        axis.set_box_aspect((1, plate.height_m / plate.width_m, .45))
        axis.view_init(elev=28, azim=-55)
    figure.suptitle(title, fontsize=11)
    figure.supxlabel("2.5D loss surfaces; area-averaged display at most 64 × 64 cells. Vertical scale exaggerated; use the 2D maps for edges.", fontsize=8)
    return paths + _save_figure(figure, output, stem + "-surfaces", guard)


def _write_csv(path, rows):
    if not rows:
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, lambda output: output.write(stream.getvalue().encode("utf-8")))


def create_report(evaluations: list[Path], output: Path, *, labels: list[str] | None = None,
                  case_count: int = 1, case_records: tuple[int, ...] | None = None,
                  cases: tuple[tuple[int, int], ...] | None = None,
                  guard: ResourceGuard | None = None) -> dict:
    """Render measured results only; never train, select a best case or refit metrics."""
    from traxis.evaluation import summarize_episodes
    if not evaluations or len(evaluations) > 8 or type(case_count) is not int or not 0 <= case_count <= 4:
        raise ValueError("reports need 1–8 evaluations and 0–4 explicitly ordered cases per evaluation")
    if case_records is not None and (not case_records or len(case_records) > 8 or any(type(n) is not int or n < 0 for n in case_records)):
        raise ValueError("case records must contain 1–8 nonnegative budgets")
    labels = labels if labels is not None else [f"run-{i}" for i in range(len(evaluations))]
    if len(labels) != len(evaluations) or len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        raise ValueError("provide a distinct nonempty label per evaluation")
    reports = [load_evaluation(path) for path in evaluations]
    _matched_runs(reports)
    selected_cases = {(run, case) for run, report in enumerate(reports)
                      for case in range(min(case_count, len(report["episodes"])))}
    if cases is not None:
        if not cases or len(cases) > 8 or any(len(pair) != 2 or any(type(n) is not int or n < 0 for n in pair) for pair in cases):
            raise ValueError("explicit cases must contain 1–8 pairs of nonnegative run and episode indices")
        if any(run >= len(reports) or case >= len(reports[run]["episodes"]) for run, case in cases):
            raise ValueError("explicit case index is outside the saved evaluations")
        selected_cases = set(cases)
    for report in reports:
        # Derive figures from the episode rows rather than trusting a stale summary.
        if canonical_hash(summarize_episodes(report["episodes"])) != canonical_hash(report["summary"]):
            raise ValueError("evaluation summary disagrees with episode rows")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("report output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if guard:
        guard.check(force=True)
    api = _figure_api(output / ".matplotlib-cache")
    manifest = {"kind": "scientific_replay_report", "passed": False,
                "debug_only": any(r["debug_only"] for r in reports), "files": [], "selected_cases": [],
                "source_sha256": sha256_file(Path(__file__)), "matplotlib_version": api[0].__version__,
                "sources": [{"label": label, "evaluation_sha256": sha256_file(Path(path) / "evaluation.json"),
                             "plan_fingerprint": report["plan_fingerprint"]} for label, path, report in zip(labels, evaluations, reports)],
                "note": "Offline replay only. Spread is uncalibrated. Cases use explicit file order, not ranking by error."}
    try:
        manifest["files"] += _curves(api, reports, labels, output, guard)
        summaries, parents, costs = [], [], []
        for run_index, (label, directory, report) in enumerate(zip(labels, evaluations, reports)):
            for row in report["summary"]:
                for metric, values in row["metrics"].items():
                    summaries.append({"run": label, "predictor": row["predictor"], "policy": row["policy"],
                                      "records": row["records"], "metric": metric, **values})
                for parent, scores in row["per_parent"].items():
                    parents.append({"run": label, "predictor": row["predictor"], "policy": row["policy"],
                                    "records": row["records"], "parent_specimen_id": parent, **scores})
            for episode_index, episode in enumerate(report["episodes"]):
                descriptor = report["predictors"][episode["predictor"]]
                for row in episode["rows"]:
                    costs.append({"run": label, "case": episode_index, "specimen_id": episode["specimen_id"],
                                  "predictor": _predictor_label(descriptor), "policy": canonical_hash(episode["policy"]),
                                  "records": row["records"], "neural_parameters": descriptor["neural_parameters"],
                                  "inference_device": descriptor["device"], "inference_seconds": row["inference_seconds"],
                                  "inference_calls_cumulative": row["inference_calls_cumulative"],
                                  "inference_seconds_cumulative": row["reconstruction_seconds_cumulative"],
                                  "selector_seconds_cumulative": row["selector_seconds_cumulative"],
                                  "unique_excitation_requests": row["unique_excitation_requests"],
                                  "hardware_acquisition_seconds": episode["hardware_acquisition_seconds"]})
                if (run_index, episode_index) not in selected_cases:
                    continue
                budgets = case_records if case_records is not None else tuple(sorted({episode["rows"][0]["records"], episode["rows"][-1]["records"]}))
                for records in budgets:
                    if guard:
                        guard.check()
                    plate, maps = load_case(directory, episode, episode_index, records)
                    title = ("DEBUG FIXTURE | " if report["debug_only"] else "") + f"{label}: {_predictor_label(descriptor)} | {episode['specimen_id']} | {records} records"
                    stem = f"run-{run_index}-case-{episode_index:05d}-records-{records}"
                    manifest["files"] += _maps(api, plate, maps, title, output, stem, guard)
                    manifest["selected_cases"].append({"run": label, "case": episode_index, "records": records,
                                                       "domain": episode["domain"], "split": episode["split"],
                                                       "data_validation_status": episode["data_validation_status"],
                                                       "native_prediction_shape": list(maps["prediction"].shape),
                                                       "nominal_thickness_mm": plate.thickness_m * 1000})
        for name, rows in (("summary.csv", summaries), ("per-parent.csv", parents), ("computation.csv", costs)):
            _write_csv(output / name, rows)
            manifest["files"].append(name)
        title = "DEBUG FIXTURES — scientific replay report" if manifest["debug_only"] else "Scientific replay report"
        body = ["<!doctype html><meta charset='utf-8'><title>" + html.escape(title) + "</title>",
                "<style>body{font:16px system-ui;max-width:1300px;margin:2em auto;padding:0 1em}img{max-width:100%}p{line-height:1.5}</style>",
                "<h1>" + html.escape(title) + "</h1>",
                "<p>Source evaluations are frozen and fingerprinted. Parent-level means average repeated scans and random seeds within each plate. Whiskers show between-parent standard deviation, not confidence intervals.</p>",
                "<p>Depth is missing aluminum in millimeters. Dense reconstructions are model references; geometric labels provide truth. Ensemble standard deviation is uncalibrated. Compute timings include preprocessing and model inference, are single measurements without a benchmark warm-up, and do not measure hardware acquisition time.</p>",
                "<p>Cases follow saved episode order; they were not selected for low error. Surface displays average to at most 64 cells per axis and exaggerate the vertical scale.</p>"]
        for name in manifest["files"]:
            escaped = html.escape(name, quote=True)
            if name.endswith(".png"):
                body.append(f"<p><a href='{escaped[:-4]}.svg'>Vector figure</a><br><img src='{escaped}' alt='{escaped}' loading='lazy'></p>")
            elif name.endswith(".csv"):
                body.append(f"<p><a href='{escaped}'>{escaped}</a></p>")
        atomic_write(output / "index.html", lambda stream: stream.write("\n".join(body).encode("utf-8")))
        manifest["files"].append("index.html")
        if guard:
            guard.check(force=True)
        manifest["file_sha256"] = {name: sha256_file(output / name) for name in manifest["files"]}
        manifest["passed"] = True
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest["resources"] = guard.report() if guard else None
        write_json(output / "report.json", manifest)
    return manifest
