"""Per-occurrence power baselines, reusable plots, and lossless CSV tables.

Power is recalculated from saved RMS traces using the recording's multitaper
settings. All times accepted by this module are seconds except event_dict,
whose start/stop pairs are milliseconds. Plot times denote window CENTERS.
Existing stored spectra and event dictionaries are never modified.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd

from .processing import parse_animal_id, select_recordings_by_condition


_IDENTITY = ["recording", "subject", "condition", "event"]
_TRIAL = [*_IDENTITY, "trial_id", "event_index", "event_onset_s"]
_SPECTRAL = ["region", "frequency_hz", "relative_time_s"]


def load_event_baseline_collection(json_path, *, recording_names=None):
    """Load only RMS traces and metadata needed for event-baseline power.

    Uses the same saved collection JSON/recordings directory as
    load_lfp_collection, without loading coherence/Granger or recomputing a
    whole-recording Fourier transform. Returns a collection-like object usable
    by build_condition_dict, update_collection_events_from_behavior, and this
    module. Import processed behavior afterward with source_unit='seconds';
    legacy saved events may have inconsistent units, so they are not loaded.
    Optional recording_names are exact .rec names for a smaller analysis.
    """
    import h5py

    json_path = Path(json_path).resolve()
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if recording_names is not None and isinstance(recording_names, str):
        raise ValueError("recording_names must be a sequence of exact recording names.")
    requested = None if recording_names is None else set(recording_names)
    recordings = []
    for path in sorted((json_path.parent / "recordings").glob("*.h5")):
        with h5py.File(path, "r") as handle:
            attrs = handle["metadata"].attrs
            name = str(attrs["name"])
            if requested is not None and name not in requested:
                continue
            if "rms_traces" not in handle["data"]:
                raise ValueError(f"{path}: no saved rms_traces.")
            regions = {str(k): int(v) for k, v in handle["brain region dict"].attrs.items()}
            regions = dict(sorted(regions.items(), key=lambda item: item[1]))
            traces = handle["data"]["rms_traces"][:]
            excluded = [str(x) for x in attrs.get("excluded_regions", [])]
            for region in excluded:
                traces[:, regions[region]] = np.nan
            recordings.append(SimpleNamespace(
                name=name, subject=str(attrs["subject"]), event_dict={},
                rms_traces=traces, brain_region_dict=regions, excluded_regions=excluded,
                resample_rate=float(attrs["resample_rate"]), timewindow=float(attrs["window duration"]),
                timestep=float(attrs["window step"]), halfbandwidth=float(attrs["half bandwidth product"]),
                source_h5=str(path),
            ))
    if not recordings:
        raise ValueError("No matching saved H5 recordings found.")
    if requested is not None and requested != {rec.name for rec in recordings}:
        raise ValueError(f"Missing recordings: {sorted(requested - {rec.name for rec in recordings})}")
    return SimpleNamespace(recordings=recordings, recording_to_event_dict={},
                           brain_region_dict=recordings[0].brain_region_dict,
                           source_json=str(json_path), metadata=metadata.get("metadata", {}))


def _mean(values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _window_power(recording, starts, n_samples, region_indices, tapers):
    """One estimate per window, averaging tapers only (never windows/trials)."""
    from spectral_connectivity import Connectivity, Multitaper

    epochs = np.full((n_samples, len(starts), len(region_indices)), np.nan)
    in_bounds = (starts >= 0) & (starts + n_samples <= recording.rms_traces.shape[0])
    for index in np.flatnonzero(in_bounds):
        start = int(starts[index])
        epochs[:, index, :] = recording.rms_traces[start:start + n_samples, :][:, region_indices]
    # Explicitly invalidate a whole region/window if any contributing sample is
    # missing. No interpolation, zero filling, or partial-window FFT is used.
    valid = np.isfinite(epochs).all(axis=0)
    epochs = np.where(valid[None, :, :], epochs, np.nan)
    transform = Multitaper(
        epochs, sampling_frequency=recording.resample_rate,
        time_window_duration=n_samples / recording.resample_rate,
        time_window_step=n_samples / recording.resample_rate,
        time_halfbandwidth_product=recording.halfbandwidth,
        detrend_type="constant", tapers=tapers,
    )
    connectivity = Connectivity.from_multitaper(transform, expectation_type="tapers")
    return np.asarray(connectivity.power()[0]), np.asarray(connectivity.frequencies), in_bounds


def _overlap_names(recording, start, stop):
    names = []
    for name, values in (recording.event_dict or {}).items():
        if str(name).startswith("baseline iti ") or str(name).endswith("s_windows"):
            continue
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            continue
        if values.ndim != 2 or values.shape[1] != 2:
            continue
        values = values / 1000.0
        valid = np.isfinite(values).all(axis=1) & (values[:, 1] > values[:, 0])
        if np.any(valid & (values[:, 0] < stop) & (values[:, 1] > start)):
            names.append(str(name))
    return json.dumps(sorted(names))


def _summary_tables(trial_power):
    """Preserve recording-first averaging, including its NaN weighting rules."""
    keys = [*_IDENTITY, *_SPECTRAL]
    rec = trial_power.groupby(keys, sort=False, dropna=False).agg(
        mean_z=("zscore", "mean"), mean_power=("power", "mean"),
        n_valid_trials=("zscore", "count"), n_raw_trials=("power", "count"),
    ).reset_index()
    group_keys = _SPECTRAL
    group = rec.groupby(group_keys, sort=False).agg(
        mean_z=("mean_z", "mean"), sd_z=("mean_z", "std"),
        n_recordings=("mean_z", "count"), n_valid_trials=("n_valid_trials", "sum"),
    ).reset_index()
    group["sem_z"] = group.sd_z / np.sqrt(group.n_recordings.replace(0, np.nan))
    # These are means of frequency-specific z-scores, NOT z-scores of band power.
    rec_trace = rec.groupby([*_IDENTITY, "region", "relative_time_s"], sort=False).agg(
        mean_z=("mean_z", "mean"), n_valid_frequencies=("mean_z", "count"),
    ).reset_index()
    group_trace = rec_trace.groupby(["region", "relative_time_s"], sort=False).agg(
        mean_z=("mean_z", "mean"), sd_z=("mean_z", "std"),
        n_recordings=("mean_z", "count"),
    ).reset_index()
    group_trace["sem_z"] = group_trace.sd_z / np.sqrt(group_trace.n_recordings.replace(0, np.nan))
    trial_trace = trial_power.groupby([*_TRIAL, "region", "relative_time_s"], sort=False).agg(
        mean_z=("zscore", "mean"), mean_power=("power", "mean"),
        n_valid_frequencies=("zscore", "count"),
    ).reset_index()
    return dict(recording_spectrograms=rec, group_spectrograms=group,
                trial_traces=trial_trace, recording_traces=rec_trace, group_traces=group_trace)


def calculate_event_baseline_power(
    lfp_collection, event, mode="power", condition_recordings=None,
    selected_condition=None, event_len=3.5, pre_window=3.5, post_window=0,
    regions=None, freq_range=None, max_traces=None, *,
    baseline_window=(-3.0, -1.0), baseline_ddof=1, min_baseline_estimates=3,
):
    """Z-score EACH event occurrence against its own baseline power.

    Parameters mirror plot_event_average_spectrogram. Only mode='power' is
    supported. Baseline and pre/post windows are seconds relative to onset.
    event_len=None uses actual event duration. Selected frequency bounds are
    inclusive. Each recording must contain rms_traces, resample_rate,
    timewindow, timestep, halfbandwidth, brain_region_dict, and event_dict.

    Spectral windows are centered on -pre_window + k*timestep, stopping before
    event_len + post_window. Full windows require half a spectral window of
    extra signal on either side. Baseline windows are placed wholly inside
    baseline_window, independently of the plot interval. Boundaries must be
    representable in LFP samples. Event onset is rounded to the nearest sample;
    its signed rounding error and actual window boundaries are retained.

    A region/frequency baseline is valid only if ALL planned baseline estimates
    are finite, there are at least min_baseline_estimates, and sample SD > 0.
    Any incomplete event or baseline epoch is excluded as a whole. All selected
    occurrences remain in event_qc; extractable raw power is retained even for
    invalid baselines. Overlap with annotated behavior is flagged, not excluded.
    Overlapping estimates are correlated; this exploratory method does not
    imply independent baseline samples or inferential z-test significance.

    Returns a reusable dict with 'tables', 'settings', 'event_counts',
    'skipped_recordings', 'regions', 'frequencies', and 'relative_time'. Tables
    retain trial power/z-scores, baseline samples/statistics, QC, recording
    means, group means/SEM, and trial/recording/group band traces. Group means
    weight recordings equally, NOT animals across multiple sessions. No input
    arrays are overwritten. Missing observations remain NaN.
    """
    if mode != "power":
        raise ValueError("Single-trial baseline normalization currently supports power only.")
    baseline = np.asarray(baseline_window, dtype=float)
    if baseline.shape != (2,) or not np.isfinite(baseline).all() or not baseline[0] < baseline[1] <= 0:
        raise ValueError("baseline_window must be two finite seconds with start < stop <= 0.")
    for name, value in (("pre_window", pre_window), ("post_window", post_window)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if event_len is not None and (not np.isfinite(event_len) or event_len <= 0):
        raise ValueError("event_len must be positive and finite, or None.")
    if baseline_ddof != 1:
        raise ValueError("baseline_ddof must be 1 (sample standard deviation).")
    if isinstance(min_baseline_estimates, (bool, np.bool_)) or not isinstance(min_baseline_estimates, (int, np.integer)) or min_baseline_estimates < 2:
        raise ValueError("min_baseline_estimates must be an integer >= 2.")
    if max_traces is not None and (isinstance(max_traces, bool) or not isinstance(max_traces, (int, np.integer)) or max_traces < 1):
        raise ValueError("max_traces must be a positive integer or None.")
    if (condition_recordings is None) != (selected_condition is None):
        raise ValueError("Supply condition_recordings and selected_condition together.")
    if not isinstance(event, str) or not event:
        raise ValueError("event must be a nonempty event name.")
    recordings = (list(lfp_collection.recordings) if condition_recordings is None else
                  select_recordings_by_condition(lfp_collection, condition_recordings, selected_condition))
    if not recordings or len({rec.name for rec in recordings}) != len(recordings):
        raise ValueError("Select recordings with unique names; collection must not be empty.")
    if freq_range is not None:
        bounds = np.asarray(freq_range, dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 <= bounds[0] <= bounds[1]:
            raise ValueError("freq_range must be finite ordered non-negative Hz bounds.")
    requested_regions = None if regions is None else list(regions)
    from spectral_connectivity import Multitaper

    frames, stats_frames, baseline_frames, qc_rows = [], [], [], []
    skipped, counts, signatures = {}, {}, []
    reference = None
    for recording in recordings:
        if not hasattr(recording, "rms_traces") or len(recording.rms_traces.shape) != 2:
            raise ValueError(f"{recording.name}: saved two-dimensional rms_traces are required.")
        fs, window, step, bandwidth = map(float, (recording.resample_rate, recording.timewindow,
                                               recording.timestep, recording.halfbandwidth))
        if not np.isfinite([fs, window, step, bandwidth]).all() or min(fs, window, step, bandwidth) <= 0:
            raise ValueError(f"{recording.name}: invalid spectral settings.")
        def samples(seconds):
            value = float(seconds) * fs
            rounded = int(round(value))
            if not np.isclose(value, rounded, rtol=0, atol=1e-6):
                raise ValueError("Window settings must be representable in whole LFP samples.")
            return rounded
        n_window, n_step = samples(window), samples(step)
        if n_window < 2 or n_step < 1 or n_window % 2:
            raise ValueError("Require an even window sample count >= 2 and step >= 1 sample.")
        base_start, base_stop = map(samples, baseline)
        base_offsets = np.arange(base_start, base_stop - n_window + 1, n_step, dtype=int)
        if len(base_offsets) < min_baseline_estimates:
            raise ValueError("Baseline is too short for min_baseline_estimates fully contained spectral windows.")
        selected_regions = (list(recording.brain_region_dict) if requested_regions is None else requested_regions.copy())
        if max_traces is not None:
            selected_regions = selected_regions[:max_traces]
        if not selected_regions or len(set(selected_regions)) != len(selected_regions):
            raise ValueError("Select at least one region, without duplicates.")
        if any(region not in recording.brain_region_dict for region in selected_regions):
            raise ValueError(f"{recording.name}: requested region is missing.")
        region_indices = [recording.brain_region_dict[region] for region in selected_regions]
        signature = (fs, window, step, bandwidth, tuple(selected_regions))
        if reference is not None and signature != reference:
            raise ValueError("Selected recordings have incompatible spectral settings or region order.")
        reference = signature
        signatures.append(dict(recording=recording.name, sampling_rate=fs, timewindow=window,
                               timestep=step, halfbandwidth=bandwidth,
                               source_h5=getattr(recording, "source_h5", None)))
        template = Multitaper(np.zeros((n_window, 1)), sampling_frequency=fs,
                             time_window_duration=window, time_halfbandwidth_product=bandwidth)
        tapers = template.tapers
        full_freqs = template.frequencies
        if hasattr(full_freqs, "get"):
            full_freqs = full_freqs.get()
        full_freqs = np.asarray(full_freqs)
        full_freqs = full_freqs[:(len(full_freqs) + 1) // 2]
        freq_indices = np.arange(len(full_freqs)) if freq_range is None else np.flatnonzero(
            (full_freqs >= freq_range[0]) & (full_freqs <= freq_range[1]))
        if not len(freq_indices):
            raise ValueError("No spectral bins inside freq_range.")
        frequencies = full_freqs[freq_indices]
        event_values = np.asarray((getattr(recording, "event_dict", None) or {}).get(event, []), dtype=float)
        if event_values.size == 0:
            skipped[recording.name] = "missing or empty event"
            counts[recording.name] = 0
            continue
        if event_values.ndim != 2 or event_values.shape[1] != 2:
            raise ValueError(f"{recording.name}: events must have shape (n_events, 2).")
        counts[recording.name] = 0
        condition = str(getattr(recording, "condition", "unknown"))
        if condition_recordings is not None:
            labels = [selected_condition] if isinstance(selected_condition, str) else list(selected_condition)
            condition = " | ".join(label for label in labels if recording.name in condition_recordings[label])
        for event_index, (onset_ms, stop_ms) in enumerate(event_values):
            identity = dict(recording=recording.name,
                            subject=str(getattr(recording, "subject", parse_animal_id(recording.name))),
                            condition=condition, event=event,
                            trial_id=json.dumps([recording.name, event, event_index]),
                            event_index=event_index, event_onset_s=onset_ms / 1000.)
            if not np.isfinite([onset_ms, stop_ms]).all() or stop_ms <= onset_ms:
                qc_rows.append({**identity, "status": "invalid_event_interval"})
                continue
            onset = onset_ms / 1000.
            anchor = int(round(onset * fs))
            rounding = anchor / fs - onset
            duration = (stop_ms - onset_ms) / 1000. if event_len is None else event_len
            centers = np.arange(-samples(pre_window),
                                int(np.ceil((duration + post_window) * fs - 1e-7)), n_step, dtype=int)
            offsets = centers - n_window // 2
            # One transform per occurrence; repeated plot/baseline windows are reused.
            all_offsets, inverse = np.unique(np.r_[offsets, base_offsets], return_inverse=True)
            power, computed_freqs, in_bounds = _window_power(
                recording, anchor + all_offsets, n_window, region_indices, tapers)
            if not np.allclose(computed_freqs, full_freqs):
                raise ValueError("Spectral estimator returned unexpected frequencies.")
            power = power[:, freq_indices, :]
            power[~np.isfinite(power)] = np.nan
            display_power = power[inverse[:len(offsets)]]
            base_power = power[inverse[len(offsets):]]
            base_n = np.isfinite(base_power).sum(axis=0)
            base_mean = _mean(base_power, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                base_sd = np.nanstd(base_power, axis=0, ddof=1)
            status = np.full(base_n.shape, "ok", dtype=object)
            status[~np.isfinite(base_sd) | (base_sd <= 0)] = "invalid_baseline_sd"
            status[base_n != len(base_offsets)] = "missing_baseline_power"
            if not in_bounds[inverse[len(offsets):]].all():
                status[:] = "baseline_out_of_bounds"
            if not in_bounds[inverse[:len(offsets)]].all():
                status[:] = "event_window_out_of_bounds"
            valid_baseline = status == "ok"
            z = np.full(display_power.shape, np.nan)
            np.divide(display_power - base_mean, base_sd, out=z,
                      where=np.broadcast_to(valid_baseline, z.shape))
            z[~np.isfinite(z)] = np.nan
            overlap = _overlap_names(recording, onset + baseline[0], onset + baseline[1])
            n_valid = int(np.isfinite(z).any(axis=0).sum())
            counts[recording.name] += int(n_valid > 0)
            qc_rows.append({**identity, "event_stop_s": stop_ms / 1000.,
                            "onset_rounding_error_s": rounding,
                            "baseline_start_s": anchor / fs + baseline[0],
                            "baseline_stop_s": anchor / fs + baseline[1],
                            "overlapping_events": overlap,
                            "n_valid_region_frequencies": n_valid,
                            "n_total_region_frequencies": int(valid_baseline.size),
                            "status": ("ok" if n_valid == valid_baseline.size else
                                       "partial" if n_valid else "no_valid_normalized_power")})
            nf, nr = len(frequencies), len(selected_regions)
            stats_frames.append(pd.DataFrame({**identity,
                "region": np.tile(selected_regions, nf), "frequency_hz": np.repeat(frequencies, nr),
                "baseline_mean": base_mean.ravel(), "baseline_sd": base_sd.ravel(),
                "n_baseline_estimates": base_n.ravel(), "n_expected_estimates": len(base_offsets),
                "status": status.ravel(), "overlapping_events": overlap}))
            for destination, values, starts, normalized in (
                (frames, display_power, offsets, z), (baseline_frames, base_power, base_offsets, None)
            ):
                nt = len(starts)
                frame = pd.DataFrame({**identity,
                    "relative_time_s": np.repeat((starts + n_window / 2) / fs, nf * nr),
                    "window_start_relative_s": np.repeat(starts / fs + rounding, nf * nr),
                    "window_stop_relative_s": np.repeat((starts + n_window) / fs + rounding, nf * nr),
                    "frequency_hz": np.tile(np.repeat(frequencies, nr), nt),
                    "region": np.tile(selected_regions, nt * nf), "power": values.ravel(),
                })
                if normalized is not None:
                    frame["zscore"] = normalized.ravel()
                    frame["baseline_status"] = np.tile(status.ravel(), nt)
                destination.append(frame)
    if not frames:
        raise ValueError(f"No valid event intervals found for {event!r}; skipped={skipped}.")
    trial_power = pd.concat(frames, ignore_index=True)
    tables = dict(trial_power=trial_power,
                  baseline_statistics=pd.concat(stats_frames, ignore_index=True),
                  baseline_samples=pd.concat(baseline_frames, ignore_index=True),
                  event_qc=pd.DataFrame(qc_rows), **_summary_tables(trial_power))
    settings = dict(event=event, mode="power", baseline_method="zscore",
                    baseline_window=baseline.tolist(), baseline_ddof=1,
                    min_baseline_estimates=int(min_baseline_estimates), require_all_baseline_estimates=True,
                    event_len=event_len, pre_window=pre_window, post_window=post_window,
                    freq_range=None if freq_range is None else list(freq_range), regions=selected_regions,
                    selected_condition=selected_condition, time_reference="spectral window center",
                    spectral_source="event-aligned saved rms_traces", spectral_settings=signatures,
                    averaging="trials within recording, frequencies for traces, equal recording weights",
                    overlap_policy="flag only", incomplete_epoch_policy="exclude whole occurrence",
                    created_utc=datetime.now(timezone.utc).isoformat())
    return dict(kind="event_baseline_power", tables=tables, settings=settings,
                event_counts=counts, skipped_recordings=skipped, regions=selected_regions,
                frequencies=frequencies, relative_time=np.sort(trial_power.relative_time_s.unique()))


def _result(source, event, kwargs):
    if isinstance(source, dict) and source.get("kind") == "event_baseline_power":
        if event is not None or kwargs:
            raise ValueError("A prepared result already fixes the event and calculation settings.")
        return source
    return calculate_event_baseline_power(source, event, **kwargs)


def plot_baselined_event_average_spectrogram(
    lfp_collection, event=None, *, plot=True, color_limit=None, cmap="RdBu_r", title=None, **kwargs,
):
    """Baseline equivalent of plot_event_average_spectrogram.

    Accept a collection plus event and the same calculation keywords as
    calculate_event_baseline_power, OR pass a prepared calculation result as
    the first argument to reuse it without recomputing. Returns a shallow copy
    of that result with axes. color_limit sets shared symmetric z-score limits;
    otherwise all region panels in this call share their maximum absolute mean.
    cmap controls only the appearance; use a diverging palette such as "PuOr"
    to keep zero neutral and distinguish positive from negative z-scores.
    """
    result = _result(lfp_collection, event, kwargs)
    axes = []
    if plot:
        import matplotlib.pyplot as plt
        table = result["tables"]["group_spectrograms"]
        values = table.mean_z.to_numpy()
        if color_limit is None:
            finite = np.abs(values[np.isfinite(values)])
            color_limit = max(float(finite.max()), 1e-12) if len(finite) else 1.
        if not np.isfinite(color_limit) or color_limit <= 0:
            raise ValueError("color_limit must be positive and finite.")
        settings = result["settings"]
        for region in result["regions"]:
            matrix = table[table.region == region].pivot(index="frequency_hz", columns="relative_time_s", values="mean_z")
            fig, ax = plt.subplots(figsize=(11, 4))
            mesh = ax.pcolormesh(matrix.columns, matrix.index, matrix.to_numpy(),
                                 shading="auto", cmap=cmap, vmin=-color_limit, vmax=color_limit)
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            for boundary in settings["baseline_window"]:
                ax.axvline(boundary, color="gray", linestyle=":", linewidth=1)
            ax.set(xlabel="Time from event onset (s; window centers)", ylabel="Frequency (Hz)",
                   title=f"{title or settings['event']} | {region}")
            fig.colorbar(mesh, ax=ax, label="Power change (baseline SD units)")
            fig.tight_layout()
            axes.append(ax)
    return {**result, "axes": axes}


def plot_baselined_event_average_traces(
    lfp_collection, event=None, *, comparison_event=None, comparison_condition=None,
    comparison_result=None, labels=None, plot=True, **kwargs,
):
    """Baseline equivalent of plot_event_average_traces, with recording-level SEM.

    Collection calls accept the same calculation keywords as the spectrogram
    function. comparison_event/condition inherit omitted fields from line one.
    For reuse, pass a prepared result first and optionally comparison_result;
    do not also supply calculation settings. Returns lines (each retaining its
    spectral result and all trial tables), axes, and trace_labels. No tests of
    statistical significance are performed. Frequencies are averaged AFTER
    averaging normalized trials within each recording, as in existing traces.
    """
    first = _result(lfp_collection, event, kwargs)
    results = [first]
    if comparison_result is not None:
        if comparison_event is not None or comparison_condition is not None:
            raise ValueError("Use comparison_result OR comparison event/condition.")
        results.append(_result(comparison_result, None, {}))
    elif comparison_event is not None or comparison_condition is not None:
        if isinstance(lfp_collection, dict):
            raise ValueError("Prepared results require comparison_result for a second line.")
        other_kwargs = dict(kwargs)
        if comparison_condition is not None:
            other_kwargs["selected_condition"] = comparison_condition
        results.append(calculate_event_baseline_power(
            lfp_collection, event if comparison_event is None else comparison_event, **other_kwargs))
    if labels is None:
        labels = [f"{res['settings']['event']} | {res['settings']['selected_condition'] or 'all recordings'}" for res in results]
    if isinstance(labels, str) or len(labels) != len(results) or any(not isinstance(x, str) or not x.strip() for x in labels) or len(set(labels)) != len(labels):
        raise ValueError("Supply a distinct nonempty label for each line.")
    for other in results[1:]:
        keys = ("baseline_window", "event_len", "pre_window", "post_window", "baseline_ddof", "min_baseline_estimates")
        if (other["regions"] != first["regions"] or not np.array_equal(other["frequencies"], first["frequencies"])
                or any(other["settings"][key] != first["settings"][key] for key in keys)
                or {tuple(x[k] for k in ("sampling_rate", "timewindow", "timestep", "halfbandwidth")) for x in other["settings"]["spectral_settings"]}
                != {tuple(x[k] for k in ("sampling_rate", "timewindow", "timestep", "halfbandwidth")) for x in first["settings"]["spectral_settings"]}):
            raise ValueError("Comparison results have incompatible spectral or baseline settings.")
    lines = [dict(label=label, result=res, event_counts=res["event_counts"],
                  skipped_recordings=res["skipped_recordings"],
                  traces=res["tables"]["group_traces"]) for label, res in zip(labels, results)]
    axes = []
    if plot:
        import matplotlib.pyplot as plt
        for region in first["regions"]:
            fig, ax = plt.subplots(figsize=(10, 4))
            for index, line in enumerate(lines):
                frame = line["traces"].loc[lambda x: x.region == region].sort_values("relative_time_s")
                time, mean, sem = (frame[key].to_numpy() for key in ("relative_time_s", "mean_z", "sem_z"))
                ax.plot(time, mean, label=line["label"], color=f"C{index}")
                ax.fill_between(time, mean - sem, mean + sem, color=f"C{index}", alpha=.2)
            ax.axvspan(*first["settings"]["baseline_window"], color="gray", alpha=.12, label="Baseline")
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axhline(0, color="gray", linewidth=.7)
            ax.set(xlabel="Time from event onset (s; window centers)",
                   ylabel="Mean power change (baseline SD units)",
                   title=f"{region} | {first['frequencies'][0]:g}-{first['frequencies'][-1]:g} Hz | mean +/- SEM")
            ax.legend(frameon=False)
            fig.tight_layout()
            axes.append(ax)
    return dict(kind="event_baseline_traces", lines=lines, axes=axes, trace_labels=first["regions"])


def export_event_baseline_trialwise_csv(results, path, *, day=None,
                                       recording_metadata=None, overwrite=False,
                                       split_by_recording=False, include_alignment_source=False,
                                       pairing_dict=None):
    """Write wide CSVs: one row per selection/trial/time/region, Hz columns.

    Accept a calculation/spectrogram result, a trace comparison, or a mapping
    of selection names to calculation results (e.g. notebook baseline_results).
    Frequency cells contain the original per-event z-scores, without averaging
    or display clipping. Missing values, including entirely missing rows, stay
    blank. Power regions are identified by feature; connectivity-only columns
    are omitted, as is alignment_source unless explicitly requested.

    trial_number is the original event index + 1 within recording/event, not a
    matched behavioral trial number. relative_time_s denotes window centers.
    day defaults to settings['inputs']['day']; unknown metadata stays blank.
    Optional recording_metadata maps exact recording names to role,
    paired_subject_id, and/or alignment_source. pairing_dict maps an animal ID
    to its partner; reverse entries are added automatically and conflicting
    pairs rejected. Alone recordings have no partner. Per-recording metadata
    overrides the map. When a map is supplied, known social conditions must
    have a partner rather than silently exporting a missing pairing.
    Numeric IDs such as '1.2' and '1-2' are exported consistently as '1-2'.
    Extra columns record selection, condition, event, stable trial ID, and
    normalization settings. Only calculated frequencies/regions are exported.
    With split_by_recording=True, path is an output directory and the return
    value maps recording names to absolute CSV Paths. Each recording's file
    combines all its selections/events. Otherwise path names one combined CSV
    and the return value is its absolute Path. No auxiliary files are written.
    """
    if results.get("kind") == "event_baseline_power":
        entries = [(results["settings"]["event"], results)]
    elif results.get("kind") == "event_baseline_traces":
        entries = [(f"{i}: {line['label']}", line["result"])
                   for i, line in enumerate(results["lines"])]
    else:
        entries = list(results.items())
    if not entries or any(not isinstance(r, dict) or r.get("kind") != "event_baseline_power"
                          for _, r in entries):
        raise ValueError("Supply baseline results, a trace comparison, or a nonempty selection mapping.")
    path = Path(path).resolve()
    if not split_by_recording and path.suffix.lower() != ".csv":
        raise ValueError("path must name a .csv file.")
    if not split_by_recording and path.exists() and not overwrite:
        raise FileExistsError("CSV already exists; choose a new path or overwrite=True.")
    recording_metadata = recording_metadata or {}
    allowed_metadata = {"role", "paired_subject_id", "alignment_source"}
    for metadata in recording_metadata.values():
        if set(metadata) - allowed_metadata:
            raise ValueError(f"recording_metadata supports only {sorted(allowed_metadata)}.")
    partners = _validated_pairing_dict(pairing_dict)

    frequencies = np.sort(np.asarray(entries[0][1]["frequencies"], dtype=float))
    frequency_columns = [f"freq_{i:03d}_{float(f):.15g}_Hz"
                         for i, f in enumerate(frequencies, start=1)]
    frames = []
    roles = {"subject with recipient": "subject", "subject alone": "subject", "recipient": "recipient"}
    choices = {"filtered selfish nose poke": "selfish", "filtered coop nose poke": "coop"}
    for selection, result in entries:
        if not np.array_equal(np.sort(result["frequencies"]), frequencies):
            raise ValueError("Combined selections must use the same frequency grid.")
        settings = result["settings"]
        source_day = settings.get("inputs", {}).get("day")
        if day is not None and source_day is not None and day != source_day:
            raise ValueError("day conflicts with the calculation's input day.")
        points = result["tables"]["trial_power"]
        keys = ["trial_id", "relative_time_s", "region"]
        # pivot (not pivot_table) preserves all-NaN observations and rejects
        # duplicate observations rather than silently averaging them.
        wide = points.pivot(index=keys, columns="frequency_hz", values="zscore")
        wide = wide.reindex(columns=frequencies)
        ids = points.drop_duplicates(keys).set_index(keys).reindex(wide.index).reset_index()
        animal_ids = ids.subject.map(_normalize_animal_id)
        records = ids.recording
        spectral = {item["recording"]: item for item in settings["spectral_settings"]}
        def metadata_column(name, defaults):
            return pd.Series([recording_metadata.get(rec, {}).get(name, default)
                              for rec, default in zip(records, defaults)], index=ids.index)
        paired_ids = metadata_column(
            "paired_subject_id",
            animal_ids.map(partners).mask(ids.condition.eq("subject alone"), "").fillna(""),
        ).map(_normalize_animal_id)
        if pairing_dict is not None:
            social = ids.condition.isin(["subject with recipient", "recipient"])
            missing = social & (paired_ids.isna() | paired_ids.eq(""))
            if missing.any():
                raise ValueError(f"Missing partner IDs for social animals: {sorted(ids.loc[missing, 'subject'].unique())}")
        frame = pd.DataFrame({
            "day": day if day is not None else source_day,
            "animal_id": animal_ids,
            "role": metadata_column("role", ids.condition.map(roles).fillna("")),
            "paired_subject_id": paired_ids,
            "h5_file": [Path(spectral.get(rec, {}).get("source_h5") or "").name for rec in records],
            "rec_file": records,
            "choice": ids.event.map(choices).fillna(ids.event),
            "trial_number": ids.event_index + 1,
            "alignment_time_s": ids.event_onset_s,
            "alignment_source": metadata_column("alignment_source", ["event_dict_onset"] * len(ids)),
            "relative_time_s": ids.relative_time_s,
            "metric": "power",
            "feature": ids.region,
            "selection": str(selection),
            "condition": ids.condition,
            "event": ids.event,
            "trial_id": ids.trial_id,
            "normalization": settings["baseline_method"],
            "baseline_start_relative_s": settings["baseline_window"][0],
            "baseline_stop_relative_s": settings["baseline_window"][1],
            "baseline_ddof": settings["baseline_ddof"],
            "time_reference": settings["time_reference"],
        })
        values = wide.reset_index(drop=True)
        values.columns = frequency_columns
        frames.append(pd.concat([frame, values], axis=1))
    combined = pd.concat(frames, ignore_index=True)
    # trial_id is a JSON string: lexical sorting puts index 10 before 1.
    combined = combined.sort_values(
        ["rec_file", "selection", "trial_number", "relative_time_s", "feature"],
        kind="stable", ignore_index=True,
    )
    if not include_alignment_source:
        combined = combined.drop(columns="alignment_source")
    return _write_trialwise_csv(combined, path, split_by_recording=split_by_recording,
                                overwrite=overwrite)


def _normalize_animal_id(value):
    if isinstance(value, str):
        value = value.strip()
        if re.fullmatch(r"\d+[.-]\d+", value):
            return value.replace(".", "-")
    return value


def _validated_pairing_dict(pairing_dict):
    partners = {}
    for first, second in (pairing_dict or {}).items():
        if not all(isinstance(value, str) and value.strip() for value in (first, second)):
            raise ValueError("Pairing keys and values must be nonempty animal ID strings.")
        first, second = _normalize_animal_id(first), _normalize_animal_id(second)
        if first == second:
            raise ValueError("An animal cannot be its own partner.")
        for animal, partner in ((first, second), (second, first)):
            if animal in partners and partners[animal] != partner:
                raise ValueError(f"Conflicting partners for {animal}.")
            partners[animal] = partner
    return partners


def _write_trialwise_csv(frame, path, *, split_by_recording, overwrite):
    """Preflight every filename before writing any recording's data."""
    path = Path(path).resolve()
    if split_by_recording:
        groups = list(frame.groupby("rec_file", sort=False, dropna=False))
        paths = {}
        for recording, _ in groups:
            safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(recording)).rstrip(" .")
            paths[recording] = path / f"trialwise_{safe}.csv"
        if len({p.name.casefold() for p in paths.values()}) != len(paths):
            raise ValueError("Recording names produce colliding CSV filenames.")
    else:
        groups = [(None, frame)]
        paths = {None: path}
    if not overwrite and any(p.exists() for p in paths.values()):
        raise FileExistsError("CSV files already exist; choose a new path or overwrite=True.")
    for recording, group in groups:
        target = paths[recording]
        target.parent.mkdir(parents=True, exist_ok=True)
        group.to_csv(target, index=False, float_format="%.17g", na_rep="")
    return paths if split_by_recording else path


def export_event_baseline_csv(result, output_dir, *, overwrite=False):
    """Export every numerical table, including exactly plotted means/SEM.

    Accept a calculation/spectrogram result or a trace comparison result. The
    latter adds comparison_line and comparison_label columns to each table.
    Missing numerical values are written as empty CSV fields, never zeros.
    Output floats use 17 significant digits. Existing files are protected unless
    overwrite=True. Return a mapping of table/metadata names to absolute paths.
    """
    if result.get("kind") == "event_baseline_traces":
        entries = [(i, line["label"], line["result"]) for i, line in enumerate(result["lines"])]
    elif result.get("kind") == "event_baseline_power":
        entries = [(0, result["settings"]["event"], result)]
    else:
        raise ValueError("Expected a baseline calculation or trace result.")
    directory = Path(output_dir).resolve()
    table_names = list(entries[0][2]["tables"])
    paths = {name: directory / f"{name}.csv" for name in table_names}
    paths["analysis_settings"] = directory / "analysis_settings.json"
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError("Export files already exist; choose a new directory or overwrite=True.")
    directory.mkdir(parents=True, exist_ok=True)
    for name in table_names:
        for index, (line_index, label, calculation) in enumerate(entries):
            frame = calculation["tables"][name].assign(comparison_line=line_index, comparison_label=label)
            frame.to_csv(paths[name], index=False, float_format="%.17g", na_rep="",
                         mode="w" if index == 0 else "a", header=index == 0)
    metadata = [dict(comparison_line=i, comparison_label=label, **calculation["settings"],
                     event_counts=calculation["event_counts"], skipped_recordings=calculation["skipped_recordings"])
                for i, label, calculation in entries]
    paths["analysis_settings"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths
