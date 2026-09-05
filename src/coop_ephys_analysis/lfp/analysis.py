"""Analysis and plotting workflows for LFP notebooks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

from .processing import select_recordings_by_condition


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _external_lfp_repo() -> Path:
    return _repo_root() / "external" / "diff_fam_social_memory_ephys"


def _ensure_external_lfp_on_path() -> Path:
    external_repo = _external_lfp_repo()
    if not external_repo.exists():
        raise FileNotFoundError(
            f"Expected external LFP repository at {external_repo}. "
            "Make sure the diff_fam_social_memory_ephys submodule is present."
        )

    external_repo_str = str(external_repo)
    if external_repo_str not in sys.path:
        sys.path.insert(0, external_repo_str)

    return external_repo


def _legacy_modules():
    _ensure_external_lfp_on_path()
    import lfp.lfp_analysis.event_extraction as ee
    import lfp.lfp_analysis.plotting as lfplt

    return ee, lfplt


def get_lfp_collection_class():
    """Return the legacy LFPCollection class through the local package boundary."""
    _ensure_external_lfp_on_path()
    from lfp.lfp_analysis.LFP_collection import LFPCollection

    return LFPCollection


def load_lfp_collection(json_path):
    """Load a saved legacy LFPCollection from its lfp_collection.json path."""
    return get_lfp_collection_class().load_collection(json_path)


def average_events_by_condition(
    lfp_collection,
    condition_recordings,
    selected_condition,
    events,
    mode="power",
    baseline=None,
    event_len=None,
    pre_window=0,
    post_window=0,
    *,
    verbose=True,
):
    """Average events after filtering an LFP collection by condition."""
    ee, _ = _legacy_modules()
    filtered_recordings = select_recordings_by_condition(
        lfp_collection,
        condition_recordings=condition_recordings,
        selected_condition=selected_condition,
    )

    if verbose:
        selected_conditions = (
            [selected_condition]
            if isinstance(selected_condition, str)
            else list(selected_condition)
        )
        print(
            f"Calculating {mode} event spectrum for condition(s) {selected_conditions} "
            f"using {len(filtered_recordings)} recording(s)."
        )

    filtered_collection = SimpleNamespace(
        recordings=filtered_recordings,
        brain_region_dict=lfp_collection.brain_region_dict,
    )

    return ee.average_events(
        filtered_collection,
        events=events,
        mode=mode,
        baseline=baseline,
        event_len=event_len,
        pre_window=pre_window,
        post_window=post_window,
        plot=False,
    )


def condition_event_spectrum(
    lfp_collection,
    condition_recordings,
    selected_condition,
    events,
    mode="power",
    baseline=None,
    event_len=None,
    pre_window=0,
    post_window=0,
    regions=None,
    freq_range=None,
    plot=True,
    *,
    verbose=True,
):
    """Average and optionally plot event spectra for one or more conditions."""
    _, lfplt = _legacy_modules()
    event_averages = average_events_by_condition(
        lfp_collection,
        condition_recordings=condition_recordings,
        selected_condition=selected_condition,
        events=events,
        mode=mode,
        baseline=baseline,
        event_len=event_len,
        pre_window=pre_window,
        post_window=post_window,
        verbose=verbose,
    )

    if plot:
        lfplt.plot_event_spectrum(
            lfp_collection,
            event_averages,
            mode=mode,
            regions=regions,
            freq_range=freq_range,
        )

    return event_averages


def _analysis_imports():
    """Import plotting dependencies lazily so non-plot imports stay light."""
    import math
    from itertools import combinations, permutations

    import matplotlib.pyplot as plt
    import numpy as np

    return math, combinations, permutations, plt, np


def _resolve_recording(lfp_collection, recording):
    """Return an LFPRecording from a recording object or recording filename."""
    if hasattr(recording, "power") or hasattr(recording, "coherence") or hasattr(recording, "granger"):
        return recording

    if hasattr(lfp_collection, "get_recording"):
        return lfp_collection.get_recording(recording)

    matches = [rec for rec in lfp_collection.recordings if rec.name == recording]
    if not matches:
        raise ValueError(f"No recording named {recording!r} found in this LFPCollection.")
    return matches[0]


def _frequency_indices(recording, freq_range):
    """Convert an optional frequency range in Hz into indices for spectral arrays."""
    _, _, _, _, np = _analysis_imports()
    if not hasattr(recording, "frequencies"):
        raise AttributeError(
            "Recording must have `frequencies`; run spectral calculations or load a processed collection first."
        )

    frequencies = np.asarray(recording.frequencies)
    if freq_range is None:
        return np.arange(len(frequencies)), f"{frequencies[0]:.1f}-{frequencies[-1]:.1f} Hz"

    low, high = freq_range
    freq_idx = np.where((frequencies >= low) & (frequencies <= high))[0]
    if freq_idx.size == 0:
        raise ValueError(f"No frequency bins found in range {freq_range} Hz.")
    return freq_idx, f"{low}-{high} Hz"


def _time_indices(recording, n_time_bins, time_window):
    """Convert an optional time window in seconds into spectral time-bin indices."""
    math, _, _, _, np = _analysis_imports()
    timestep = recording.timestep
    if time_window is None:
        start_idx, stop_idx = 0, n_time_bins
    else:
        start_s, stop_s = time_window
        if start_s < 0 or stop_s <= start_s:
            raise ValueError(
                "time_window must be None or a tuple like (start_s, stop_s), with stop_s > start_s >= 0."
            )
        start_idx = max(0, math.floor(start_s / timestep))
        stop_idx = min(n_time_bins, math.ceil(stop_s / timestep))
        if start_idx >= n_time_bins:
            raise ValueError("time_window starts after the available spectral data.")

    time_axis = np.arange(start_idx, stop_idx) * timestep
    return start_idx, stop_idx, time_axis


def _region_labels(recording):
    """Return region names ordered by their array index."""
    return [recording.brain_region_dict.inverse[i] for i in range(len(recording.brain_region_dict))]


def _selected_regions(recording, regions=None, max_traces=None):
    """Return region names filtered by user selection and optional count cap."""
    region_names = _region_labels(recording)
    selected = region_names if regions is None else list(regions)

    missing = [region for region in selected if region not in recording.brain_region_dict]
    if missing:
        raise ValueError(f"Region(s) not found in recording {recording.name!r}: {missing}")

    if max_traces is not None:
        selected = selected[:max_traces]
    return selected


def _pair_indices(recording, mode, pairs=None):
    """Return region-pair indices and labels for coherence or Granger plotting."""
    _, combinations, permutations, _, _ = _analysis_imports()
    region_dict = recording.brain_region_dict
    region_names = _region_labels(recording)

    if pairs is None:
        if mode == "coherence":
            pair_names = list(combinations(region_names, 2))
        else:
            pair_names = list(permutations(region_names, 2))
    else:
        pair_names = list(pairs)

    pair_indices = []
    pair_labels = []
    for first, second in pair_names:
        if first not in region_dict or second not in region_dict:
            raise ValueError(f"Pair {(first, second)!r} contains a region not found in {recording.name!r}.")
        i = region_dict[first]
        j = region_dict[second]
        pair_indices.append((i, j, first, second))
        if mode == "coherence":
            pair_labels.append(f"{first} & {second}")
        else:
            pair_labels.append(f"{second} to {first}")
    return pair_indices, pair_labels


def _style_spectral_axis(ax, title, ylabel):
    """Apply shared plot styling for spectral trace axes."""
    _, _, _, plt, _ = _analysis_imports()
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()


def plot_recording_spectral_traces(
    lfp_collection,
    recording,
    mode="power",
    time_window=None,
    freq_range=None,
    regions=None,
    pairs=None,
    max_traces=None,
    ax=None,
):
    """Plot power, coherence, or Granger traces across time for one recording."""
    _, _, _, plt, np = _analysis_imports()
    mode = mode.lower()
    if mode not in {"power", "coherence", "granger"}:
        raise ValueError("mode must be 'power', 'coherence', or 'granger'.")

    recording = _resolve_recording(lfp_collection, recording)
    if not hasattr(recording, mode):
        raise AttributeError(f"Recording {recording.name!r} does not have `{mode}`. Calculate or load it first.")

    values = getattr(recording, mode)
    freq_idx, freq_label = _frequency_indices(recording, freq_range)
    start_idx, stop_idx, time_axis = _time_indices(recording, values.shape[0], time_window)
    axes = []

    if mode == "power":
        selected_regions = _selected_regions(recording, regions=regions, max_traces=max_traces)
        if ax is not None and len(selected_regions) != 1:
            raise ValueError("ax can only be supplied when exactly one power region will be plotted.")

        for region in selected_regions:
            region_idx = recording.brain_region_dict[region]
            trace = np.nanmean(values[start_idx:stop_idx][:, freq_idx, region_idx], axis=1)
            current_ax = ax if ax is not None else plt.subplots(figsize=(12, 4))[1]
            current_ax.plot(time_axis, trace, label=region)
            _style_spectral_axis(
                current_ax,
                title=f"{recording.name}\n{region} {mode.title()} over time ({freq_label})",
                ylabel=mode.title(),
            )
            axes.append(current_ax)
        return axes

    selected_anchor_regions = _selected_regions(recording, regions=regions, max_traces=None)
    pair_indices, pair_labels = _pair_indices(recording, mode, pairs=pairs)
    if ax is not None and len(selected_anchor_regions) != 1:
        raise ValueError("ax can only be supplied when exactly one region-specific plot will be produced.")

    for anchor_region in selected_anchor_regions:
        anchor_idx = recording.brain_region_dict[anchor_region]
        traces_for_anchor = []
        for pair_info, label in zip(pair_indices, pair_labels):
            first_idx, second_idx, first_region, second_region = pair_info
            if mode == "coherence":
                if anchor_region not in {first_region, second_region}:
                    continue
            elif first_idx != anchor_idx:
                continue

            trace = np.nanmean(values[start_idx:stop_idx][:, freq_idx, first_idx, second_idx], axis=1)
            traces_for_anchor.append((trace, label))

        if max_traces is not None:
            traces_for_anchor = traces_for_anchor[:max_traces]
        if not traces_for_anchor:
            continue

        current_ax = ax if ax is not None else plt.subplots(figsize=(12, 5))[1]
        for trace, label in traces_for_anchor:
            current_ax.plot(time_axis, trace, label=label)

        title_region = f"{anchor_region}" if mode == "coherence" else f"target {anchor_region}"
        _style_spectral_axis(
            current_ax,
            title=f"{recording.name}\n{title_region} {mode.title()} over time ({freq_label})",
            ylabel=mode.title(),
        )
        axes.append(current_ax)

    return axes


def plot_collection_spectral_traces(
    lfp_collection,
    mode="power",
    time_window=None,
    freq_range=None,
    regions=None,
    pairs=None,
    max_traces=None,
):
    """Plot power, coherence, or Granger traces across time for every recording."""
    axes = []
    for recording in lfp_collection.recordings:
        axes.extend(
            plot_recording_spectral_traces(
                lfp_collection=lfp_collection,
                recording=recording,
                mode=mode,
                time_window=time_window,
                freq_range=freq_range,
                regions=regions,
                pairs=pairs,
                max_traces=max_traces,
            )
        )
    return axes


def _get_event_windows_ms(recording, event):
    """Return one event's windows from recording.event_dict as millisecond pairs."""
    _, _, _, _, np = _analysis_imports()
    if getattr(recording, "event_dict", None) is None:
        raise ValueError(f"Recording {recording.name!r} has no event_dict.")
    if event not in recording.event_dict:
        raise ValueError(f"Event {event!r} not found in recording {recording.name!r}.")

    event_windows = np.asarray(recording.event_dict[event], dtype=float)
    if event_windows.size == 0:
        return np.empty((0, 2), dtype=float)

    event_windows = np.atleast_2d(event_windows)
    if event_windows.shape[1] < 2:
        raise ValueError(f"Event {event!r} must contain start/stop pairs.")

    event_windows = event_windows[:, :2]
    return event_windows[event_windows[:, 1] > event_windows[:, 0]]


def _event_windows_to_spectral_slices(recording, event_windows_ms, n_time_bins):
    """Convert millisecond event windows into spectral time-bin slices."""
    math, _, _, _, _ = _analysis_imports()
    freq_timebin_ms = recording.timestep * 1000
    slices = []
    segment_lengths = []

    for start_ms, stop_ms in event_windows_ms:
        start_idx = max(0, math.ceil(start_ms / freq_timebin_ms))
        stop_idx = min(n_time_bins, math.ceil(stop_ms / freq_timebin_ms))
        if stop_idx <= start_idx:
            continue
        slices.append(slice(start_idx, stop_idx))
        segment_lengths.append(stop_idx - start_idx)

    return slices, segment_lengths


def _concatenate_event_spectral_values(recording, values, event):
    """Concatenate spectral values across all windows of one event."""
    _, _, _, _, np = _analysis_imports()
    event_windows = _get_event_windows_ms(recording, event)
    if len(event_windows) == 0:
        return None, []

    slices, segment_lengths = _event_windows_to_spectral_slices(
        recording,
        event_windows,
        n_time_bins=values.shape[0],
    )
    if not slices:
        return None, []

    concatenated = np.concatenate([values[event_slice, ...] for event_slice in slices], axis=0)
    return concatenated, segment_lengths


def _event_concat_time_axis(recording, n_time_bins):
    """Build the x-axis for concatenated event snippets."""
    _, _, _, _, np = _analysis_imports()
    return np.arange(n_time_bins) * recording.timestep


def _event_boundary_positions(recording, segment_lengths):
    """Return x-axis positions where one event snippet ends and the next begins."""
    _, _, _, _, np = _analysis_imports()
    if len(segment_lengths) <= 1:
        return []
    return np.cumsum(segment_lengths)[:-1] * recording.timestep


def _plot_event_spectrogram(ax, spectrogram, time_axis, frequencies, event_boundaries, title, colorbar_label):
    """Draw one event-concatenated spectrogram heatmap."""
    _, _, _, plt, _ = _analysis_imports()
    image = ax.pcolormesh(time_axis, frequencies, spectrogram.T, shading="auto")
    for boundary in event_boundaries:
        ax.axvline(boundary, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Concatenated event time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.colorbar(image, ax=ax, label=colorbar_label)
    plt.tight_layout()


def plot_recording_event_spectrogram(
    lfp_collection,
    recording,
    event,
    mode="power",
    freq_range=None,
    regions=None,
    pairs=None,
    max_traces=None,
):
    """Plot event-concatenated spectrograms for one recording."""
    _, _, _, plt, np = _analysis_imports()
    mode = mode.lower()
    if mode not in {"power", "coherence", "granger"}:
        raise ValueError("mode must be 'power', 'coherence', or 'granger'.")

    recording = _resolve_recording(lfp_collection, recording)
    if not hasattr(recording, mode):
        raise AttributeError(f"Recording {recording.name!r} does not have `{mode}`. Calculate or load it first.")

    values = getattr(recording, mode)
    event_values, segment_lengths = _concatenate_event_spectral_values(recording, values, event)
    if event_values is None:
        raise ValueError(f"Event {event!r} has no valid windows inside recording {recording.name!r}.")

    freq_idx, freq_label = _frequency_indices(recording, freq_range)
    frequencies = np.asarray(recording.frequencies)[freq_idx]
    time_axis = _event_concat_time_axis(recording, event_values.shape[0])
    event_boundaries = _event_boundary_positions(recording, segment_lengths)
    axes = []

    if mode == "power":
        selected_regions = _selected_regions(recording, regions=regions, max_traces=max_traces)
        for region in selected_regions:
            region_idx = recording.brain_region_dict[region]
            spectrogram = event_values[:, freq_idx, region_idx]
            _, ax = plt.subplots(figsize=(12, 4))
            _plot_event_spectrogram(
                ax,
                spectrogram=spectrogram,
                time_axis=time_axis,
                frequencies=frequencies,
                event_boundaries=event_boundaries,
                title=f"{recording.name}\n{event}: {region} {mode.title()} ({freq_label})",
                colorbar_label=mode.title(),
            )
            axes.append(ax)
        return axes

    pair_indices, pair_labels = _pair_indices(recording, mode, pairs=pairs)
    if max_traces is not None:
        pair_indices = pair_indices[:max_traces]
        pair_labels = pair_labels[:max_traces]

    for pair_info, pair_label in zip(pair_indices, pair_labels):
        first_idx, second_idx, _, _ = pair_info
        spectrogram = event_values[:, freq_idx, first_idx, second_idx]
        _, ax = plt.subplots(figsize=(12, 4))
        _plot_event_spectrogram(
            ax,
            spectrogram=spectrogram,
            time_axis=time_axis,
            frequencies=frequencies,
            event_boundaries=event_boundaries,
            title=f"{recording.name}\n{event}: {pair_label} {mode.title()} ({freq_label})",
            colorbar_label=mode.title(),
        )
        axes.append(ax)

    return axes


def plot_collection_event_spectrogram(
    lfp_collection,
    event,
    mode="power",
    freq_range=None,
    regions=None,
    pairs=None,
    max_traces=None,
):
    """Plot event-concatenated spectrograms for every recording in a collection."""
    axes = []
    for recording in lfp_collection.recordings:
        try:
            axes.extend(
                plot_recording_event_spectrogram(
                    lfp_collection=lfp_collection,
                    recording=recording,
                    event=event,
                    mode=mode,
                    freq_range=freq_range,
                    regions=regions,
                    pairs=pairs,
                    max_traces=max_traces,
                )
            )
        except ValueError as exc:
            print(f"Skipping {recording.name}: {exc}")
            continue
    return axes
