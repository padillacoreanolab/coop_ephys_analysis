"""Processing helpers for LFP analysis notebooks."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_RECIPIENT_IDS = {"1-3", "2-1", "4-2", "4-1"}
DEFAULT_SUBJECT_WITH_RECIPIENT_IDS = {"1-2", "2-4", "4-4", "4-3"}
DEFAULT_SUBJECT_ALONE_IDS = {"6-1", "6-3"}


def parse_animal_id(recording_name: str, animal_id_index: int = 4) -> str | None:
    """Parse the animal ID token from a recording filename."""
    parts = str(recording_name).split("_")
    if -len(parts) <= animal_id_index < len(parts):
        animal_id = parts[animal_id_index].replace(".", "-")
        if re.fullmatch(r"\d+-\d+", animal_id):
            return animal_id

    for part in parts:
        animal_id = part.replace(".", "-")
        if re.fullmatch(r"\d+-\d+", animal_id):
            return animal_id

    return None


def build_condition_dict(
    collection,
    recipients: Iterable[str] | None = None,
    subject_with_recipient: Iterable[str] | None = None,
    subject_alone: Iterable[str] | None = None,
    *,
    animal_id_index: int = 4,
    unknown_label: str = "unknown",
    set_recording_attr: bool = True,
):
    """Group recording names by Stage 4 role using animal IDs from filenames."""
    recipient_ids = set(recipients or DEFAULT_RECIPIENT_IDS)
    subject_with_recipient_ids = set(
        subject_with_recipient or DEFAULT_SUBJECT_WITH_RECIPIENT_IDS
    )
    subject_alone_ids = set(subject_alone or DEFAULT_SUBJECT_ALONE_IDS)

    condition_dict = defaultdict(list)

    for recording in collection.recordings:
        animal_id = parse_animal_id(recording.name, animal_id_index=animal_id_index)

        if animal_id in subject_alone_ids:
            condition = "subject alone"
        elif animal_id in subject_with_recipient_ids:
            condition = "subject with recipient"
        elif animal_id in recipient_ids:
            condition = "recipient"
        else:
            condition = unknown_label

        if set_recording_attr:
            recording.condition = condition
        condition_dict[condition].append(recording.name)

    return condition_dict


def _as_condition_list(selected_condition) -> list[str]:
    if isinstance(selected_condition, str):
        return [selected_condition]
    return list(selected_condition)


def select_recordings_by_condition(collection, condition_recordings, selected_condition):
    """Return collection recordings whose names belong to one or more conditions."""
    selected_conditions = _as_condition_list(selected_condition)

    missing_conditions = [
        condition for condition in selected_conditions if condition not in condition_recordings
    ]
    if missing_conditions:
        raise ValueError(f"Condition(s) not found in condition_recordings: {missing_conditions}")

    selected_recording_names = []
    for condition in selected_conditions:
        selected_recording_names.extend(condition_recordings[condition])
    selected_recording_names = list(dict.fromkeys(selected_recording_names))

    selected_recording_set = set(selected_recording_names)
    filtered_recordings = [
        recording
        for recording in collection.recordings
        if recording.name in selected_recording_set
    ]

    if not filtered_recordings:
        raise ValueError(
            f"No recordings for selected_condition={selected_condition} were found in this LFP collection."
        )

    return filtered_recordings


def _recording_data_for_mode(recording, mode: str):
    if mode == "power":
        return recording.power
    if mode == "coherence":
        return recording.coherence
    if mode == "granger":
        return recording.granger
    raise ValueError(f"Unsupported mode: {mode}")


def _empty_event_row(recording, condition: str, event_name: str, mode: str, reason: str):
    data = _recording_data_for_mode(recording, mode)
    freq_timebin_ms = recording.timestep * 1000
    return {
        "condition": condition,
        "recording": recording.name,
        "event": event_name,
        "mode": mode,
        "status": reason,
        "event_count": 0,
        "recording_duration_ms": data.shape[0] * freq_timebin_ms,
        "freq_timebin_ms": freq_timebin_ms,
        "raw_start_min": np.nan,
        "raw_start_max": np.nan,
        "raw_stop_min": np.nan,
        "raw_stop_max": np.nan,
        "duration_min": np.nan,
        "duration_max": np.nan,
        "post_index_min": np.nan,
        "post_index_max": np.nan,
        "bin_width_min": np.nan,
        "bin_width_max": np.nan,
        "within_recording_count": 0,
        "nonempty_count": 0,
        "zero_width_count": 0,
    }


def _event_window_row(recording, condition: str, event_name: str, mode: str):
    event_dict = getattr(recording, "event_dict", None) or {}
    if event_name not in event_dict:
        return _empty_event_row(recording, condition, event_name, mode, "missing")

    events = np.asarray(event_dict[event_name], dtype=float)
    if events.size == 0:
        return _empty_event_row(recording, condition, event_name, mode, "empty")

    events = np.atleast_2d(events)
    if events.shape[1] < 2:
        return _empty_event_row(recording, condition, event_name, mode, "invalid_shape")

    events = events[:, :2]
    data = _recording_data_for_mode(recording, mode)
    freq_timebin_ms = recording.timestep * 1000
    durations = events[:, 1] - events[:, 0]
    pre_indices = np.ceil(events[:, 0] / freq_timebin_ms).astype(int)
    post_indices = np.ceil(events[:, 1] / freq_timebin_ms).astype(int)
    bin_widths = post_indices - pre_indices
    within_recording = post_indices < data.shape[0]
    nonempty = within_recording & (bin_widths > 0)

    return {
        "condition": condition,
        "recording": recording.name,
        "event": event_name,
        "mode": mode,
        "status": "ok",
        "event_count": len(events),
        "recording_duration_ms": data.shape[0] * freq_timebin_ms,
        "freq_timebin_ms": freq_timebin_ms,
        "raw_start_min": np.nanmin(events[:, 0]),
        "raw_start_max": np.nanmax(events[:, 0]),
        "raw_stop_min": np.nanmin(events[:, 1]),
        "raw_stop_max": np.nanmax(events[:, 1]),
        "duration_min": np.nanmin(durations),
        "duration_max": np.nanmax(durations),
        "post_index_min": np.nanmin(post_indices),
        "post_index_max": np.nanmax(post_indices),
        "bin_width_min": np.nanmin(bin_widths),
        "bin_width_max": np.nanmax(bin_widths),
        "within_recording_count": int(np.sum(within_recording)),
        "nonempty_count": int(np.sum(nonempty)),
        "zero_width_count": int(np.sum(bin_widths <= 0)),
    }


def summarize_event_windows(
    collection,
    condition_recordings: Mapping[str, Sequence[str]],
    selected_condition,
    events: Sequence[str],
    mode: str = "power",
) -> pd.DataFrame:
    """Summarize event-window extraction counts and bin widths for selected conditions.

    Event timestamps are assumed to already be in milliseconds.
    """
    selected_conditions = _as_condition_list(selected_condition)
    missing_conditions = [
        condition for condition in selected_conditions if condition not in condition_recordings
    ]
    if missing_conditions:
        raise ValueError(f"Condition(s) not found in condition_recordings: {missing_conditions}")

    rows = []
    for condition in selected_conditions:
        recording_names = set(condition_recordings[condition])
        condition_recordings_in_collection = [
            recording for recording in collection.recordings if recording.name in recording_names
        ]
        if not condition_recordings_in_collection:
            raise ValueError(
                f"No recordings for selected_condition={condition} were found in this LFP collection."
            )

        for recording in condition_recordings_in_collection:
            for event_name in events:
                rows.append(_event_window_row(recording, condition, event_name, mode))

    return pd.DataFrame(rows)


def _iti_window_key(window_duration):
    """Return the behavior key for ITI windows, e.g. 2.0 -> '2s_windows'."""
    duration = float(window_duration)
    if duration.is_integer():
        duration_label = str(int(duration))
    else:
        duration_label = str(duration).rstrip("0").rstrip(".")
    return f"{duration_label}s_windows"


def _is_iti_window_key(event_name):
    """Identify generated ITI window events so they are not treated as behavior."""
    return str(event_name).endswith("s_windows")


def _collect_behavior_events(recording_behaviors):
    """Flatten one recording's behavior events into sorted millisecond pairs."""
    all_events = []
    if recording_behaviors is None:
        return np.empty((0, 2), dtype=float)

    for behavior_name, events in recording_behaviors.items():
        if _is_iti_window_key(behavior_name) or events is None:
            continue

        events = np.asarray(events, dtype=float)
        if events.size == 0:
            continue
        events = np.atleast_2d(events)
        if events.shape[1] < 2:
            continue

        all_events.extend(events[:, :2].tolist())

    if not all_events:
        return np.empty((0, 2), dtype=float)

    all_events = np.asarray(all_events, dtype=float)
    all_events = all_events[np.argsort(all_events[:, 0])]
    return all_events


def _merge_behavior_events(events):
    """Merge overlapping or touching event intervals."""
    if len(events) == 0:
        return events

    merged_events = [events[0].tolist()]
    for current_start, current_stop in events[1:]:
        last_event = merged_events[-1]
        if current_start <= last_event[1]:
            last_event[1] = max(last_event[1], current_stop)
        else:
            merged_events.append([current_start, current_stop])

    return np.asarray(merged_events, dtype=float)


def _resolve_time_bounds_ms(recording, events, time_bounds=None):
    """Return millisecond bounds for one recording."""
    if time_bounds is not None:
        min_time, max_time = [float(value) for value in time_bounds]
        if max_time <= min_time:
            raise ValueError(
                "time_bounds must be (min_time_ms, max_time_ms), with max_time_ms > min_time_ms."
            )
        return min_time, max_time

    if hasattr(recording, "traces") and hasattr(recording, "resample_rate"):
        max_time = recording.traces.shape[0] / recording.resample_rate * 1000
        return 0.0, float(max_time)

    if len(events) == 0:
        return None

    return 0.0, float(events[:, 1].max() + 10000)


def _get_iti_windows_for_recording(recording, window_duration=2.0, time_bounds=None):
    """Return non-overlapping ITI windows for one LFPRecording as millisecond pairs."""
    if window_duration <= 0:
        raise ValueError("window_duration must be greater than 0 seconds.")

    window_duration_ms = float(window_duration) * 1000
    events = _collect_behavior_events(getattr(recording, "event_dict", None))
    bounds = _resolve_time_bounds_ms(recording, events, time_bounds=time_bounds)
    if bounds is None:
        return []

    min_time, max_time = bounds
    if len(events) == 0:
        merged_events = np.empty((0, 2), dtype=float)
    else:
        events = events.copy()
        events[:, 0] = np.maximum(events[:, 0], min_time)
        events[:, 1] = np.minimum(events[:, 1], max_time)
        events = events[events[:, 1] > events[:, 0]]
        merged_events = _merge_behavior_events(events)

    gaps = []
    cursor = min_time
    for event_start, event_stop in merged_events:
        if event_start > cursor:
            gaps.append((cursor, event_start))
        cursor = max(cursor, event_stop)
    if cursor < max_time:
        gaps.append((cursor, max_time))

    iti_windows = []
    for gap_start, gap_end in gaps:
        current_start = gap_start
        while current_start + window_duration_ms <= gap_end:
            current_stop = current_start + window_duration_ms
            iti_windows.append([int(round(current_start)), int(round(current_stop))])
            current_start = current_stop

    return iti_windows


def _get_time_bounds_for_recording(recording, time_bounds):
    """Allow one shared bounds tuple or a recording-name-to-bounds dictionary."""
    if isinstance(time_bounds, dict):
        return time_bounds.get(recording.name)
    return time_bounds


def _recording_to_iti_dict(recording, window_duration=2.0, time_bounds=None):
    """Return one recording's ITI windows in event-dictionary format."""
    window_key = _iti_window_key(window_duration)
    recording_time_bounds = _get_time_bounds_for_recording(recording, time_bounds)
    return {
        window_key: _get_iti_windows_for_recording(
            recording,
            window_duration=window_duration,
            time_bounds=recording_time_bounds,
        )
    }


def _update_recording_iti_event(recording, iti_dict):
    """Write generated ITI windows into one recording's event dictionary."""
    if getattr(recording, "event_dict", None) is None:
        recording.event_dict = {}

    for event_name, windows in iti_dict.items():
        recording.event_dict[event_name] = np.asarray(windows, dtype=float)


def _update_collection_event_metadata(lfp_collection, recording, iti_dict):
    """Keep collection-level event metadata aligned with recording.event_dict."""
    if not hasattr(lfp_collection, "recording_to_event_dict"):
        return

    if lfp_collection.recording_to_event_dict is None:
        lfp_collection.recording_to_event_dict = {}

    if recording.name not in lfp_collection.recording_to_event_dict:
        lfp_collection.recording_to_event_dict[recording.name] = {}

    for event_name, windows in iti_dict.items():
        lfp_collection.recording_to_event_dict[recording.name][event_name] = np.asarray(
            windows,
            dtype=float,
        )


def _is_lfp_collection(lfp_object):
    """Check for the attributes needed from an LFPCollection-like object."""
    return hasattr(lfp_object, "recordings")


def _is_lfp_recording(lfp_object):
    """Check for the attributes needed from an LFPRecording-like object."""
    return hasattr(lfp_object, "event_dict") and hasattr(lfp_object, "name")


def get_iti_windows(lfp_object, window_duration=2.0, time_bounds=None, update_collection=False):
    """Return non-overlapping ITI windows from an LFPRecording or LFPCollection.

    Stored behavior events are expected to be in milliseconds. ``window_duration`` is
    specified in seconds, and returned ITI windows are integer millisecond pairs.
    """
    if _is_lfp_recording(lfp_object):
        iti_dict = _recording_to_iti_dict(
            lfp_object,
            window_duration=window_duration,
            time_bounds=time_bounds,
        )
        if update_collection:
            _update_recording_iti_event(lfp_object, iti_dict)
        return iti_dict

    if _is_lfp_collection(lfp_object):
        collection_iti = {}
        for recording in lfp_object.recordings:
            iti_dict = _recording_to_iti_dict(
                recording,
                window_duration=window_duration,
                time_bounds=time_bounds,
            )
            collection_iti[recording.name] = iti_dict

            if update_collection:
                _update_recording_iti_event(recording, iti_dict)
                _update_collection_event_metadata(lfp_object, recording, iti_dict)

        return collection_iti

    raise ValueError("lfp_object must be an LFPRecording or LFPCollection-like object.")
