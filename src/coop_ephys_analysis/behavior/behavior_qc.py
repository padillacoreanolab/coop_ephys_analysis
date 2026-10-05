"""Utilities for quality-control checks on behavior events."""
import external.diff_fam_social_memory_ephys.behavior.behavioral_epoch_tools as bet
import matplotlib.pyplot as plt
import numpy as np


_FILTERED_NOSE_POKE_PAIRS = (
    ("selfish nose poke", "selfish light", "filtered selfish nose pokes"),
    ("coop nose poke", "coop light", "filtered coop nose pokes"),
)


def threshold_stage_behaviors(stage_list, behaviors, min_iti, min_bout):
    """
    Threshold selected behaviors across every recording in a stage list.

    Parameters:
        stage_list (list): List of dictionaries where each dictionary corresponds
            to a day/stage. Each is keyed by recording name and maps to a dict of
            behavior names to an (N, 2) array of start/stop times.
        behaviors (list): Behavior names to threshold in every recording.
        min_iti (float): Minimum inter-bout interval in seconds. Bouts separated
            by less than this value are combined.
        min_bout (float): Minimum bout length in seconds. Shorter bouts are removed.

    Returns:
        list: The same stage list, updated in place.
    """
    for day_data in stage_list:
        for behavior_dict in day_data.values():
            for behavior in behaviors:
                behavior_dict[behavior] = bet.threshold_bouts(
                    behavior_dict[behavior],
                    min_iti=min_iti,
                    min_bout=min_bout,
                )

    return stage_list


def count_events(
    behaviors_dict,
    mode="multiple recordings",
    stage=None,
    day=None,
    recording_name=None,
):
    """
    Count and print the number of events for each behavior.

    Parameters:
        behaviors_dict (dict): Behavior events for either one recording or multiple
            recordings. For a single recording, the expected structure is
            {behavior_name: events}. For multiple recordings, the expected structure
            is {recording_name: {behavior_name: events}}.
        mode (str): Either "multiple recordings" or "single recording".
        stage (int or str): Stage label to print for multiple-recording summaries.
        day (int or str): Day label to print for multiple-recording summaries.
        recording_name (str): Recording name to print for single-recording summaries.

    Returns:
        None.
    """
    if mode == "multiple recordings":
        if stage is None or day is None:
            raise ValueError(
                "stage and day are required when mode is 'multiple recordings'."
            )
        _count_multiple_recordings(behaviors_dict, stage=stage, day=day)
        return None

    if mode == "single recording":
        if recording_name is None:
            raise ValueError(
                "recording_name is required when mode is 'single recording'."
            )
        _count_single_recording(
            behaviors_dict,
            recording_name=recording_name,
        )
        return None

    raise ValueError("mode must be either 'multiple recordings' or 'single recording'.")


def events_duration_distribution(
    behaviors_dict,
    behavior,
    xmin=0,
    xmax=6,
    mode="multiple recordings",
    stage=4,
    day=0,
):
    """
    Plot a histogram of event durations for a given behavior.

    Parameters:
        behaviors_dict (dict): Behavior events for either one recording or multiple
            recordings. For a single recording, the expected structure is
            {behavior_name: events}. For multiple recordings, the expected structure
            is {recording_name: {behavior_name: events}}.
        behavior (str): The behavior to plot.
        xmin (float): Minimum duration shown on the histogram.
        xmax (float): Maximum duration shown on the histogram.
        mode (str): Either "multiple recordings" or "single recording".
        stage (int or str): Stage label for the plot title.
        day (int or str): Day label for the plot title.

    Returns:
        None.
    """
    if mode == "single recording":
        behavior_array = np.array(behaviors_dict[behavior])
    elif mode == "multiple recordings":
        behavior_array = []
        for rec, rec_behaviors in behaviors_dict.items():
            if len(rec_behaviors[behavior]) > 0:
                for event in rec_behaviors[behavior]:
                    behavior_array.append(event)
        behavior_array = np.array(behavior_array)

    durations = behavior_array[:, 1] - behavior_array[:, 0]

    plt.figure(figsize=(12, 5))
    plt.hist(durations, bins=30, range=(xmin, xmax), color="#15616F", edgecolor="black")
    plt.title(
        f"Distribution of Event Durations - {mode} - Stage{stage} Day{day}",
        fontsize=16,
    )
    plt.xlabel(f"{behavior} event durations (seconds)", fontsize=14)
    plt.ylabel("# of Events", fontsize=14)
    ticks = np.linspace(xmin, xmax, 10)
    plt.xticks(np.round(ticks, 2))
    plt.show()


def between_events_duration_distribution(
    behaviors_dict,
    behaviors,
    xmin=0,
    xmax=10,
    mode="multiple recordings",
    stage=4,
    day=0,
):
    """
    Plot a histogram of durations between separate events.

    Parameters:
        behaviors_dict (dict): Behavior events for either one recording or multiple
            recordings. For a single recording, the expected structure is
            {behavior_name: events}. For multiple recordings, the expected structure
            is {recording_name: {behavior_name: events}}.
        behaviors (list): Behaviors to include in the between-event interval plot.
        xmin (float): Minimum interval shown on the histogram.
        xmax (float): Maximum interval shown on the histogram.
        mode (str): Either "multiple recordings" or "single recording".
        stage (int or str): Stage label for the plot title.
        day (int or str): Day label for the plot title.

    Returns:
        None.
    """
    behavior_array = []
    if mode == "single recording":
        for behavior in behaviors:
            for event in behaviors_dict[behavior]:
                behavior_array.append(event)
        behavior_array.sort(key=lambda x: x[0])
        total_between_event_durations = (
            np.array(behavior_array)[1:, 0] - np.array(behavior_array)[:-1, 1]
        )

    elif mode == "multiple recordings":
        total_between_event_durations = np.array([])
        for rec, rec_behaviors in behaviors_dict.items():
            # skip 6.1 and 6.3 for recipient port entry since they were trained alone
            if (
                rec.split("_")[4] == "6-1" or rec.split("_")[4] == "6-3"
            ) and "recipient port entry" in behaviors:
                continue

            behavior_array = []
            for behavior in behaviors:
                for event in rec_behaviors[behavior]:
                    behavior_array.append(event)
            behavior_array.sort(key=lambda x: x[0])
            rec_between_event_durations = (
                np.array(behavior_array)[1:, 0] - np.array(behavior_array)[:-1, 1]
            )
            total_between_event_durations = np.concatenate(
                (total_between_event_durations, rec_between_event_durations)
            )

    plt.figure(figsize=(12, 5))
    plt.hist(
        total_between_event_durations,
        bins=50,
        range=(xmin, xmax),
        color="#FFAF00",
        edgecolor="black",
    )
    plt.title(
        f"Distribution of Between-event Durations - {mode} - Stage{stage} Day{day}",
        fontsize=16,
    )
    plt.xlabel(f"Time Gaps Between {' & '.join(behaviors)}", fontsize=14)
    plt.ylabel("# of Events", fontsize=14)
    ticks = np.linspace(xmin, xmax, 20)
    plt.xticks(np.round(ticks, 2))
    plt.show()


_DEFAULT_BEHAVIOR_ORDER = (
    "selfish light",
    "selfish nose poke",
    "subject port entry",
    "coop light",
    "coop nose poke",
    "recipient port entry",
)
_BASELINE_ITI_COLOR = "#B0B0B0"


def plot_recordings_behaviors(
    stage_list,
    behavior_order=_DEFAULT_BEHAVIOR_ORDER,
    time_window=None,
    day=None,
    mode="multiple recordings",
    recording_name=None,
):
    """Plot one figure per recording showing every behavior as a horizontal bar.

    Parameters:
        stage_list (list): List of dictionaries where each dictionary corresponds
            to a day/stage. Each is keyed by recording name and maps to a dict of
            behavior names to an (N, 2) array of start/stop times.
        behavior_order (sequence): Explicit order of behaviors to display. When
            omitted, the standard behaviors are shown followed by any saved
            baseline ITI events. If None, all events in each recording are shown.
        time_window (tuple): If provided, only events that fall within the window
            are shown and the x-axis limits are set accordingly. Use (xmin, xmax).
        day (int): If provided, limits plotting to that one day using a 1-based
            index. None plots all days in the list.
        mode (str): Either "multiple recordings" or "single recording".
        recording_name (str): Recording to plot when mode is "single recording".

    Returns:
        None.
    """
    def _plot_single(rec_name, rec_beh):
        baseline_events = sorted(
            (event for event in rec_beh if _is_baseline_iti_event(event)),
            key=_baseline_iti_sort_key,
        )
        if behavior_order is _DEFAULT_BEHAVIOR_ORDER:
            plot_order = list(_DEFAULT_BEHAVIOR_ORDER)
            plot_order.extend(
                event for event in baseline_events if event not in plot_order
            )
        elif behavior_order is None:
            ordinary_events = sorted(
                event for event in rec_beh if not _is_baseline_iti_event(event)
            )
            plot_order = ordinary_events + baseline_events
        else:
            plot_order = list(behavior_order)

        ordered = {b: np.array(rec_beh.get(b, [])) for b in plot_order}
        n_beh = len(ordered)
        fig, ax = plt.subplots(figsize=(10, 0.6 * n_beh))
        color_map = {
            label: (
                _BASELINE_ITI_COLOR
                if _is_baseline_iti_event(label)
                else [np.random.random() for _ in range(3)]
            )
            for label in ordered
        }

        yticks = []
        yticklabels = []
        for i, (label, events) in enumerate(ordered.items()):
            y = n_beh - 1 - i
            for start, stop in events:
                if time_window is not None:
                    xmin, xmax = time_window
                    if stop < xmin or start > xmax:
                        continue
                    start = max(start, xmin)
                    stop = min(stop, xmax)
                ax.barh(
                    y,
                    stop - start,
                    left=start,
                    height=0.4,
                    color=color_map[label],
                    edgecolor="black",
                )
            yticks.append(y)
            yticklabels.append(label)

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Time")
        if time_window is not None:
            ax.set_xlim(time_window)
        ax.set_title(rec_name)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    days_to_plot = stage_list
    if day is not None:
        idx = day - 1
        if idx < 0 or idx >= len(stage_list):
            raise ValueError(f"day must be between 1 and {len(stage_list)}")
        days_to_plot = [stage_list[idx]]

    if mode == "multiple recordings":
        for stage_idx, stage in enumerate(days_to_plot, start=(day or 1)):
            for rec, rec_beh in stage.items():
                _plot_single(rec, rec_beh)
        return None

    if mode == "single recording":
        if recording_name is None:
            raise ValueError(
                "recording_name is required when mode is 'single recording'."
            )

        for stage_idx, stage in enumerate(days_to_plot, start=(day or 1)):
            if recording_name in stage:
                _plot_single(f"Stage{stage_idx} {recording_name}", stage[recording_name])
                return None

        raise ValueError(f"Recording not found: {recording_name}")

    raise ValueError("mode must be either 'multiple recordings' or 'single recording'.")


def find_filtered_nose_pokes(
    behaviors,
    max_gap=1.0,
    stage=None,
    day=None,
    save=False,
):
    """Find nose-poke bouts associated with their corresponding light onset.

    A light can match an unused nose poke when it begins during the poke or no
    more than ``max_gap`` seconds after the poke ends. Lights are processed in
    chronological order and select the closest eligible unused poke. The input
    hierarchy is preserved in the returned result.

    Parameters:
        behaviors (list or dict): A stage list, day dictionary, or one recording's
            behavior dictionary.
        max_gap (float): Maximum seconds allowed between a poke ending and its
            corresponding light beginning.
        stage (int or str): Optional stage label retained for API consistency.
        day (int): Optional 1-based day to process when ``behaviors`` is a stage list.
        save (bool): If True, copy filtered arrays into the supplied behavior
            dictionaries under ``filtered selfish nose pokes`` and
            ``filtered coop nose pokes``.

    Returns:
        dict or list: Filtered event arrays following the input hierarchy.
    """
    del stage
    max_gap = float(max_gap)
    if not np.isfinite(max_gap) or max_gap < 0:
        raise ValueError("max_gap must be a finite value greater than or equal to 0.")

    if isinstance(behaviors, list):
        if day is not None:
            day_index = day - 1
            if day_index < 0 or day_index >= len(behaviors):
                raise ValueError(f"day must be between 1 and {len(behaviors)}")
            return _filter_nose_pokes_for_day(
                behaviors[day_index],
                max_gap=max_gap,
                save=save,
            )

        return [
            _filter_nose_pokes_for_day(day_data, max_gap=max_gap, save=save)
            for day_data in behaviors
        ]

    if isinstance(behaviors, dict):
        if _is_recording_behaviors(behaviors):
            filtered = _filter_nose_pokes_for_recording(
                behaviors,
                max_gap=max_gap,
            )
            if save:
                _save_filtered_nose_pokes(behaviors, filtered)
            return filtered

        return _filter_nose_pokes_for_day(
            behaviors,
            max_gap=max_gap,
            save=save,
        )

    raise ValueError(
        "behaviors must be a stage list, one day dictionary, or one recording dictionary."
    )


def _filter_nose_pokes_for_day(day_data, max_gap, save):
    filtered_day = {}
    for recording_name, recording_behaviors in day_data.items():
        filtered = _filter_nose_pokes_for_recording(
            recording_behaviors,
            max_gap=max_gap,
        )
        if save:
            _save_filtered_nose_pokes(recording_behaviors, filtered)
        filtered_day[recording_name] = filtered
    return filtered_day


def _filter_nose_pokes_for_recording(recording_behaviors, max_gap):
    filtered = {}
    for nose_poke_name, light_name, filtered_name in _FILTERED_NOSE_POKE_PAIRS:
        nose_pokes = _valid_behavior_intervals(
            recording_behaviors.get(nose_poke_name),
            event_name=nose_poke_name,
        )
        lights = _valid_behavior_intervals(
            recording_behaviors.get(light_name),
            event_name=light_name,
        )
        filtered[filtered_name] = _match_nose_pokes_to_lights(
            nose_pokes,
            lights,
            max_gap=max_gap,
        )
    return filtered


def _valid_behavior_intervals(events, event_name):
    if events is None:
        return np.empty((0, 2), dtype=float)

    event_array = np.asarray(events, dtype=float)
    if event_array.size == 0:
        return np.empty((0, 2), dtype=float)

    event_array = np.atleast_2d(event_array)
    if event_array.ndim != 2 or event_array.shape[1] != 2:
        raise ValueError(f"Behavior {event_name!r} must contain [start, stop] pairs.")

    valid_rows = (
        np.isfinite(event_array).all(axis=1)
        & (event_array[:, 1] > event_array[:, 0])
    )
    return event_array[valid_rows].copy()


def _match_nose_pokes_to_lights(nose_pokes, lights, max_gap):
    if len(nose_pokes) == 0 or len(lights) == 0:
        return np.empty((0, 2), dtype=float)

    nose_order = np.lexsort((nose_pokes[:, 1], nose_pokes[:, 0]))
    light_order = np.lexsort((lights[:, 1], lights[:, 0]))
    sorted_nose_pokes = nose_pokes[nose_order]
    sorted_lights = lights[light_order]
    used_nose_pokes = set()
    matched_nose_pokes = []

    for light_start, _ in sorted_lights:
        eligible = []
        for poke_index, (poke_start, poke_stop) in enumerate(sorted_nose_pokes):
            if poke_index in used_nose_pokes:
                continue
            if poke_start <= light_start <= poke_stop + max_gap:
                separation = max(0.0, light_start - poke_stop)
                eligible.append(
                    (separation, -poke_stop, -poke_start, poke_index)
                )

        if not eligible:
            continue

        _, _, _, matched_index = min(eligible)
        used_nose_pokes.add(matched_index)
        matched_nose_pokes.append(sorted_nose_pokes[matched_index].copy())

    if not matched_nose_pokes:
        return np.empty((0, 2), dtype=float)

    matched = np.asarray(matched_nose_pokes, dtype=float)
    return matched[np.lexsort((matched[:, 1], matched[:, 0]))]


def _save_filtered_nose_pokes(recording_behaviors, filtered):
    for event_name, event_array in filtered.items():
        recording_behaviors[event_name] = event_array.copy()


def find_iti_windows(
    behaviors,
    window_duration=2.0,
    time_bounds=None,
    stage=None,
    day=None,
    buffer=0.5,
    save=False,
):
    """
    Find and optionally save non-overlapping ITI windows.

    ITI windows are defined as periods where no behavioral events occur.
    Each event is expanded by ``buffer`` seconds on both sides before overlapping
    events are merged and gaps are detected.

    Parameters:
        behaviors (list or dict): Either an entire stage list shaped like
            [{recording_name: {behavior_name: events}}, ...], one day dictionary
            shaped like {recording_name: {behavior_name: events}}, or one
            recording dictionary shaped like {behavior_name: events}.
        window_duration (float): Duration of ITI window to find. The default is
            2.0 seconds.
        time_bounds (tuple): Recording time bounds (min_time, max_time). If None,
            uses the data range.
        stage (int or str): Stage label accepted for consistency with other QC
            functions.
        day (int): Optional 1-based day index for stage inputs.
        buffer (float): Seconds excluded before and after every annotated event.
            The default is 0.5 seconds, matching half of the current 1-second
            LFP spectral window.
        save (bool): If True, add the windows to each recording's behavior
            dictionary under ``"baseline iti Xs"``. This updates the supplied
            dictionaries in memory; it does not write a pickle file.

    Returns:
        int or None: Returns an ITI window count when given one recording.
        Prints summaries and returns None when given one day or a stage list.
        When ``save=True``, the input behavior dictionaries are also updated.
    """
    if isinstance(behaviors, list):
        print("=" * 60)
        print(f"{window_duration}-SECOND ITI WINDOW ANALYSIS")
        print("=" * 60)

        if day is not None:
            idx = day - 1
            if idx < 0 or idx >= len(behaviors):
                raise ValueError(f"day must be between 1 and {len(behaviors)}")
            _print_iti_windows_for_day(
                behaviors[idx],
                day_idx=day,
                window_duration=window_duration,
                time_bounds=time_bounds,
                buffer=buffer,
                save=save,
            )
            return None

        for day_idx, day_data in enumerate(behaviors, start=1):
            _print_iti_windows_for_day(
                day_data,
                day_idx=day_idx,
                window_duration=window_duration,
                time_bounds=time_bounds,
                buffer=buffer,
                save=save,
            )
        return None

    if isinstance(behaviors, dict):
        if _is_recording_behaviors(behaviors):
            windows = _find_iti_windows_for_recording(
                behaviors,
                window_duration=window_duration,
                time_bounds=time_bounds,
                buffer=buffer,
            )
            if save:
                behaviors[_baseline_iti_event_name(window_duration)] = windows
            return len(windows)

        print("=" * 60)
        print(f"{window_duration}-SECOND ITI WINDOW ANALYSIS")
        print("=" * 60)

        _print_iti_windows_for_day(
            behaviors,
            day_idx=day,
            window_duration=window_duration,
            time_bounds=time_bounds,
            buffer=buffer,
            save=save,
        )
        return None

    raise ValueError(
        "behaviors must be a stage list, one day dictionary, or one recording dictionary."
    )


def _is_recording_behaviors(behaviors):
    return all(not isinstance(events, dict) for events in behaviors.values())


def _print_iti_windows_for_day(
    day_data,
    day_idx,
    window_duration,
    time_bounds,
    buffer,
    save,
):
    if day_idx is not None:
        print(f"\n--- Day {day_idx} ---")

    for recording_name, recording_behaviors in day_data.items():
        windows = _find_iti_windows_for_recording(
            recording_behaviors,
            window_duration=window_duration,
            time_bounds=time_bounds,
            buffer=buffer,
        )
        if save:
            recording_behaviors[_baseline_iti_event_name(window_duration)] = windows
        print(f"{recording_name}: {len(windows)} {window_duration}s ITI windows")


def _count_iti_windows_for_recording(
    recording_behaviors,
    window_duration=2.0,
    time_bounds=None,
    buffer=0.5,
):
    return len(
        _find_iti_windows_for_recording(
            recording_behaviors,
            window_duration=window_duration,
            time_bounds=time_bounds,
            buffer=buffer,
        )
    )


def _baseline_iti_event_name(window_duration):
    duration = float(window_duration)
    duration_label = str(int(duration)) if duration.is_integer() else f"{duration:g}"
    return f"baseline iti {duration_label}s"


def _is_baseline_iti_event(event_name):
    return str(event_name).casefold().startswith("baseline iti ")


def _baseline_iti_sort_key(event_name):
    duration_label = str(event_name).rsplit(" ", maxsplit=1)[-1]
    try:
        duration = float(duration_label.removesuffix("s"))
    except ValueError:
        duration = float("inf")
    return duration, str(event_name)


def _find_iti_windows_for_recording(
    recording_behaviors,
    window_duration=2.0,
    time_bounds=None,
    buffer=0.5,
):
    if window_duration <= 0:
        raise ValueError("window_duration must be greater than 0 seconds.")
    if buffer < 0:
        raise ValueError("buffer must be greater than or equal to 0 seconds.")

    all_events = []
    for behavior_name, events in recording_behaviors.items():
        if _is_baseline_iti_event(behavior_name) or events is None:
            continue

        events = np.asarray(events, dtype=float)
        if events.size == 0:
            continue
        events = np.atleast_2d(events)
        if events.shape[1] < 2:
            raise ValueError(f"Behavior {behavior_name!r} must contain [start, stop] pairs.")

        valid_events = events[:, :2]
        valid_events = valid_events[
            np.isfinite(valid_events).all(axis=1)
            & (valid_events[:, 1] > valid_events[:, 0])
        ]
        all_events.extend(valid_events.tolist())

    if time_bounds is None:
        if not all_events:
            return np.empty((0, 2), dtype=float)
        min_time = 0.0
        max_time = max(event[1] for event in all_events) + 10.0
    else:
        min_time, max_time = [float(value) for value in time_bounds]
        if max_time <= min_time:
            raise ValueError("time_bounds must satisfy max_time > min_time.")

    if not all_events:
        merged_events = np.empty((0, 2), dtype=float)
    else:
        padded_events = np.asarray(all_events, dtype=float)
        padded_events[:, 0] = np.maximum(padded_events[:, 0] - buffer, min_time)
        padded_events[:, 1] = np.minimum(padded_events[:, 1] + buffer, max_time)
        padded_events = padded_events[padded_events[:, 1] > padded_events[:, 0]]
        padded_events = padded_events[np.argsort(padded_events[:, 0])]

        if len(padded_events) == 0:
            merged_events = np.empty((0, 2), dtype=float)
        else:
            merged_events = [padded_events[0].tolist()]
            for current_start, current_stop in padded_events[1:]:
                last_event = merged_events[-1]
                if current_start <= last_event[1]:
                    last_event[1] = max(last_event[1], current_stop)
                else:
                    merged_events.append([current_start, current_stop])
            merged_events = np.asarray(merged_events, dtype=float)

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
        while current_start + window_duration <= gap_end:
            current_stop = current_start + window_duration
            iti_windows.append([current_start, current_stop])
            current_start = current_stop

    return np.asarray(iti_windows, dtype=float).reshape(-1, 2)


def _count_multiple_recordings(behaviors_dict, stage, day):
    print(f"Stage {stage} Day {day}")
    print("-" * 20)

    for recording_name, recording_behaviors in behaviors_dict.items():
        print(recording_name)
        _count_behaviors(recording_behaviors)
        print()


def _count_single_recording(behaviors_dict, recording_name):
    print(recording_name)
    _count_behaviors(behaviors_dict)


def _count_behaviors(behaviors_dict):
    for behavior_name, events in behaviors_dict.items():
        count = len(events)
        print("#", behavior_name, "-", count)
