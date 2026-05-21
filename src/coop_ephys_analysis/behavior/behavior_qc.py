"""Utilities for quality-control checks on behavior events."""
import matplotlib.pyplot as plt
import numpy as np


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

    print(total_between_event_durations.size)

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


def plot_recordings_behaviors(
    stage_list,
    behavior_order=None,
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
        behavior_order (list): Explicit order of behaviors to display. If None,
            the union of all behaviors across recordings is used.
        time_window (tuple): If provided, only events that fall within the window
            are shown and the x-axis limits are set accordingly. Use (xmin, xmax).
        day (int): If provided, limits plotting to that one day using a 1-based
            index. None plots all days in the list.
        mode (str): Either "multiple recordings" or "single recording".
        recording_name (str): Recording to plot when mode is "single recording".

    Returns:
        None.
    """
    if behavior_order is None:
        behaviors = set()
        for stage in stage_list:
            for rec_dict in stage.values():
                behaviors.update(rec_dict.keys())
        behavior_order = sorted(behaviors)

    def _plot_single(rec_name, rec_beh):
        ordered = {b: np.array(rec_beh.get(b, [])) for b in behavior_order}
        n_beh = len(ordered)
        fig, ax = plt.subplots(figsize=(10, 0.6 * n_beh))
        color_map = {label: [np.random.random() for _ in range(3)] for label in ordered}

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
                _plot_single(f"Stage{stage_idx} {rec}", rec_beh)
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


def find_iti_windows(behaviors, window_duration=2.0, time_bounds=None, stage=None, day=None):
    """
    Print non-overlapping ITI window counts for a day or stage.

    ITI windows are defined as periods where no behavioral events occur.
    Overlapping events are merged before gap detection.

    Parameters:
        behaviors (list or dict): Either an entire stage list shaped like
            [{recording_name: {behavior_name: events}}, ...] or one day dictionary
            shaped like {recording_name: {behavior_name: events}}.
        window_duration (float): Duration of ITI window to find. The default is
            2.0 seconds.
        time_bounds (tuple): Recording time bounds (min_time, max_time). If None,
            uses the data range.
        stage (int or str): Stage label accepted for consistency with other QC
            functions.
        day (int): Optional 1-based day index for stage inputs.

    Returns:
        None.
    """
    print("=" * 60)
    print(f"{window_duration}-SECOND ITI WINDOW ANALYSIS")
    print("=" * 60)

    if isinstance(behaviors, list):
        if day is not None:
            idx = day - 1
            if idx < 0 or idx >= len(behaviors):
                raise ValueError(f"day must be between 1 and {len(behaviors)}")
            _print_iti_windows_for_day(
                behaviors[idx],
                day_idx=day,
                window_duration=window_duration,
                time_bounds=time_bounds,
            )
            return None

        for day_idx, day_data in enumerate(behaviors, start=1):
            _print_iti_windows_for_day(
                day_data,
                day_idx=day_idx,
                window_duration=window_duration,
                time_bounds=time_bounds,
            )
        return None

    if isinstance(behaviors, dict):
        _print_iti_windows_for_day(
            behaviors,
            day_idx=day,
            window_duration=window_duration,
            time_bounds=time_bounds,
        )
        return None

    raise ValueError("behaviors must be either a stage list or one day dictionary.")


def _print_iti_windows_for_day(day_data, day_idx, window_duration, time_bounds):
    if day_idx is not None:
        print(f"\n--- Day {day_idx} ---")

    for recording_name, recording_behaviors in day_data.items():
        num_windows = _count_iti_windows_for_recording(
            recording_behaviors,
            window_duration=window_duration,
            time_bounds=time_bounds,
        )
        print(f"{recording_name}: {num_windows} {window_duration}s ITI windows")


def _count_iti_windows_for_recording(
    recording_behaviors,
    window_duration=2.0,
    time_bounds=None,
):
    all_events = []
    for behavior_name, events in recording_behaviors.items():
        if len(events) > 0:
            all_events.extend([tuple(event) for event in events])

    if not all_events:
        return 0

    all_events.sort(key=lambda x: x[0])
    all_events = np.array(all_events)

    merged_events = [all_events[0]]
    for current_event in all_events[1:]:
        last_event = merged_events[-1]
        if current_event[0] < last_event[1]:
            merged_events[-1] = [last_event[0], max(last_event[1], current_event[1])]
        else:
            merged_events.append(current_event)

    merged_events = np.array(merged_events)

    if time_bounds is None:
        min_time = 0
        max_time = merged_events[:, 1].max() + 10
    else:
        min_time, max_time = time_bounds

    gaps = []

    for i in range(len(merged_events) - 1):
        gap_start = merged_events[i, 1]
        gap_end = merged_events[i + 1, 0]
        if gap_end > gap_start:
            gaps.append((gap_start, gap_end))

    gaps.append((merged_events[-1, 1], max_time))

    total_windows = 0
    for gap_start, gap_end in gaps:
        gap_duration = gap_end - gap_start
        if gap_duration >= window_duration:
            num_windows = int(gap_duration // window_duration)
            total_windows += num_windows

    return total_windows


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
