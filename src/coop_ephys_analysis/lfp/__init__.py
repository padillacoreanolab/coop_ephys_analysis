"""LFP helpers for coop_ephys_analysis."""

from .analysis import (
    average_events_by_condition,
    condition_event_spectrum,
    get_lfp_collection_class,
    load_lfp_collection,
    plot_collection_event_spectrogram,
    plot_collection_spectral_traces,
    plot_recording_event_spectrogram,
    plot_recording_spectral_traces,
)
from .processing import (
    DEFAULT_RECIPIENT_IDS,
    DEFAULT_SUBJECT_ALONE_IDS,
    DEFAULT_SUBJECT_WITH_RECIPIENT_IDS,
    build_condition_dict,
    get_iti_windows,
    parse_animal_id,
    select_recordings_by_condition,
    summarize_event_windows,
)

__all__ = [
    "DEFAULT_RECIPIENT_IDS",
    "DEFAULT_SUBJECT_ALONE_IDS",
    "DEFAULT_SUBJECT_WITH_RECIPIENT_IDS",
    "average_events_by_condition",
    "build_condition_dict",
    "condition_event_spectrum",
    "get_iti_windows",
    "get_lfp_collection_class",
    "load_lfp_collection",
    "parse_animal_id",
    "plot_collection_event_spectrogram",
    "plot_collection_spectral_traces",
    "plot_recording_event_spectrogram",
    "plot_recording_spectral_traces",
    "select_recordings_by_condition",
    "summarize_event_windows",
]
