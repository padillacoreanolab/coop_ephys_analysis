from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coop_ephys_analysis.lfp import (  # noqa: E402
    average_events_by_condition,
    build_condition_dict,
    get_iti_windows,
    plot_recording_event_spectrogram,
    plot_recording_spectral_traces,
    select_recordings_by_condition,
    summarize_event_windows,
)


class RegionDict(dict):
    @property
    def inverse(self):
        return {value: key for key, value in self.items()}


def make_recording(name, events=None):
    region_dict = RegionDict({"mPFC": 0, "BLA": 1})
    return SimpleNamespace(
        name=name,
        event_dict=events or {},
        timestep=0.5,
        frequencies=np.array([1, 4, 8, 12], dtype=float),
        brain_region_dict=region_dict,
        traces=np.zeros((5000, 2)),
        resample_rate=1000,
        power=np.ones((10, 4, 2)),
        coherence=np.ones((10, 4, 2, 2)),
        granger=np.ones((10, 4, 2, 2)),
    )


class LfpHelperTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_condition_grouping_assigns_stage4_roles_and_unknown(self):
        collection = SimpleNamespace(
            recordings=[
                make_recording("20250508_100203_Stage4_D1_1-3_merged.rec"),
                make_recording("20250508_100203_Stage4_D1_1-2_merged.rec"),
                make_recording("20250508_112121_Stage4_D1_6-1_merged.rec"),
                make_recording("20250508_112121_Stage4_D1_9-9_merged.rec"),
            ]
        )

        condition_dict = build_condition_dict(collection)

        self.assertEqual(condition_dict["recipient"], [collection.recordings[0].name])
        self.assertEqual(
            condition_dict["subject with recipient"], [collection.recordings[1].name]
        )
        self.assertEqual(condition_dict["subject alone"], [collection.recordings[2].name])
        self.assertEqual(condition_dict["unknown"], [collection.recordings[3].name])
        self.assertEqual(collection.recordings[2].condition, "subject alone")

    def test_select_recordings_supports_one_or_multiple_conditions(self):
        recordings = [make_recording("a.rec"), make_recording("b.rec"), make_recording("c.rec")]
        collection = SimpleNamespace(recordings=recordings)
        condition_dict = {"first": ["a.rec"], "second": ["b.rec"]}

        selected = select_recordings_by_condition(collection, condition_dict, "first")
        self.assertEqual([recording.name for recording in selected], ["a.rec"])

        selected = select_recordings_by_condition(
            collection, condition_dict, ["first", "second"]
        )
        self.assertEqual([recording.name for recording in selected], ["a.rec", "b.rec"])

    def test_select_recordings_raises_for_missing_or_unmatched_condition(self):
        collection = SimpleNamespace(recordings=[make_recording("a.rec")])

        with self.assertRaises(ValueError):
            select_recordings_by_condition(collection, {"first": ["a.rec"]}, "missing")

        with self.assertRaises(ValueError):
            select_recordings_by_condition(collection, {"first": ["b.rec"]}, "first")

    def test_summarize_event_windows_reports_counts_and_widths(self):
        recording = make_recording(
            "a.rec",
            events={"evt": np.array([[0, 500], [500, 500], [4500, 5500]])},
        )
        collection = SimpleNamespace(recordings=[recording])

        qc_df = summarize_event_windows(
            collection,
            {"first": ["a.rec"]},
            "first",
            events=["evt"],
            mode="power",
        )

        row = qc_df.iloc[0]
        self.assertEqual(row["event_count"], 3)
        self.assertEqual(row["duration_min"], 0)
        self.assertEqual(row["duration_max"], 1000)
        self.assertEqual(row["bin_width_min"], 0)
        self.assertEqual(row["bin_width_max"], 2)
        self.assertEqual(row["within_recording_count"], 2)
        self.assertEqual(row["nonempty_count"], 1)
        self.assertEqual(row["zero_width_count"], 1)

    def test_get_iti_windows_returns_non_overlapping_ms_windows(self):
        recording = make_recording(
            "a.rec",
            events={"event": np.array([[1000, 2000], [3000, 3500]], dtype=float)},
        )

        iti_dict = get_iti_windows(
            recording,
            window_duration=1.0,
            time_bounds=(0, 5000),
        )

        self.assertEqual(
            iti_dict,
            {"1s_windows": [[0, 1000], [2000, 3000], [3500, 4500]]},
        )

    def test_get_iti_windows_update_collection_writes_numpy_arrays(self):
        recording = make_recording(
            "a.rec",
            events={"event": np.array([[1000, 2000]], dtype=float)},
        )
        collection = SimpleNamespace(recordings=[recording], recording_to_event_dict={})

        result = get_iti_windows(
            collection,
            window_duration=1.0,
            time_bounds={"a.rec": (0, 3000)},
            update_collection=True,
        )

        self.assertEqual(result["a.rec"], {"1s_windows": [[0, 1000], [2000, 3000]]})
        self.assertIsInstance(recording.event_dict["1s_windows"], np.ndarray)
        self.assertIsInstance(
            collection.recording_to_event_dict["a.rec"]["1s_windows"],
            np.ndarray,
        )

    def test_plot_recording_spectral_traces_returns_axes_for_modes(self):
        recording = make_recording("a.rec")
        collection = SimpleNamespace(recordings=[recording])

        power_axes = plot_recording_spectral_traces(
            collection,
            "a.rec",
            mode="power",
            freq_range=(4, 12),
            regions=["mPFC"],
        )
        coherence_axes = plot_recording_spectral_traces(
            collection,
            recording,
            mode="coherence",
            freq_range=(4, 12),
            regions=["mPFC"],
        )
        granger_axes = plot_recording_spectral_traces(
            collection,
            recording,
            mode="granger",
            freq_range=(4, 12),
            regions=["mPFC"],
        )

        self.assertEqual(len(power_axes), 1)
        self.assertEqual(len(coherence_axes), 1)
        self.assertEqual(len(granger_axes), 1)

    def test_plot_recording_event_spectrogram_skips_invalid_windows_and_raises_when_empty(self):
        recording = make_recording(
            "a.rec",
            events={
                "evt": np.array([[0, 1000], [1000, 1000], [9000, 10000]], dtype=float),
                "empty_evt": np.array([[1000, 1000]], dtype=float),
            },
        )
        collection = SimpleNamespace(recordings=[recording])

        axes = plot_recording_event_spectrogram(
            collection,
            recording,
            event="evt",
            mode="power",
            freq_range=(4, 12),
            regions=["mPFC"],
        )
        self.assertEqual(len(axes), 1)

        with self.assertRaisesRegex(ValueError, "no valid windows"):
            plot_recording_event_spectrogram(
                collection,
                recording,
                event="empty_evt",
                mode="power",
                freq_range=(4, 12),
                regions=["mPFC"],
            )

    def test_average_events_by_condition_calls_legacy_with_filtered_recordings(self):
        recordings = [make_recording("a.rec"), make_recording("b.rec")]
        collection = SimpleNamespace(recordings=recordings, brain_region_dict={"mPFC": 0})
        fake_ee = SimpleNamespace(average_events=Mock(return_value={"evt": ["avg"]}))
        fake_lfplt = SimpleNamespace()

        with patch(
            "coop_ephys_analysis.lfp.analysis._legacy_modules",
            return_value=(fake_ee, fake_lfplt),
        ):
            result = average_events_by_condition(
                collection,
                {"first": ["b.rec"]},
                "first",
                events=["evt"],
                mode="power",
                verbose=False,
            )

        self.assertEqual(result, {"evt": ["avg"]})
        called_collection = fake_ee.average_events.call_args.args[0]
        self.assertEqual([recording.name for recording in called_collection.recordings], ["b.rec"])
        self.assertEqual(called_collection.brain_region_dict, {"mPFC": 0})


if __name__ == "__main__":
    unittest.main()
