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
    plot_event_average_spectrogram,
    plot_recording_event_spectrogram,
    plot_recording_spectral_traces,
    select_recordings_by_condition,
    summarize_event_windows,
    update_collection_events_from_behavior,
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

    def test_update_collection_events_merges_converts_and_preserves_source(self):
        recording = make_recording(
            "a.rec",
            events={
                "lfp only": np.array([[500.0, 750.0]]),
                "selfish light": np.array([[999.0, 1000.0]]),
            },
        )
        collection = SimpleNamespace(
            recordings=[recording],
            recording_to_event_dict={
                "a.rec": {"lfp only": np.array([[500.0, 750.0]])}
            },
        )
        source = {
            "a": {
                "selfish light": np.array([[1.0, 2.0]]),
                "baseline iti 2s": np.array([[3.0, 5.0]]),
                "filtered selfish nose pokes": np.array([[6.0, 7.0]]),
            },
            "unused": {"event": np.array([[1.0, 2.0]])},
        }
        source_copy = {
            name: {event: values.copy() for event, values in events.items()}
            for name, events in source.items()
        }

        summary = update_collection_events_from_behavior(
            collection,
            source,
            source_unit="seconds",
            strict=False,
        )

        np.testing.assert_allclose(
            recording.event_dict["selfish light"],
            np.array([[1000.0, 2000.0]]),
        )
        np.testing.assert_allclose(
            recording.event_dict["baseline iti 2s"],
            np.array([[3000.0, 5000.0]]),
        )
        np.testing.assert_allclose(
            recording.event_dict["filtered selfish nose pokes"],
            np.array([[6000.0, 7000.0]]),
        )
        np.testing.assert_allclose(
            recording.event_dict["lfp only"],
            np.array([[500.0, 750.0]]),
        )
        np.testing.assert_allclose(
            collection.recording_to_event_dict["a.rec"]["baseline iti 2s"],
            np.array([[3000.0, 5000.0]]),
        )
        np.testing.assert_allclose(
            collection.recording_to_event_dict["a.rec"][
                "filtered selfish nose pokes"
            ],
            np.array([[6000.0, 7000.0]]),
        )
        self.assertEqual(summary["updated_recordings"], ["a.rec"])
        self.assertEqual(summary["unused_behavior_recordings"], ["unused"])
        for name, events in source.items():
            for event, values in events.items():
                np.testing.assert_array_equal(values, source_copy[name][event])

    def test_update_collection_events_strict_missing_is_transactional(self):
        first = make_recording("a.rec", events={"old": np.array([[1.0, 2.0]])})
        second = make_recording("b.rec", events={"old": np.array([[3.0, 4.0]])})
        collection = SimpleNamespace(
            recordings=[first, second],
            recording_to_event_dict={},
        )

        with self.assertRaisesRegex(ValueError, "missing collection recording"):
            update_collection_events_from_behavior(
                collection,
                {"a": {"new": np.array([[5.0, 6.0]])}},
                strict=True,
            )

        self.assertNotIn("new", first.event_dict)
        self.assertEqual(collection.recording_to_event_dict, {})

    def test_update_collection_events_rejects_duplicate_normalized_names(self):
        collection = SimpleNamespace(recordings=[make_recording("a.rec")])

        with self.assertRaisesRegex(ValueError, "Duplicate behavior recording"):
            update_collection_events_from_behavior(
                collection,
                {
                    "a": {"event": np.array([[1.0, 2.0]])},
                    "a.rec": {"event": np.array([[1.0, 2.0]])},
                },
            )

    def test_update_collection_events_validates_units_and_event_shapes(self):
        collection = SimpleNamespace(recordings=[make_recording("a.rec")])

        with self.assertRaisesRegex(ValueError, "source_unit"):
            update_collection_events_from_behavior(
                collection,
                {"a": {"event": np.array([[1.0, 2.0]])}},
                source_unit="minutes",
            )

        with self.assertRaisesRegex(ValueError, "shape"):
            update_collection_events_from_behavior(
                collection,
                {"a": {"event": np.array([[1.0, 2.0, 3.0]])}},
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

    def test_event_average_spectrogram_uses_equal_recording_weights_and_windows(self):
        first = make_recording(
            "a.rec",
            events={"evt": np.array([[500, 1500], [2000, 3000]], dtype=float)},
        )
        second = make_recording(
            "b.rec",
            events={"evt": np.array([[500, 1500]], dtype=float)},
        )
        first.power.fill(1.0)
        second.power.fill(3.0)
        collection = SimpleNamespace(recordings=[first, second])

        result = plot_event_average_spectrogram(
            collection,
            event="evt",
            mode="power",
            event_len=1,
            pre_window=0.5,
            post_window=0.5,
            regions=["mPFC"],
            plot=True,
        )

        np.testing.assert_allclose(result["average"], 2.0)
        np.testing.assert_allclose(result["relative_time"], [-0.5, 0.0, 0.5, 1.0])
        self.assertEqual(result["event_counts"], {"a.rec": 2, "b.rec": 1})
        self.assertEqual(result["recording_names"], ["a.rec", "b.rec"])
        self.assertEqual(len(result["axes"]), 1)

    def test_event_average_spectrogram_filters_collection_by_condition(self):
        first = make_recording(
            "a.rec",
            events={"evt": np.array([[0, 1000]], dtype=float)},
        )
        second = make_recording(
            "b.rec",
            events={"evt": np.array([[0, 1000]], dtype=float)},
        )
        first.power.fill(1.0)
        second.power.fill(4.0)
        collection = SimpleNamespace(recordings=[first, second])

        result = plot_event_average_spectrogram(
            collection,
            event="evt",
            condition_recordings={"selected": ["b.rec"]},
            selected_condition="selected",
            event_len=1,
            plot=False,
        )

        np.testing.assert_allclose(result["average"], 4.0)
        self.assertEqual(result["recording_names"], ["b.rec"])
        self.assertEqual(result["axes"], [])

    def test_event_average_spectrogram_supports_connectivity_modes(self):
        recording = make_recording(
            "a.rec",
            events={"evt": np.array([[0, 1000]], dtype=float)},
        )
        collection = SimpleNamespace(recordings=[recording])

        for mode in ("coherence", "granger"):
            result = plot_event_average_spectrogram(
                collection,
                event="evt",
                mode=mode,
                event_len=1,
                pairs=[("mPFC", "BLA")],
            )

            self.assertEqual(result["average"].shape, (2, 4, 2, 2))
            self.assertEqual(len(result["axes"]), 1)

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
