from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coop_ephys_analysis.behavior import (  # noqa: E402
    find_iti_windows,
    plot_recordings_behaviors,
)


class BehaviorItiTests(unittest.TestCase):
    def setUp(self):
        self.show_patch = patch("matplotlib.pyplot.show")
        self.show_patch.start()

    def tearDown(self):
        self.show_patch.stop()
        plt.close("all")

    def test_save_adds_buffered_windows_to_recording(self):
        behaviors = {
            "selfish nose poke": np.array([[4.0, 5.0]]),
            "coop light": np.array([[10.0, 11.0]]),
        }

        count = find_iti_windows(
            behaviors,
            window_duration=2,
            time_bounds=(0, 15),
            buffer=0.5,
            save=True,
        )

        expected = np.array(
            [
                [0.0, 2.0],
                [5.5, 7.5],
                [7.5, 9.5],
                [11.5, 13.5],
            ]
        )
        self.assertEqual(count, 4)
        np.testing.assert_allclose(behaviors["baseline iti 2s"], expected)

    def test_save_updates_every_recording_and_is_stable_on_rerun(self):
        day = {
            "a": {"event": np.array([[4.0, 5.0]])},
            "b": {"event": np.array([[7.0, 8.0]])},
        }

        with redirect_stdout(StringIO()):
            find_iti_windows(
                day,
                window_duration=1.5,
                time_bounds=(0, 10),
                buffer=0.5,
                save=True,
            )
        first_windows = {
            name: values["baseline iti 1.5s"].copy()
            for name, values in day.items()
        }

        with redirect_stdout(StringIO()):
            find_iti_windows(
                day,
                window_duration=1.5,
                time_bounds=(0, 10),
                buffer=0.5,
                save=True,
            )

        for name, values in day.items():
            np.testing.assert_allclose(values["baseline iti 1.5s"], first_windows[name])

    def test_no_events_uses_explicit_recording_bounds(self):
        behaviors = {"event": np.empty((0, 2))}

        count = find_iti_windows(
            behaviors,
            window_duration=2,
            time_bounds=(0, 5),
            buffer=0.5,
            save=True,
        )

        self.assertEqual(count, 2)
        np.testing.assert_allclose(
            behaviors["baseline iti 2s"],
            np.array([[0.0, 2.0], [2.0, 4.0]]),
        )

    def test_invalid_window_or_buffer_raises(self):
        behaviors = {"event": np.array([[1.0, 2.0]])}

        with self.assertRaises(ValueError):
            find_iti_windows(behaviors, window_duration=0)
        with self.assertRaises(ValueError):
            find_iti_windows(behaviors, buffer=-0.1)

    def test_default_plot_appends_gray_baseline_row(self):
        stage = [
            {
                "recording": {
                    "selfish light": np.array([[1.0, 2.0]]),
                    "baseline iti 2s": np.array([[4.0, 6.0]]),
                }
            }
        ]

        plot_recordings_behaviors(stage)

        ax = plt.gcf().axes[0]
        labels = [label.get_text() for label in ax.get_yticklabels()]
        self.assertEqual(labels[-1], "baseline iti 2s")
        baseline_bar = ax.patches[-1]
        self.assertAlmostEqual(baseline_bar.get_x(), 4.0)
        self.assertAlmostEqual(baseline_bar.get_width(), 2.0)
        self.assertEqual(
            baseline_bar.get_facecolor(),
            mcolors.to_rgba("#B0B0B0"),
        )

    def test_default_plot_orders_multiple_baselines_by_duration(self):
        stage = [
            {
                "recording": {
                    "baseline iti 10s": np.array([[20.0, 30.0]]),
                    "baseline iti 1.5s": np.array([[1.0, 2.5]]),
                    "baseline iti 2s": np.array([[5.0, 7.0]]),
                }
            }
        ]

        plot_recordings_behaviors(stage)

        labels = [label.get_text() for label in plt.gcf().axes[0].get_yticklabels()]
        self.assertEqual(
            labels[-3:],
            ["baseline iti 1.5s", "baseline iti 2s", "baseline iti 10s"],
        )

    def test_plot_without_baseline_keeps_standard_rows(self):
        stage = [{"recording": {"selfish light": np.array([[1.0, 2.0]])}}]

        plot_recordings_behaviors(stage)

        labels = [label.get_text() for label in plt.gcf().axes[0].get_yticklabels()]
        self.assertEqual(
            labels,
            [
                "selfish light",
                "selfish nose poke",
                "subject port entry",
                "coop light",
                "coop nose poke",
                "recipient port entry",
            ],
        )

    def test_none_includes_all_events_with_baseline_last(self):
        stage = [
            {
                "recording": {
                    "custom event": np.array([[1.0, 2.0]]),
                    "baseline iti 2s": np.array([[4.0, 6.0]]),
                }
            }
        ]

        plot_recordings_behaviors(stage, behavior_order=None)

        labels = [label.get_text() for label in plt.gcf().axes[0].get_yticklabels()]
        self.assertEqual(labels, ["custom event", "baseline iti 2s"])

    def test_custom_order_remains_authoritative(self):
        stage = [
            {
                "recording": {
                    "selfish light": np.array([[1.0, 2.0]]),
                    "baseline iti 2s": np.array([[4.0, 6.0]]),
                }
            }
        ]

        plot_recordings_behaviors(stage, behavior_order=["selfish light"])

        labels = [label.get_text() for label in plt.gcf().axes[0].get_yticklabels()]
        self.assertEqual(labels, ["selfish light"])


if __name__ == "__main__":
    unittest.main()
