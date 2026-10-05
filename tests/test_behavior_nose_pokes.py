from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coop_ephys_analysis.behavior import find_filtered_nose_pokes  # noqa: E402


class FilteredNosePokeTests(unittest.TestCase):
    def test_retains_exact_gapped_and_overlapping_selfish_pokes(self):
        behaviors = {
            "selfish nose poke": np.array(
                [[1.0, 2.0], [10.0, 11.0], [20.0, 24.0], [30.0, 31.0]]
            ),
            "selfish light": np.array(
                [[2.0, 7.0], [11.5, 16.5], [22.0, 27.0], [33.0, 38.0]]
            ),
            "coop nose poke": np.array([[40.0, 41.0]]),
            "coop light": np.empty((0, 2)),
        }

        result = find_filtered_nose_pokes(behaviors, max_gap=1.0)

        np.testing.assert_allclose(
            result["filtered selfish nose pokes"],
            np.array([[1.0, 2.0], [10.0, 11.0], [20.0, 24.0]]),
        )
        self.assertEqual(result["filtered coop nose pokes"].shape, (0, 2))
        self.assertNotIn("filtered selfish nose pokes", behaviors)

    def test_corresponding_lights_do_not_cross_match(self):
        behaviors = {
            "selfish nose poke": np.empty((0, 2)),
            "selfish light": np.array([[2.0, 7.0]]),
            "coop nose poke": np.array([[1.0, 2.0]]),
            "coop light": np.empty((0, 2)),
        }

        result = find_filtered_nose_pokes(behaviors)

        self.assertEqual(result["filtered selfish nose pokes"].shape, (0, 2))
        self.assertEqual(result["filtered coop nose pokes"].shape, (0, 2))

    def test_uses_closest_poke_and_does_not_reuse_it(self):
        behaviors = {
            "selfish nose poke": np.array([[1.0, 2.0], [2.2, 2.8]]),
            "selfish light": np.array([[3.0, 8.0], [3.5, 8.5]]),
            "coop nose poke": np.array([[10.0, 14.0]]),
            "coop light": np.array([[12.0, 17.0], [13.0, 18.0]]),
        }

        result = find_filtered_nose_pokes(behaviors, max_gap=1.0)

        np.testing.assert_allclose(
            result["filtered selfish nose pokes"],
            np.array([[2.2, 2.8]]),
        )
        np.testing.assert_allclose(
            result["filtered coop nose pokes"],
            np.array([[10.0, 14.0]]),
        )

    def test_save_is_idempotent_and_does_not_replace_originals(self):
        original = np.array([[1.0, 2.0], [5.0, 6.0]])
        behaviors = {
            "selfish nose poke": original.copy(),
            "selfish light": np.array([[2.5, 7.5]]),
        }

        first = find_filtered_nose_pokes(behaviors, save=True)
        saved_first = behaviors["filtered selfish nose pokes"].copy()
        second = find_filtered_nose_pokes(behaviors, save=True)

        np.testing.assert_array_equal(behaviors["selfish nose poke"], original)
        np.testing.assert_allclose(saved_first, np.array([[1.0, 2.0]]))
        np.testing.assert_array_equal(
            behaviors["filtered selfish nose pokes"], saved_first
        )
        np.testing.assert_array_equal(
            first["filtered selfish nose pokes"],
            second["filtered selfish nose pokes"],
        )
        self.assertIsNot(
            first["filtered selfish nose pokes"],
            behaviors["filtered selfish nose pokes"],
        )

    def test_preserves_stage_day_and_recording_return_hierarchy(self):
        stage = [
            {
                "day1.rec": {
                    "selfish nose poke": np.array([[1.0, 2.0]]),
                    "selfish light": np.array([[2.0, 7.0]]),
                }
            },
            {
                "day2.rec": {
                    "coop nose poke": np.array([[3.0, 4.0]]),
                    "coop light": np.array([[4.5, 9.5]]),
                }
            },
        ]

        all_days = find_filtered_nose_pokes(stage)
        day_two = find_filtered_nose_pokes(stage, day=2)

        self.assertIsInstance(all_days, list)
        self.assertEqual(len(all_days), 2)
        self.assertIn("day1.rec", all_days[0])
        np.testing.assert_allclose(
            day_two["day2.rec"]["filtered coop nose pokes"],
            np.array([[3.0, 4.0]]),
        )

    def test_missing_invalid_and_malformed_events(self):
        missing = find_filtered_nose_pokes({"unrelated": np.array([[1.0, 2.0]])})
        self.assertEqual(missing["filtered selfish nose pokes"].shape, (0, 2))
        self.assertEqual(missing["filtered coop nose pokes"].shape, (0, 2))

        invalid_rows = find_filtered_nose_pokes(
            {
                "selfish nose poke": np.array(
                    [[1.0, 1.0], [np.nan, 2.0], [3.0, 4.0]]
                ),
                "selfish light": np.array([[4.5, 9.5]]),
            }
        )
        np.testing.assert_allclose(
            invalid_rows["filtered selfish nose pokes"],
            np.array([[3.0, 4.0]]),
        )

        with self.assertRaisesRegex(ValueError, "start, stop"):
            find_filtered_nose_pokes(
                {
                    "selfish nose poke": np.array([[1.0, 2.0, 3.0]]),
                    "selfish light": np.array([[3.0, 8.0]]),
                }
            )
        with self.assertRaisesRegex(ValueError, "max_gap"):
            find_filtered_nose_pokes({}, max_gap=-0.1)


if __name__ == "__main__":
    unittest.main()
