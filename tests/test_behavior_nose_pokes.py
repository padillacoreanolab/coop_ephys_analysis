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
    def test_matches_onsets_independent_of_valid_offsets(self):
        behaviors = {
            "selfish nose poke": np.array(
                [[1.0, 1.1], [10.0, 15.0], [20.0, 24.0], [30.0, 31.0]]
            ),
            "selfish light": np.array(
                [[2.0, 7.0], [11.0, 16.5], [22.0, 27.0], [31.5, 38.0]]
            ),
            "coop nose poke": np.array([[40.0, 41.0]]),
            "coop light": np.empty((0, 2)),
        }

        result = find_filtered_nose_pokes(behaviors)

        np.testing.assert_allclose(
            result["filtered selfish nose poke"],
            np.array([[1.0, 1.1], [10.0, 15.0]]),
        )
        self.assertEqual(result["filtered coop nose poke"].shape, (0, 2))
        self.assertNotIn("filtered selfish nose poke", behaviors)

    def test_corresponding_lights_do_not_cross_match(self):
        behaviors = {
            "selfish nose poke": np.empty((0, 2)),
            "selfish light": np.array([[2.0, 7.0]]),
            "coop nose poke": np.array([[1.0, 2.0]]),
            "coop light": np.empty((0, 2)),
        }

        result = find_filtered_nose_pokes(behaviors)

        self.assertEqual(result["filtered selfish nose poke"].shape, (0, 2))
        self.assertEqual(result["filtered coop nose poke"].shape, (0, 2))

    def match_selfish(self, pokes, lights, **kwargs):
        return find_filtered_nose_pokes(
            {"selfish nose poke": np.asarray(pokes),
             "selfish light": np.asarray(lights)}, **kwargs
        )["filtered selfish nose poke"]

    def test_default_inclusive_boundaries_and_outside_window(self):
        for delay, retained in [(0.9, True), (1.0, True), (1.1, True),
                                (0.8999, False), (1.1001, False), (-1.0, False)]:
            with self.subTest(delay=delay):
                result = self.match_selfish([[0.0, 0.1]], [[delay, delay + 2]])
                self.assertEqual(result.shape, (int(retained), 2))

    def test_custom_parameters_and_zero_tolerance(self):
        for delay in (1.75, 2.0, 2.25):
            result = self.match_selfish(
                [[0, 0.1]], [[delay, delay + 1]],
                expected_delay=2, tolerance=0.25,
            )
            np.testing.assert_array_equal(result, [[0, 0.1]])
        for delay, count in [(2, 1), (2.001, 0)]:
            result = self.match_selfish(
                [[0, 0.1]], [[delay, delay + 1]],
                expected_delay=2, tolerance=0,
            )
            self.assertEqual(result.shape, (count, 2))
        np.testing.assert_array_equal(
            self.match_selfish([[0, 1]], [[0, 2]], expected_delay=0, tolerance=0),
            [[0, 1]],
        )

    def test_closest_light_is_selected_without_global_reassignment(self):
        # Choosing the exact-delay light for the first poke leaves the second
        # unmatched, even though a different assignment could retain both.
        result = self.match_selfish(
            [[0, 0.01], [0.0625, 0.1]], [[0.9375, 2], [1, 2]],
            tolerance=0.0625,
        )
        np.testing.assert_array_equal(result, [[0, 0.01]])

    def test_chronological_priority_and_no_light_reuse(self):
        result = self.match_selfish(
            [[0.05, 0.1], [0, 0.02]], [[1.05, 2]],
        )
        np.testing.assert_array_equal(result, [[0, 0.02]])

    def test_equal_distance_prefers_earlier_light(self):
        # The later poke needs the later light, so both survive only if the
        # first poke breaks its equal-distance tie in favor of the earlier one.
        result = self.match_selfish(
            [[0, 0.1], [0.25, 0.3]], [[1.125, 2], [0.875, 2]],
            tolerance=0.125,
        )
        np.testing.assert_array_equal(result, [[0, 0.1], [0.25, 0.3]])

    def test_equal_onsets_preserve_poke_input_order(self):
        result = self.match_selfish([[0, 5], [0, 0.1]], [[1, 2]])
        np.testing.assert_array_equal(result, [[0, 5]])
        result = self.match_selfish([[0, 5], [0, 0.1]], [[1, 3], [1, 2]])
        np.testing.assert_array_equal(result, [[0, 5], [0, 0.1]])

    def test_one_poke_is_not_duplicated_for_multiple_lights(self):
        result = self.match_selfish([[0, 5]], [[1, 2], [1.05, 3]])
        np.testing.assert_array_equal(result, [[0, 5]])

    def test_unsorted_inputs_return_complete_chronological_intervals(self):
        result = self.match_selfish(
            [[10, 20], [0, 0.1], [5, 5.01]], [[6, 7], [11, 12], [1, 2]],
        )
        np.testing.assert_array_equal(result, [[0, 0.1], [5, 5.01], [10, 20]])

    def test_recordings_do_not_cross_match(self):
        result = find_filtered_nose_pokes({
            "a": {"coop nose poke": [[0, 0.1]]},
            "b": {"coop light": [[1, 2]]},
        })
        for recording in result.values():
            self.assertEqual(recording["filtered coop nose poke"].shape, (0, 2))

    def test_save_is_idempotent_and_does_not_replace_originals(self):
        original = np.array([[1.0, 2.0], [5.0, 6.0]])
        behaviors = {
            "selfish nose poke": original.copy(),
            "selfish light": np.array([[2.0, 7.5]]),
        }

        first = find_filtered_nose_pokes(behaviors, save=True)
        saved_first = behaviors["filtered selfish nose poke"].copy()
        second = find_filtered_nose_pokes(behaviors, save=True)

        np.testing.assert_array_equal(behaviors["selfish nose poke"], original)
        np.testing.assert_allclose(saved_first, np.array([[1.0, 2.0]]))
        np.testing.assert_array_equal(
            behaviors["filtered selfish nose poke"], saved_first
        )
        np.testing.assert_array_equal(
            first["filtered selfish nose poke"],
            second["filtered selfish nose poke"],
        )
        self.assertIsNot(
            first["filtered selfish nose poke"],
            behaviors["filtered selfish nose poke"],
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
                    "coop light": np.array([[4.0, 9.5]]),
                }
            },
        ]

        all_days = find_filtered_nose_pokes(stage)
        day_two = find_filtered_nose_pokes(stage, day=2)

        self.assertIsInstance(all_days, list)
        self.assertEqual(len(all_days), 2)
        self.assertIn("day1.rec", all_days[0])
        np.testing.assert_allclose(
            day_two["day2.rec"]["filtered coop nose poke"],
            np.array([[3.0, 4.0]]),
        )

    def test_missing_invalid_and_malformed_events(self):
        missing = find_filtered_nose_pokes({"unrelated": np.array([[1.0, 2.0]])})
        self.assertEqual(missing["filtered selfish nose poke"].shape, (0, 2))
        self.assertEqual(missing["filtered coop nose poke"].shape, (0, 2))

        invalid_rows = find_filtered_nose_pokes(
            {
                "selfish nose poke": np.array(
                    [[1.0, 1.0], [np.nan, 2.0], [3.0, 4.0]]
                ),
                "selfish light": np.array([[4.0, 9.5]]),
            }
        )
        np.testing.assert_allclose(
            invalid_rows["filtered selfish nose poke"],
            np.array([[3.0, 4.0]]),
        )

        with self.assertRaisesRegex(ValueError, "start, stop"):
            find_filtered_nose_pokes(
                {
                    "selfish nose poke": np.array([[1.0, 2.0, 3.0]]),
                    "selfish light": np.array([[3.0, 8.0]]),
                }
            )

    def test_day_adjustment_is_per_recording_for_both_choices(self):
        day = {}
        for number in (1, 6, 7, 8):
            # Both possible timings exist; only the appropriate poke survives.
            day[f"20250508_100203_Stage4_D{number}_1-2_merged.rec"] = {
                f"{kind} {event}": intervals
                for kind in ("selfish", "coop")
                for event, intervals in (
                    ("nose poke", [[0, 0.2], [10, 10.2]]),
                    ("light", [[0, 2], [11, 13]]),
                )
            }
        result = find_filtered_nose_pokes(day, recording_name="Stage4_D1_ignored")
        for number, (name, filtered) in zip((1, 6, 7, 8), result.items()):
            expected = [[0, 0.2]] if number <= 6 else [[10, 10.2]]
            for kind in ("selfish", "coop"):
                np.testing.assert_array_equal(filtered[f"filtered {kind} nose poke"], expected)
                self.assertNotIn(f"filtered {kind} nose poke", day[name])

    def test_named_single_recording_uses_custom_base_delay(self):
        for number, delay in ((1, 1.5), (6, 1.5), (7, 2.5), (8, 2.5)):
            with self.subTest(day=number):
                result = self.match_selfish(
                    [[0, 0.2]], [[delay, delay + 1]], expected_delay=2.5,
                    tolerance=0, recording_name=f"Stage4_D{number}_example.rec",
                )
                np.testing.assert_array_equal(result, [[0, 0.2]])

    def test_missing_unrecognized_and_other_days_keep_base_delay(self):
        for name in (None, "", "recording", "Stage4_D_merged", "Stage4_D1x_merged",
                     "Stage4_XD1_merged", "Stage4_d1_merged", "Stage4_D0_merged",
                     "Stage4_D9_merged", "Stage4_D10_merged", "Stage4_D60_merged"):
            with self.subTest(name=name):
                result = self.match_selfish(
                    [[0, 0.2]], [[1, 2]], recording_name=name,
                )
                np.testing.assert_array_equal(result, [[0, 0.2]])

    def test_day_token_at_name_boundaries(self):
        for name in ("D1", "D6.rec", "D1_example", "Stage4_D6"):
            with self.subTest(name=name):
                np.testing.assert_array_equal(
                    self.match_selfish([[0, 0.2]], [[0, 2]], recording_name=name),
                    [[0, 0.2]],
                )

    def test_adjusted_window_is_inclusive_and_allows_small_negative_delays(self):
        for delay, count in ((-0.1, 1), (0, 1), (0.1, 1), (-0.1001, 0), (0.1001, 0)):
            with self.subTest(delay=delay):
                result = self.match_selfish(
                    [[0, 0.2]], [[delay, delay + 1]], recording_name="Stage4_D1_test",
                )
                self.assertEqual(result.shape, (count, 2))

    def test_negative_effective_delay_identifies_recording_and_base(self):
        for number in (1, 6):
            name = f"Stage4_D{number}_test"
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name + ".*expected_delay=0.5"):
                    find_filtered_nose_pokes(
                        {name: {"selfish nose poke": [[0, 1]]}}, expected_delay=0.5,
                    )
        np.testing.assert_array_equal(
            self.match_selfish([[0, 0.2]], [[0.5, 2]], expected_delay=0.5,
                               recording_name="Stage4_D7_test"),
            [[0, 0.2]],
        )

    def test_selected_list_position_does_not_determine_recording_day(self):
        stage = [
            {"Stage4_D7_late": {"coop nose poke": [[0, 0.2]], "coop light": [[1, 2]]}},
            {"Stage4_D1_early": {"coop nose poke": [[0, 0.2]], "coop light": [[0, 2]]}},
        ]
        for selected, name in ((1, "Stage4_D7_late"), (2, "Stage4_D1_early")):
            result = find_filtered_nose_pokes(stage, day=selected, save=True)
            np.testing.assert_array_equal(result[name]["filtered coop nose poke"], [[0, 0.2]])
        first = find_filtered_nose_pokes(stage, save=True)
        second = find_filtered_nose_pokes(stage, save=True)
        for day_index, name in enumerate(("Stage4_D7_late", "Stage4_D1_early")):
            key = "filtered coop nose poke"
            np.testing.assert_array_equal(first[day_index][name][key], second[day_index][name][key])
            self.assertFalse(np.shares_memory(second[day_index][name][key], stage[day_index][name][key]))
            np.testing.assert_array_equal(stage[day_index][name]["coop nose poke"], [[0, 0.2]])

    def test_invalid_parameters_and_removed_max_gap(self):
        for name in ("expected_delay", "tolerance"):
            for value in (-0.1, np.nan, np.inf, -np.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        find_filtered_nose_pokes({}, **{name: value})
        with self.assertRaisesRegex(TypeError, "max_gap"):
            find_filtered_nose_pokes({}, max_gap=0.5)
        for day in (0, 2):
            with self.assertRaisesRegex(ValueError, "day"):
                find_filtered_nose_pokes([{}], day=day)



if __name__ == "__main__":
    unittest.main()
