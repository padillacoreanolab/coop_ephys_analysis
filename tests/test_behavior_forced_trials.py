from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coop_ephys_analysis.behavior import remove_forced_trials_nose_poke  # noqa: E402


SELFISH = "filtered selfish nose poke"
COOP = "filtered coop nose poke"


def intervals(n):
    starts = np.arange(n, dtype=float)
    return np.column_stack((starts, starts + 0.5))


def recording(selfish=25, coop=30):
    return {SELFISH: intervals(selfish), COOP: intervals(coop)}


class ForcedTrialNosePokeTests(unittest.TestCase):
    def test_default_count_updates_same_object_and_preserves_other_events(self):
        data = recording()
        for key in ("selfish nose poke", "coop nose poke", "coop light", "baseline iti 2s"):
            data[key] = intervals(4)
        originals = data.copy()
        result = remove_forced_trials_nose_poke(data)
        self.assertIs(result, data)
        np.testing.assert_array_equal(data[SELFISH], originals[SELFISH][20:])
        np.testing.assert_array_equal(data[COOP], originals[COOP][20:])
        for key in originals.keys() - {SELFISH, COOP}:
            self.assertIs(data[key], originals[key])
        self.assertFalse(np.shares_memory(data[SELFISH], originals[SELFISH]))

    def test_day_and_stage_trim_each_recording_and_type_independently(self):
        for as_stage in (False, True):
            with self.subTest(as_stage=as_stage):
                day = {"a": recording(5, 2), "b": recording(2, 6)}
                data = [day, {"c": recording(4, 4)}] if as_stage else day
                self.assertIs(remove_forced_trials_nose_poke(data, n_remove=3), data)
                for name, expected in (("a", (2, 0)), ("b", (0, 3))):
                    self.assertEqual(tuple(len(day[name][key]) for key in (SELFISH, COOP)), expected)
                if as_stage:
                    self.assertEqual(len(data[1]["c"][COOP]), 1)

    def test_sorts_by_onset_stably_and_keeps_complete_intervals(self):
        data = {SELFISH: np.array([[3, 10], [1, 8], [1, 2], [2, 6]]), COOP: intervals(0)}
        remove_forced_trials_nose_poke(data, n_remove=1)
        np.testing.assert_array_equal(data[SELFISH], [[1, 2], [2, 6], [3, 10]])

    def test_zero_does_not_reorder_or_replace_arrays(self):
        data = {SELFISH: np.array([[3, 4], [1, 2]]), COOP: intervals(2)}
        originals = data.copy()
        remove_forced_trials_nose_poke(data, n_remove=0)
        for key in (SELFISH, COOP):
            self.assertIs(data[key], originals[key])
        np.testing.assert_array_equal(data[SELFISH], [[3, 4], [1, 2]])

    def test_short_exact_and_empty_arrays_become_empty_pairs(self):
        for size in (0, 1, 19, 20):
            with self.subTest(size=size):
                data = recording(size, size)
                remove_forced_trials_nose_poke(data)
                self.assertEqual(data[SELFISH].shape, (0, 2))
                self.assertEqual(data[COOP].shape, (0, 2))
        data = {SELFISH: [], COOP: [[1, 2], [3, 4]]}
        remove_forced_trials_nose_poke(data, n_remove=np.int64(1))
        self.assertEqual(data[SELFISH].shape, (0, 2))
        np.testing.assert_array_equal(data[COOP], [[3, 4]])

    def test_repeated_calls_remove_additional_batches(self):
        data = recording(6, 7)
        remove_forced_trials_nose_poke(data, n_remove=2)
        remove_forced_trials_nose_poke(data, n_remove=2)
        np.testing.assert_array_equal(data[SELFISH], intervals(6)[4:])
        np.testing.assert_array_equal(data[COOP], intervals(7)[4:])

    def test_invalid_counts_leave_arrays_unchanged(self):
        for n_remove in (-1, 1.5, 2.0, True, np.bool_(False), "2", None, np.nan):
            with self.subTest(n_remove=n_remove):
                data = recording()
                original = data[SELFISH]
                with self.assertRaisesRegex(ValueError, "n_remove"):
                    remove_forced_trials_nose_poke(data, n_remove=n_remove)
                self.assertIs(data[SELFISH], original)

    def test_missing_key_in_later_recording_does_not_partially_mutate(self):
        data = [{"good": recording()}, {"missing": {SELFISH: intervals(25)}}]
        first_original = data[0]["good"][SELFISH]
        last_original = data[1]["missing"][SELFISH]
        with self.assertRaisesRegex(KeyError, "missing.*filtered coop nose poke"):
            remove_forced_trials_nose_poke(data)
        self.assertIs(data[0]["good"][SELFISH], first_original)
        self.assertIs(data[1]["missing"][SELFISH], last_original)

    def test_invalid_intervals_do_not_partially_mutate_or_drop_rows(self):
        for invalid in ([[1, 2, 3]], [1, 2], [[np.nan, 2]], [[1, 1]], [[2, 1]],
                        [[1, np.inf]], [["bad", 2]], None, np.empty((0, 3))):
            with self.subTest(invalid=invalid):
                data = {"good": recording(), "bad": {SELFISH: intervals(25), COOP: invalid}}
                original = data["good"][SELFISH]
                with self.assertRaisesRegex(ValueError, "bad.*filtered coop nose poke"):
                    remove_forced_trials_nose_poke(data)
                self.assertIs(data["good"][SELFISH], original)

    def test_invalid_hierarchy(self):
        for data in (None, 3, [recording(), None], {"recording": {"nested": {}}}):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    remove_forced_trials_nose_poke(data)


if __name__ == "__main__":
    unittest.main()
