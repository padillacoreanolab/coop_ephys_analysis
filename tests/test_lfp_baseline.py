"""Numerical, timing, export, and plot regression checks for trial baselines."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coop_ephys_analysis.lfp import (
    calculate_event_baseline_power,
    export_event_baseline_csv,
    export_event_baseline_trialwise_csv,
    load_event_baseline_collection,
    plot_baselined_event_average_spectrogram,
    plot_baselined_event_average_traces,
)
from coop_ephys_analysis.lfp.baseline import _window_power


def recording(name="a", events=None):
    rng = np.random.default_rng(123)
    return SimpleNamespace(
        name=name, subject=name, condition="social", resample_rate=100,
        timewindow=1., timestep=.5, halfbandwidth=2,
        brain_region_dict={"mPFC": 0, "BLA": 1},
        rms_traces=rng.normal(size=(3000, 2)),
        event_dict={"poke": [[8000.3, 8100.3], [18000.3, 18100.3]]} if events is None else events,
    )


def fake_power(rec, starts, n_samples, indices, tapers):
    # Within each trial: power grows linearly with time. Trial 2 has a
    # different offset and gain; separate normalization should remove both.
    seconds = starts / rec.resample_rate
    gain = np.where(seconds < 12, 1., 10.)
    values = gain * (seconds + 2.)
    freq = np.arange(50, dtype=float)
    powers = np.broadcast_to(values[:, None, None], (len(starts), 50, len(indices))).copy()
    bounds = (starts >= 0) & (starts + n_samples <= rec.rms_traces.shape[0])
    powers[~bounds] = np.nan
    return powers, freq, bounds


class EventBaselineTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def calc(self, rec=None, **kwargs):
        kwargs.setdefault("freq_range", (4, 8))
        return calculate_event_baseline_power(
            SimpleNamespace(recordings=[rec or recording()]), "poke", **kwargs)

    def test_rms_only_loader_and_processed_event_import(self):
        import h5py
        from coop_ephys_analysis.lfp import update_collection_events_from_behavior

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "recordings").mkdir()
            path = root / "lfp_collection.json"
            path.write_text(json.dumps({"metadata": {"number of recordings": 1}}))
            with h5py.File(root / "recordings" / "a.h5", "w") as handle:
                attrs = handle.create_group("metadata").attrs
                attrs.update({"name": "a.rec", "subject": "a", "resample_rate": 100,
                              "window duration": 1., "window step": .5,
                              "half bandwidth product": 2, "excluded_regions": ["BLA"]})
                handle.create_group("brain region dict").attrs.update({"mPFC": "0", "BLA": "1"})
                handle.create_group("data").create_dataset("rms_traces", data=np.ones((3000, 2)))
            collection = load_event_baseline_collection(path, recording_names=["a.rec"])
            rec = collection.recordings[0]
            self.assertEqual(list(rec.brain_region_dict), ["mPFC", "BLA"])
            self.assertEqual(rec.event_dict, {})
            self.assertTrue(np.isnan(rec.rms_traces[:, 1]).all())
            self.assertFalse(hasattr(rec, "power"))
            update_collection_events_from_behavior(collection, {"a": {"poke": [[8, 8.1]]}})
            np.testing.assert_allclose(rec.event_dict["poke"], [[8000, 8100]])
            with self.assertRaises(ValueError):
                load_event_baseline_collection(path, recording_names=["a.rec", "missing.rec"])

    @patch("coop_ephys_analysis.lfp.baseline._window_power", side_effect=fake_power)
    def test_each_trial_has_own_baseline_and_normalizes_before_averaging(self, _):
        result = self.calc()
        stats = result["tables"]["baseline_statistics"]
        first = stats[stats.event_index == 0]
        second = stats[stats.event_index == 1]
        np.testing.assert_allclose(first.baseline_mean, 7.5)
        np.testing.assert_allclose(first.baseline_sd, .5)
        np.testing.assert_allclose(second.baseline_mean, 175.)
        np.testing.assert_allclose(second.baseline_sd, 5.)
        points = result["tables"]["trial_power"]
        np.testing.assert_allclose(points.loc[points.relative_time_s == 0, "zscore"], 4.)
        np.testing.assert_allclose(result["tables"]["group_traces"].query("relative_time_s == 0").mean_z, 4.)
        self.assertTrue(result["tables"]["group_traces"].sem_z.isna().all())

    @patch("coop_ephys_analysis.lfp.baseline._window_power", side_effect=fake_power)
    def test_exact_baseline_support_and_subsample_rounding_retained(self, _):
        result = self.calc()
        samples = result["tables"]["baseline_samples"].query("event_index == 0 and frequency_hz == 4 and region == 'mPFC'")
        np.testing.assert_allclose(samples.relative_time_s, [-2.5, -2, -1.5])
        np.testing.assert_allclose(samples.window_start_relative_s, np.array([-3, -2.5, -2]) - .0003)
        np.testing.assert_allclose(samples.window_stop_relative_s, np.array([-2, -1.5, -1]) - .0003)
        np.testing.assert_allclose(result["tables"]["event_qc"].onset_rounding_error_s, -.0003)

    def test_real_multitaper_matches_independent_single_window_estimates(self):
        from spectral_connectivity import Multitaper, Connectivity
        rec = recording()
        transform = Multitaper(np.zeros((100, 1)), sampling_frequency=100,
                               time_window_duration=1, time_halfbandwidth_product=2)
        values, freqs, bounds = _window_power(rec, np.array([500, 550, 600]), 100, [0, 1], transform.tapers)
        self.assertTrue(bounds.all())
        for i, start in enumerate([500, 550, 600]):
            direct = Connectivity.from_multitaper(Multitaper(
                rec.rms_traces[start:start+100], sampling_frequency=100,
                time_window_duration=1, time_halfbandwidth_product=2))
            np.testing.assert_allclose(values[i], direct.power()[0])
            np.testing.assert_array_equal(freqs, direct.frequencies)

    def test_real_zscore_matches_exported_baseline_samples_without_mutation(self):
        rec = recording()
        original = rec.rms_traces.copy()
        result = self.calc(rec)
        tables = result["tables"]
        for _, stats in tables["baseline_statistics"].iterrows():
            base = tables["baseline_samples"]
            base = base[(base.trial_id == stats.trial_id) & (base.region == stats.region) & (base.frequency_hz == stats.frequency_hz)]
            self.assertAlmostEqual(stats.baseline_mean, base.power.mean())
            self.assertAlmostEqual(stats.baseline_sd, base.power.std(ddof=1))
        points = tables["trial_power"].merge(tables["baseline_statistics"], on=["trial_id", "region", "frequency_hz"])
        np.testing.assert_allclose(points.zscore, (points.power - points.baseline_mean) / points.baseline_sd)
        np.testing.assert_array_equal(original, rec.rms_traces)

    def test_missing_and_zero_sd_baselines_not_silently_filled(self):
        rec = recording()
        rec.rms_traces[:, 0] = 0  # All-zero power -> zero baseline SD.
        rec.rms_traces[500, 1] = np.nan  # Corrupt first trial's baseline only.
        result = self.calc(rec)
        stats = result["tables"]["baseline_statistics"]
        self.assertTrue((stats[stats.region == "mPFC"].status == "invalid_baseline_sd").all())
        self.assertTrue((stats.query("region == 'BLA' and event_index == 0").status == "missing_baseline_power").all())
        self.assertTrue((stats.query("region == 'BLA' and event_index == 1").status == "ok").all())
        points = result["tables"]["trial_power"]
        self.assertTrue(points[points.region == "mPFC"].zscore.isna().all())
        self.assertEqual(result["event_counts"]["a"], 1)

    def test_boundary_and_invalid_events_stay_in_qc(self):
        rec = recording(events={"poke": [[1000, 1100], [8000, 8100], [29000, 29100], [5000, 4000]],
                                "light": [[5200, 5700]]})
        result = self.calc(rec)
        qc = result["tables"]["event_qc"]
        self.assertEqual(len(qc), 4)
        self.assertEqual(qc.iloc[3].status, "invalid_event_interval")
        self.assertIn("light", json.loads(qc.iloc[1].overlapping_events))
        stats = result["tables"]["baseline_statistics"]
        self.assertTrue((stats[stats.event_index == 2].status == "event_window_out_of_bounds").all())
        self.assertEqual(result["event_counts"]["a"], 1)

    @patch("coop_ephys_analysis.lfp.baseline._window_power", side_effect=fake_power)
    def test_recording_weights_counts_and_sem(self, _):
        a, b = recording("a"), recording("b", {"poke": [[8000, 8100]]})
        # Alter post-onset power only for b; records should have equal weights
        # despite a having twice as many trials.
        def altered(rec, starts, *args):
            power, freqs, bounds = fake_power(rec, starts, *args)
            if rec.name == "b":
                power[starts >= 750] += 2
            return power, freqs, bounds
        with patch("coop_ephys_analysis.lfp.baseline._window_power", side_effect=altered):
            res = calculate_event_baseline_power(SimpleNamespace(recordings=[a, b]), "poke", freq_range=(4, 8))
        frame = res["tables"]["group_traces"].query("relative_time_s == 0")
        np.testing.assert_allclose(frame.mean_z, 6)
        np.testing.assert_allclose(frame.sem_z, 2)
        np.testing.assert_array_equal(frame.n_recordings, 2)

    def test_reuse_plots_and_csv_exact_values(self):
        result = self.calc()
        with patch("coop_ephys_analysis.lfp.baseline._window_power", side_effect=AssertionError("recomputed")):
            heatmap = plot_baselined_event_average_spectrogram(result, color_limit=5)
            trace = plot_baselined_event_average_traces(result, comparison_result=result, labels=["A", "B"])
        self.assertEqual(len(heatmap["axes"]), 2)
        for region, ax in zip(result["regions"], trace["axes"]):
            table = result["tables"]["group_traces"].query("region == @region").sort_values("relative_time_s")
            np.testing.assert_allclose(ax.lines[0].get_ydata(), table.mean_z)
            matrix = result["tables"]["group_spectrograms"].query("region == @region").pivot(
                index="frequency_hz", columns="relative_time_s", values="mean_z")
            index = result["regions"].index(region)
            np.testing.assert_allclose(np.asarray(heatmap["axes"][index].collections[0].get_array()).ravel(), matrix.to_numpy().ravel())
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_event_baseline_csv(trace, tmp)
            for name, original in result["tables"].items():
                loaded = pd.read_csv(paths[name])
                self.assertEqual(len(loaded), 2 * len(original))
                for column in original.select_dtypes(include="number"):
                    np.testing.assert_allclose(loaded.loc[loaded.comparison_line == 0, column], original[column], equal_nan=True)
            with self.assertRaises(FileExistsError):
                export_event_baseline_csv(result, tmp)
            self.assertEqual(len(json.loads(paths["analysis_settings"].read_text())), 2)

    def test_trialwise_csv_preserves_points_missing_rows_and_metadata(self):
        rec = recording()
        rec.source_h5 = str(Path("a.rec.h5").resolve())
        result = self.calc(rec)
        result["settings"]["inputs"] = {"day": 8}
        points = result["tables"]["trial_power"]
        first_trial = points.trial_id.iloc[0]
        points.loc[(points.trial_id == first_trial) & (points.region == "BLA"), "zscore"] = np.nan
        points.loc[points.index[0], "zscore"] = 0.
        with tempfile.TemporaryDirectory() as tmp:
            path = export_event_baseline_trialwise_csv(
                {"selfish_social": result}, Path(tmp) / "trialwise.csv",
                recording_metadata={"a": {"role": "subject", "paired_subject_id": "b"}},
            )
            loaded = pd.read_csv(path, float_precision="round_trip")
            self.assertEqual(len(list(Path(tmp).iterdir())), 1)
            self.assertEqual(loaded.columns[:12].tolist(), [
                "day", "animal_id", "role", "paired_subject_id", "h5_file", "rec_file",
                "choice", "trial_number", "alignment_time_s",
                "relative_time_s", "metric", "feature",
            ])
            self.assertEqual(set(loaded.day), {8})
            self.assertEqual(set(loaded.role), {"subject"})
            self.assertEqual(set(loaded.paired_subject_id), {"b"})
            self.assertEqual(set(loaded.h5_file), {"a.rec.h5"})
            self.assertEqual(set(loaded.trial_number), {1, 2})
            self.assertEqual(set(loaded.normalization), {"zscore"})
            self.assertNotIn("alignment_source", loaded)
            self.assertTrue({"source_region", "target_region", "direction"}.isdisjoint(loaded.columns))
            np.testing.assert_array_equal(loaded.baseline_start_relative_s, -3)
            np.testing.assert_array_equal(loaded.baseline_stop_relative_s, -1)
            freq_cols = [c for c in loaded if c.startswith("freq_")]
            self.assertEqual(freq_cols, [f"freq_{i:03d}_{f}_Hz" for i, f in enumerate(range(4, 9), 1)])
            expected = points.pivot(index=["trial_id", "relative_time_s", "region"], columns="frequency_hz", values="zscore")
            actual = loaded.rename(columns={"feature": "region"}).set_index(["trial_id", "relative_time_s", "region"])[freq_cols]
            self.assertEqual(len(actual), len(expected))
            np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
            self.assertTrue(loaded.loc[(loaded.trial_id == first_trial) & (loaded.feature == "BLA"), freq_cols].isna().all().all())
            self.assertEqual((actual.to_numpy() == 0).sum(), 1)
            with self.assertRaises(FileExistsError):
                export_event_baseline_trialwise_csv(result, path)
            with self.assertRaises(ValueError):
                export_event_baseline_trialwise_csv(result, Path(tmp) / "bad.csv", day=9)

    def test_trialwise_csv_reconstructs_plots_with_missing_data_and_unequal_trials(self):
        a, b = recording("a"), recording("b", events={"poke": [[8000.3, 8100.3]]})
        with patch("coop_ephys_analysis.lfp.baseline._window_power", side_effect=fake_power):
            result = calculate_event_baseline_power(SimpleNamespace(recordings=[a, b]), "poke", freq_range=(4, 8))
        points = result["tables"]["trial_power"]
        points.loc[(points.recording == "a") & (points.event_index == 0) & (points.frequency_hz == 4), "zscore"] = np.nan
        points.loc[(points.recording == "b") & (points.frequency_hz == 8), "zscore"] += 5
        from coop_ephys_analysis.lfp.baseline import _summary_tables
        result["tables"].update(_summary_tables(points))
        with tempfile.TemporaryDirectory() as tmp:
            path = export_event_baseline_trialwise_csv({"first": result, "second": result}, Path(tmp) / "points.csv")
            loaded = pd.read_csv(path, float_precision="round_trip")
            self.assertEqual(set(loaded.selection), {"first", "second"})
            freq_cols = [c for c in loaded if c.startswith("freq_")]
            rec = loaded.query("selection == 'first'").groupby(["rec_file", "feature", "relative_time_s"])[freq_cols].mean()
            heatmap = rec.groupby(["feature", "relative_time_s"]).mean()
            expected_heatmap = result["tables"]["group_spectrograms"].pivot(index=["region", "relative_time_s"], columns="frequency_hz", values="mean_z")
            np.testing.assert_allclose(heatmap, expected_heatmap)
            lines = rec.mean(axis=1).groupby(["feature", "relative_time_s"]).agg(["mean", "sem"])
            expected_lines = result["tables"]["group_traces"].set_index(["region", "relative_time_s"]).sort_index()
            np.testing.assert_allclose(lines["mean"], expected_lines.mean_z)
            np.testing.assert_allclose(lines["sem"], expected_lines.sem_z)
            trace = plot_baselined_event_average_traces(result, comparison_result=result,
                                                      labels=("First", "Second"), plot=False)
            trace_path = export_event_baseline_trialwise_csv(trace, Path(tmp) / "comparison.csv")
            self.assertEqual(len(pd.read_csv(trace_path)), len(loaded))
            paths = export_event_baseline_trialwise_csv(
                {"first": result, "second": result}, Path(tmp) / "recordings", split_by_recording=True,
            )
            self.assertEqual(set(paths), {"a", "b"})
            for recording_name, csv in paths.items():
                frame = pd.read_csv(csv, float_precision="round_trip")
                self.assertEqual(set(frame.rec_file), {recording_name})
                self.assertEqual(set(frame.selection), {"first", "second"})
                # CSV type inference may choose int for a complete subset of
                # a float column that had missing values in another recording.
                pd.testing.assert_frame_equal(
                    frame, loaded[loaded.rec_file == recording_name].reset_index(drop=True),
                    check_dtype=False, check_exact=True,
                )
            # Existing files must be detected before writing any missing ones.
            paths["a"].unlink()
            with self.assertRaises(FileExistsError):
                export_event_baseline_trialwise_csv(result, Path(tmp) / "recordings", split_by_recording=True)
            self.assertFalse(paths["a"].exists())
            optional = export_event_baseline_trialwise_csv(
                result, Path(tmp) / "with_source.csv", include_alignment_source=True,
            )
            self.assertEqual(set(pd.read_csv(optional).alignment_source), {"event_dict_onset"})

    def test_trialwise_csv_sorts_trial_numbers_numerically(self):
        rec = recording(events={"poke": [[8000 + i * 500, 8100 + i * 500] for i in range(12)]})
        with patch("coop_ephys_analysis.lfp.baseline._window_power", side_effect=fake_power):
            result = self.calc(rec)
        with tempfile.TemporaryDirectory() as tmp:
            path = export_event_baseline_trialwise_csv(result, Path(tmp) / "ordered.csv")
            loaded = pd.read_csv(path)
            trials = loaded.drop_duplicates("trial_id")
            self.assertEqual(trials.trial_number.tolist(), list(range(1, 13)))
            self.assertEqual([json.loads(t)[-1] for t in trials.trial_id], list(range(12)))
            for _, trial in loaded.groupby("trial_id", sort=False):
                self.assertTrue(trial.relative_time_s.is_monotonic_increasing)

    def test_trialwise_pairing_map_and_recording_override(self):
        a, b, alone = recording("a"), recording("b"), recording("alone")
        a.condition = "subject with recipient"
        b.condition = "recipient"
        alone.condition = "subject alone"
        with patch("coop_ephys_analysis.lfp.baseline._window_power", side_effect=fake_power):
            result = calculate_event_baseline_power(
                SimpleNamespace(recordings=[a, b, alone]), "poke", freq_range=(4, 8))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            export_event_baseline_trialwise_csv(result, path, pairing_dict={"a": "b", "alone": "other"})
            loaded = pd.read_csv(path)
            self.assertEqual(set(loaded.loc[loaded.animal_id == "a", "paired_subject_id"]), {"b"})
            self.assertEqual(set(loaded.loc[loaded.animal_id == "b", "paired_subject_id"]), {"a"})
            self.assertTrue(loaded.loc[loaded.animal_id == "alone", "paired_subject_id"].isna().all())
            override = export_event_baseline_trialwise_csv(
                result, Path(tmp) / "override.csv", pairing_dict={"a": "b"},
                recording_metadata={"a": {"paired_subject_id": "c"}},
            )
            self.assertEqual(set(pd.read_csv(override).query("animal_id == 'a'").paired_subject_id), {"c"})
            for invalid in ({}, {"a": "a"}, {"a": "b", "b": "c"}, {"a": None}):
                with self.subTest(pairing_dict=invalid), self.assertRaises(ValueError):
                    export_event_baseline_trialwise_csv(result, Path(tmp) / "bad.csv", pairing_dict=invalid)
                self.assertFalse((Path(tmp) / "bad.csv").exists())

    def test_trialwise_pairing_normalizes_dotted_saved_subject_ids(self):
        a, b = recording("subject"), recording("recipient")
        a.subject, a.condition = "1.2", "subject with recipient"
        b.subject, b.condition = "1-3", "recipient"
        with patch("coop_ephys_analysis.lfp.baseline._window_power", side_effect=fake_power):
            result = calculate_event_baseline_power(
                SimpleNamespace(recordings=[a, b]), "poke", freq_range=(4, 8))
        with tempfile.TemporaryDirectory() as tmp:
            path = export_event_baseline_trialwise_csv(result, Path(tmp) / "pairs.csv",
                                                      pairing_dict={"1-2": "1.3"})
            ids = pd.read_csv(path)[["animal_id", "paired_subject_id"]].drop_duplicates()
            self.assertEqual(set(ids.itertuples(index=False, name=None)), {("1-2", "1-3"), ("1-3", "1-2")})
            with self.assertRaises(ValueError):
                export_event_baseline_trialwise_csv(result, Path(tmp) / "bad.csv",
                                                   pairing_dict={"1.2": "1-2"})

    def test_condition_comparison_and_variable_duration(self):
        a, b = recording("a"), recording("b")
        a.event_dict["other"] = [[9000, 9200]]
        collection = SimpleNamespace(recordings=[a, b])
        res = plot_baselined_event_average_traces(collection, "poke", condition_recordings={"A": ["a"], "B": ["b"]},
            selected_condition="A", comparison_condition="B", event_len=None,
            freq_range=(4, 8), plot=False)
        self.assertEqual(len(res["lines"]), 2)
        self.assertEqual(list(res["lines"][0]["event_counts"]), ["a"])
        self.assertEqual(list(res["lines"][1]["event_counts"]), ["b"])
        self.assertLess(res["lines"][0]["result"]["relative_time"][-1], .1)

    def test_validation_and_missing_recording(self):
        for kwargs in ({"mode": "coherence"}, {"baseline_window": (-1, -2)},
                       {"baseline_window": (-2, -1)}, {"pre_window": -.5},
                       {"freq_range": (12, 4)}, {"regions": ["missing"]},
                       {"baseline_ddof": 0}, {"min_baseline_estimates": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.calc(**kwargs)
        a, b = recording("a"), recording("b", {})
        result = calculate_event_baseline_power(SimpleNamespace(recordings=[a, b]), "poke", freq_range=(4, 8))
        self.assertIn("b", result["skipped_recordings"])
        with self.assertRaises(ValueError):
            plot_baselined_event_average_spectrogram(result, event="poke")


if __name__ == "__main__":
    unittest.main()
