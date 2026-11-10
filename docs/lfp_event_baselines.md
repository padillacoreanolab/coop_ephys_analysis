# Event-specific power baselines

Start with `notebooks/lfp/event_baseline_analysis.ipynb`. It follows the existing
time-based analysis notebook and produces power spectrograms, comparison lines,
and one trialwise CSV per recording with frequency columns. The old notebooks/functions
are unchanged.

## Public functions

Import these from `coop_ephys_analysis.lfp`:

- `load_event_baseline_collection(json_path)`: load saved RMS traces and metadata
  without the larger coherence/Granger arrays. Import processed events afterward
  with `update_collection_events_from_behavior(..., source_unit="seconds")`.
  An existing full `LFPCollection` also works with the calculation functions.
- `calculate_event_baseline_power(collection, event, ...)`: calculate and retain
  individual trials, baselines, QC, and summaries without creating figures.
- `plot_baselined_event_average_spectrogram(collection, event, ...)`: calculate
  and plot regional heatmaps. Alternatively pass a prepared calculation result
  as the first argument, with no event/calculation arguments, to reuse it.
- `plot_baselined_event_average_traces(collection, event, ...)`: calculate and
  plot recording-weighted means/SEM. Supports `comparison_event`,
  `comparison_condition`, and `labels`, as in the existing trace helper.
  Alternatively pass a prepared result and optional `comparison_result`.
- `export_event_baseline_trialwise_csv(results, path, day=None)`: write CSVs
  matching the trialwise reference layout. Use `split_by_recording=True` with an
  output directory to get one file per recording (the notebook default). Pass a calculation result, a trace
  comparison, or the notebook's `baseline_results` selection dictionary.
- `export_event_baseline_csv(result, output_dir)`: optional detailed export of all numerical tables and
  settings. Accepts a calculation result, a spectrogram result, or a comparison
  trace result. Existing files are protected unless `overwrite=True`.

```python
common = dict(
    condition_recordings=condition_dict,
    selected_condition="subject with recipient",
    event_len=3.5, pre_window=3.5, post_window=0,
    freq_range=(4, 12), regions=["mPFC", "BLA"],
    baseline_window=(-3, -1), min_baseline_estimates=3,
)
selfish = calculate_event_baseline_power(
    collection, "filtered selfish nose poke", **common,
)
coop = calculate_event_baseline_power(
    collection, "filtered coop nose poke", **common,
)
heatmap = plot_baselined_event_average_spectrogram(selfish, color_limit=4, cmap="PuOr")
lines = plot_baselined_event_average_traces(
    selfish, comparison_result=coop, labels=("Selfish", "Cooperative"),
)
paths = export_event_baseline_trialwise_csv(
    {"selfish_social": selfish, "coop_social": coop},
    "results/lfp/day08/recordings", day=8, split_by_recording=True,
)
```

## Numerical conventions

The notebook uses `SPECTROGRAM_CMAP = "PuOr"`: negative z-scores are orange,
zero is near white, and positive z-scores are purple. The plotting helper also
accepts other Matplotlib colormaps (its default remains `RdBu_r`). Use a
diverging palette and the same symmetric `COLOR_LIMIT` across comparisons.
These settings only change the display, not the calculations or CSV values.

- Each occurrence uses its own baseline mean and sample SD (`ddof=1`) at each
  frequency and region: `(power - baseline_mean) / baseline_sd`.
- Power is recomputed from the saved `rms_traces` using the recording's sampling
  rate, spectral window duration, step, and time-half-bandwidth product. Taper
  averaging never averages together separate trials or spectral windows.
- Window centers, not window starts, label the new figures. Centers begin at
  `-pre_window` and stop before `event_len + post_window`. An extra half-window
  of signal on both sides supports those estimates. `event_len=None` uses each
  event's duration; later points may have fewer contributing trials.
- Event onsets are rounded to the nearest LFP sample. The original onset,
  rounding error, and actual relative window boundaries are retained.
- With a one-second window, 0.5-second step, and baseline `(-3, -1)`, the three
  baseline windows are `[-3,-2)`, `[-2.5,-1.5)`, and `[-2,-1)`. Their overlap
  means they are correlated; three estimates do not ensure a stable SD.
- All planned baseline estimates must be finite and at least
  `min_baseline_estimates` must exist. SD must be positive and finite. Invalid
  region/frequency baselines yield NaN z-scores, not zero. No epsilon or
  interpolation is used. Raw extractable power is still retained.
- An incomplete baseline or display epoch excludes that occurrence. Original
  event indices are preserved in QC. Behavior overlaps are flagged only.
- Means first average normalized trials within each recording. Traces then
  average frequency bins within recordings. Recordings have equal group weight;
  SEM uses recording sample SD / sqrt(n), and is NaN for fewer than two records.
  This is not automatically animal-level aggregation across multiple days.
- Only power is supported. Original `recording.power`, traces, and events are
  unchanged. RMS-normalized power is not absolute voltage power.

## Trialwise CSVs by recording

The notebook exports one CSV per recording, combining its selected events and conditions. Each
row represents a selection / recording / event occurrence / time center / region.
The first 12 metadata columns retain the relevant fields from the reference file: `day`, `animal_id`,
`role`, `paired_subject_id`, `h5_file`, `rec_file`, `choice`, `trial_number`,
`alignment_time_s`, `relative_time_s`, `metric`, `feature`.

Additional columns retain selection, condition, event, stable trial ID, z-score
normalization, baseline bounds, sample-SD convention, and time reference.
Frequency columns (e.g. `freq_001_4_Hz`) contain unchanged per-event z-scores.
The ordinal counts exported bins starting at 1; the Hz suffix gives the actual
frequency. Only calculated frequencies/regions are exported. Combined selections
must use the same frequency grid. Missing cells and entirely missing rows are
preserved; floats use 17 significant digits. No auxiliary files are written.
`split_by_recording=True` returns a recording-to-CSV-path dictionary and checks
all filenames for conflicts before writing. Omit that option and supply a `.csv`
path to retain the single combined file behavior.

`trial_number` is the original event index + 1 within each recording/event, not
a matched trial number across animals or choices. `alignment_time_s` is the
imported event onset in seconds, and relative times remain spectral window
centers. `alignment_source` describes how the alignment time was obtained. It is
redundant here because all events use the imported onset, so it is omitted by
default. Use `include_alignment_source=True` to retain it when needed (e.g. mixed
alignment methods). The exporter does not introduce recipient light-delay alignment.
The power-only pipeline labels `metric=power` and `feature=region`.
`source_region`, `target_region`, and `direction` are omitted. It does not add connectivity
metrics merely because those appear in the reference file.

Day comes from `settings['inputs']['day']` or the `day` argument. Known subject
conditions map to `role=subject`, and recipients to `role=recipient`; the full
condition is also retained. Unknown roles and source H5 files stay blank.
Supply `pairing_dict={subject_id: recipient_id, ...}` to populate the partner ID
for both animals; reverse entries are added automatically. Conflicting pairs
and self-pairs are rejected. Alone recordings have no partner. With a dictionary
supplied, a missing partner for a known social condition raises an error before
any CSV is written. The notebook passes its editable `PAIRING_DICT`. Numeric IDs use a consistent
hyphenated form (`1.2` becomes `1-2`) in both animal and partner columns.
The confirmed subject-to-recipient pairs are `1-2: 1-3`, `2-4: 2-1`,
`4-4: 4-2`, and `4-3: 4-1`.
Per-recording metadata can override the mapping with
`recording_metadata={recording_name: {'paired_subject_id': '1-2'}}` (also accepts
`role` and `alignment_source`). Pairings are never inferred from filename order
or from the unordered subject/recipient sets.

Reconstruct heatmaps by averaging trial z-scores within recording at each
frequency/time/region, then averaging recordings equally. To reconstruct lines,
average the recording-level frequency means across frequencies, then compute
mean and SEM across recordings. Preserve that order when data are missing.
The CSV retains the individual values needed for both calculations.

## In-memory tables and optional detailed export

Each result has a `tables` dictionary with pandas DataFrames:

| Table | Observation represented by a row |
|---|---|
| `trial_power` | Trial / region / frequency / plot time: raw power and z-score |
| `baseline_samples` | Trial / region / frequency / baseline window: raw power |
| `baseline_statistics` | Trial / region / frequency: mean, SD, counts, validity |
| `event_qc` | Original event occurrence: timing, overlap, validity summary |
| `recording_spectrograms` | Recording / region / frequency / time: mean and trial counts |
| `group_spectrograms` | Region / frequency / time: exact heatmap mean, SEM, counts |
| `trial_traces` | Trial / region / time: mean across selected frequencies |
| `recording_traces` | Recording / region / time: trace entering the group summary |
| `group_traces` | Region / time: exact plotted line mean, SEM, record count |

All trial tables share a `trial_id` formed from recording, event name, and original
event index. The optional `export_event_baseline_csv` helper exports all nine
tables and adds comparison line/label columns. NaNs become empty
fields, and floats use 17 significant digits. `analysis_settings.json` records
parameters, source paths when available, counts, and skipped recordings.

The order of averaging with missing data matters: reconstruct plotted lines
from `recording_traces`, not from pooled `trial_traces`. For independent later
inference, retain the animal/session hierarchy; these functions do not perform
significance tests.
