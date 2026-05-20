"""Utilities for extracting ECU/Trodes behavior events."""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from external.diff_fam_social_memory_ephys.trodes import read_exported as tre


DEFAULT_BOX_TO_ECU = {
    1: {
        "dio_ECU_Din1": "selfish light",
        "dio_ECU_Din2": "coop light",
        "dio_ECU_Din6": "selfish nose poke",
        "dio_ECU_Din10": "coop nose poke",
        "dio_ECU_Din8": "subject port entry",
        "dio_ECU_Din16": "recipient port entry",
    },
    2: {
        "dio_ECU_Din3": "selfish light",
        "dio_ECU_Din4": "coop light",
        "dio_ECU_Din22": "selfish nose poke",
        "dio_ECU_Din26": "coop nose poke",
        "dio_ECU_Din24": "subject port entry",
        "dio_ECU_Din32": "recipient port entry",
    },
}

DEFAULT_ALLOWED_BAD_RECORDINGS = {
    "20250516_111920_Stage4_D7_2-1_merged",
    "20250516_111920_Stage4_D7_2-4_merged",
}


def save_pickle(obj, path):
    """
    Save a Python object to a pickle file.

    Parameters:
        obj (object): Python object to save.
        path (str or Path): Output path ending in .pkl or another pickle filename.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(path):
    """
    Load a Python object from a pickle file.

    Parameters:
        path (str or Path): Path to an existing pickle file.

    Returns:
        object: The Python object stored in the pickle file.
    """
    with Path(path).open("rb") as file:
        return pickle.load(file)


def load_recording_box_map(csv_path):
    """
    Read a CSV that maps recording names to ECU box numbers.

    Parameters:
        csv_path (str or Path): Path to the box organization CSV. The CSV must contain
            "Individual name" and "Box" columns.

    Returns:
        dict: Mapping from merged recording name to box number.
    """
    df = pd.read_csv(csv_path)
    required_columns = {"Individual name", "Box"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Box map CSV is missing required columns: {missing}")

    df = df.dropna(subset=["Individual name", "Box"])
    names = df["Individual name"].astype(str).str.strip()
    boxes = df["Box"].astype(int)
    return {f"{name}_merged": int(box) for name, box in zip(names, boxes)}


def trodes_data_to_events(
    recording_name: str,
    data,
    clockrate,
    first_timestamp,
    allowed_bad_recordings=None,
):
    """
    Convert Trodes DIO transition rows into behavior start/stop times.

    Parameters:
        recording_name (str): Name of the recording being processed.
        data (array-like): Trodes DIO data rows containing timestamp and status values.
        clockrate (int or float): Trodes clock rate used to convert timestamps to seconds.
        first_timestamp (int): First timestamp of the recording, used to align events to zero.
        allowed_bad_recordings (set): Recording names allowed to bypass transition-order checks.

    Returns:
        list: List of [start, stop] behavior events in seconds.
    """
    if allowed_bad_recordings is None:
        allowed_bad_recordings = DEFAULT_ALLOWED_BAD_RECORDINGS

    events = []
    event = []
    expected_status = 1
    clockrate = float(clockrate)
    first_timestamp = int(first_timestamp)

    for time_value in list(data)[1:]:
        timestamp, value = time_value
        status = int(_as_scalar(value))
        if status != expected_status and recording_name not in allowed_bad_recordings:
            raise ValueError(
                "Invalid timestamp - the event has not started/stopped"
                f" - {recording_name}"
            )

        timestamp = int(_as_scalar(timestamp))
        event.append((timestamp - first_timestamp) / clockrate)
        if expected_status == 0:
            events.append(event)
            event = []
        expected_status = 1 - expected_status

    return np.array(events, dtype=float).tolist()


def load_trodes_recording(
    dio_path,
    recording_to_box: dict,
    box_to_ecu: dict | None = None,
    allowed_bad_recordings=None,
):
    """
    Extract behavior events from one Trodes .DIO directory.

    Parameters:
        dio_path (str or Path): Path to a recording .DIO directory.
        recording_to_box (dict): Mapping from recording name to ECU box number.
        box_to_ecu (dict): Mapping from box number to DIO channel behavior names.
        allowed_bad_recordings (set): Recording names allowed to bypass transition-order checks.

    Returns:
        tuple: Recording name and a dictionary mapping behavior names to [start, stop] events.
    """
    dio_path = Path(dio_path)
    recording_name = dio_path.stem
    box_to_ecu = box_to_ecu or DEFAULT_BOX_TO_ECU

    try:
        box = int(recording_to_box[recording_name])
    except KeyError as exc:
        raise KeyError(f"No box mapping found for recording: {recording_name}") from exc

    behaviors = {}
    for dat_path in sorted(dio_path.rglob("*.dat")):
        din = _extract_din_name(dat_path.name)
        if din not in box_to_ecu[box]:
            continue

        trodes_data = tre.read_trodes_extracted_data_file(dat_path)
        behavior_name = box_to_ecu[box][din]
        behaviors[behavior_name] = trodes_data_to_events(
            recording_name=recording_name,
            data=trodes_data["data"],
            clockrate=trodes_data["clockrate"],
            first_timestamp=trodes_data["first_timestamp"],
            allowed_bad_recordings=allowed_bad_recordings,
        )

    return recording_name, behaviors


def extract_from_folder(
    root_path,
    recording_to_box: dict,
    output_dir=None,
    save: bool = True,
    box_to_ecu: dict | None = None,
    allowed_bad_recordings=None,
):
    """
    Extract behavior events from all Stage folders inside a root folder.

    Expected folder architecture:
        root_path/
            Coop_ephys_Box_organization(Sheet1).csv
            Stage4_D1/
                20250508_100203_Stage4_D1_1-2_merged.DIO/
                    20250508_100203_Stage4_D1_1-2_merged.dio_ECU_Din1.dat
                    20250508_100203_Stage4_D1_1-2_merged.dio_ECU_Din10.dat
                    ...
                20250508_100203_Stage4_D1_1-3_merged.DIO/
                    ...
            Stage4_D2/
                ...

    Parameters:
        root_path (str or Path): Path containing Stage folders, such as Stage1_D1 or Stage4_D1.
        recording_to_box (dict): Mapping from recording name to ECU box number.
        output_dir (str or Path): Folder where daily pickle files should be saved.
        save (bool): If True, save one pickle file per Stage folder.
        box_to_ecu (dict): Mapping from box number to DIO channel behavior names.
        allowed_bad_recordings (set): Recording names allowed to bypass transition-order checks.

    Returns:
        dict: Nested dictionary of extracted behavior events grouped by Stage folder.
    """
    root_path = Path(root_path)
    output_dir = Path(output_dir) if output_dir is not None else root_path
    stage_behaviors = {}

    for day_path in sorted(root_path.iterdir()):
        if not day_path.is_dir() or not day_path.name.startswith("Stage"):
            continue

        daily_behaviors = {}
        daily_count = 0
        for dio_path in sorted(path for path in day_path.rglob("*") if path.is_dir()):
            if not dio_path.name.endswith(".DIO"):
                continue
            recording_name, behaviors = load_trodes_recording(
                dio_path=dio_path,
                recording_to_box=recording_to_box,
                box_to_ecu=box_to_ecu,
                allowed_bad_recordings=allowed_bad_recordings,
            )
            daily_behaviors[recording_name] = behaviors
            daily_count += 1

        stage_behaviors[day_path.name] = daily_behaviors
        print(f"In {day_path.name}, {daily_count} mice have been processed.")
        if save:
            save_pickle(daily_behaviors, output_dir / f"{day_path.name}.pkl")

    return stage_behaviors


def _as_scalar(value):
    """
    Convert a Trodes value or numpy value into a single scalar.

    Parameters:
        value (object): Scalar-like value, list, tuple, or numpy array.

    Returns:
        object: The first scalar value.
    """
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return array.flat[0]


def _extract_din_name(file_name: str) -> str:
    """
    Extract the DIO channel name from a Trodes .dat filename.

    Parameters:
        file_name (str): Trodes .dat filename containing a dio_* channel name.

    Returns:
        str: DIO channel name, such as "dio_ECU_Din10".
    """
    parts = file_name.split(".")
    for part in parts:
        if part.startswith("dio_"):
            return part
    raise ValueError(f"Could not determine DIO channel from file name: {file_name}")
