"""Read-only cached RGB/PPG helpers, independent of preprocessing loaders.

The file lists must already be resolved by ``get_config``. CSV input paths use
the existing loaders' semantics: absolute paths are preserved and relative
paths are relative to the process working directory, not to the CSV folder.
No model construction, downloads, face detection, or cache writes occur here.
"""

import csv
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


_INPUT_NAME = re.compile(r"^(?P<recording>.+)_input(?P<clip>\d+)\.npy$")
_MMPD_RECORDING = re.compile(
    r"^subject(?P<subject>[1-9]\d*)"
    r"_L(?P<L>\d+)_MO(?P<MO>\d+)_E(?P<E>\d+)_S(?P<S>\d+)"
    r"_GE(?P<GE>\d+)_GL(?P<GL>\d+)_H(?P<H>\d+)_MA(?P<MA>\d+)$"
)
_MMPD_FIELDS = {
    "L": "LIGHT", "MO": "MOTION", "E": "EXERCISE", "S": "SKIN_COLOR",
    "GE": "GENDER", "GL": "GLASSER", "H": "HAIR_COVER", "MA": "MAKEUP",
}


def _get(node, name):
    return node[name] if isinstance(node, Mapping) else getattr(node, name)


def _natural_key(value):
    # Tagged components avoid mixing int and str comparisons.
    return tuple((1, int(piece)) if piece.isdigit() else (0, piece)
                 for piece in re.split(r"(\d+)", value))


def _mmpd_subject(recording):
    match = _MMPD_RECORDING.fullmatch(recording)
    if match is None:
        raise ValueError(f"Cannot identify MMPD subject/conditions from recording: {recording!r}")
    return "subject" + str(int(match["subject"])), match


def _pure_subject(recording):
    # PURELoader removes '-' then converts to int, e.g. 01-01 -> 101.
    if not re.fullmatch(r"\d{3,4}", recording):
        raise ValueError(f"Cannot identify PURE subject/trial from recording: {recording!r}")
    subject_trial = recording.zfill(4)
    if int(subject_trial[:2]) < 1 or int(subject_trial[2:]) < 1:
        raise ValueError(f"Invalid PURE subject/trial recording: {recording!r}")
    return "subject" + subject_trial[:2]


def read_cache_records(config, split="valid"):
    """Return deterministic recording groups from a resolved split file list.

    Each result is ``{'recording': str, 'subject': str, 'clips': [...]}``;
    ``clips`` contains numeric-sorted ``(clip_id, input_Path, label_Path)``
    tuples. MMPD membership checks reproduce all eight ``INFO`` filters used
    by MMPDLoader. PURE subject/trial IDs are also supported. Unknown datasets,
    malformed names and duplicate logical clips fail instead of being guessed.
    Input/label array existence and contents are checked when loading a clip.
    """
    split_name = split.upper()
    if split_name not in ("TRAIN", "VALID", "TEST"):
        raise ValueError("split must be train, valid, or test.")
    data = _get(_get(config, split_name), "DATA")
    dataset = str(_get(data, "DATASET"))
    if dataset not in ("MMPD", "PURE"):
        raise ValueError(f"Audit cache reader supports MMPD and PURE, received {dataset!r}.")
    if str(_get(data, "DATA_FORMAT")) != "NDCHW":
        raise ValueError("Regional audit expects DATA_FORMAT: NDCHW after loading HWC cache arrays.")
    file_list = Path(_get(data, "FILE_LIST_PATH")).expanduser()
    if not file_list.is_file():
        raise FileNotFoundError(f"Resolved {split_name} file list does not exist: {file_list}")
    grouped, seen = {}, set()
    with file_list.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "input_files" not in reader.fieldnames:
            raise ValueError(f"File list needs an input_files column: {file_list}")
        for row_number, row in enumerate(reader, start=2):
            value = row.get("input_files")
            if value is None or not value.strip():
                raise ValueError(f"Empty input_files entry at {file_list}:{row_number}")
            input_path = Path(value.strip()).expanduser()
            # Parse either slash convention; actual filesystem paths themselves
            # must remain valid on the execution host.
            name = re.split(r"[\\/]", value.strip())[-1]
            match = _INPUT_NAME.fullmatch(name)
            if match is None:
                raise ValueError(f"Malformed cache input filename at row {row_number}: {name!r}")
            recording, clip_id = match["recording"], int(match["clip"])
            if dataset == "MMPD":
                subject, condition = _mmpd_subject(recording)
                info = _get(data, "INFO")
                keep = all(int(condition[token]) in _get(info, field)
                           for token, field in _MMPD_FIELDS.items())
            else:
                subject, keep = _pure_subject(recording), True
            logical_clip = (recording, clip_id)
            if logical_clip in seen:
                raise ValueError(f"Duplicate recording/clip in file list: {recording}, clip {clip_id}")
            seen.add(logical_clip)
            if not keep:
                continue
            label_name = f"{recording}_label{match['clip']}.npy"
            # Replace the final filename only, never 'input' within directories.
            label_path = input_path.with_name(label_name)
            record = grouped.setdefault(recording, {
                "recording": recording, "subject": subject, "clips": [],
            })
            record["clips"].append((clip_id, input_path, label_path))
    if not grouped:
        raise ValueError(f"No {dataset} cached clips remain for {split_name} after filtering.")
    result = [grouped[name] for name in sorted(grouped, key=_natural_key)]
    for record in result:
        record["clips"].sort(key=lambda clip: clip[0])
    return result


def assert_subject_disjoint(train_records, val_records):
    """Reject overlapping MMPD subjects, independently checking recording IDs.

    This audit deliberately requires MMPD subject/condition names. It cannot
    certify subject splits in an unknown dataset by trusting arbitrary labels.
    Returns None when both nonempty subject sets are disjoint.
    """
    def subjects(records, label):
        identifiers = set()
        for record in records:
            identifier, _ = _mmpd_subject(record["recording"])
            if record.get("subject") != identifier:
                raise ValueError(f"{label} subject metadata disagrees with recording {record['recording']!r}.")
            identifiers.add(identifier)
        if not identifiers:
            raise ValueError(f"Cannot audit an empty {label} subject set.")
        return identifiers

    overlap = subjects(train_records, "training") & subjects(val_records, "validation")
    if overlap:
        names = ", ".join(sorted(overlap, key=_natural_key))
        raise ValueError(f"Training/validation subject overlap: {names}")


def contiguous_segments(record):
    """Return numeric-sorted clip lists, starting a new segment at every gap.

    Joining different segments would invent adjacent frames. Clip IDs need not
    start at zero. Duplicate IDs fail even for a manually assembled record.
    """
    clips = sorted(record["clips"], key=lambda clip: clip[0])
    segments = []
    previous = None
    for clip in clips:
        index = clip[0]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("Clip IDs must be nonnegative integers.")
        if previous is not None and index == previous:
            raise ValueError(f"Duplicate clip {index} for recording {record.get('recording', '')!r}.")
        if previous is None or index != previous + 1:
            segments.append([])
        segments[-1].append(clip)
        previous = index
    return segments


def load_cached_clip(input_path, label_path):
    """Read HWC cache arrays and return float32 [T,3,H,W] RGB and [T] PPG.

    At least three rows/columns are required for the default 3x3 grid. Shape,
    numeric type, finiteness, RGB scale and label length are checked strictly.
    Float cache values entirely within [0,1] are rejected as ambiguous with
    normalized input; uint8 black controls remain valid raw RGB.
    """
    video = np.load(Path(input_path), allow_pickle=False)
    target = np.load(Path(label_path), allow_pickle=False)
    if (video.ndim != 4 or video.shape[-1] != 3 or video.shape[0] < 1
            or min(video.shape[1:3]) < 3):
        raise ValueError(f"Expected raw cached RGB [T,H,W,3] with H,W>=3, received {video.shape}.")
    if target.ndim != 1 or target.shape[0] != video.shape[0]:
        raise ValueError(f"Expected target [{video.shape[0]}], received {target.shape}.")
    for name, array in (("RGB", video), ("PPG", target)):
        if (not np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.complexfloating)):
            raise ValueError(f"{name} cache must contain real numeric values.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} cache contains NaN/Inf.")
    low, high = float(video.min()), float(video.max())
    if low < 0 or high > 255 or (np.issubdtype(video.dtype, np.floating) and high <= 1):
        raise ValueError(f"Expected raw RGB [0,255], received range [{low:g},{high:g}].")
    video = np.ascontiguousarray(video.transpose(0, 3, 1, 2), dtype=np.float32)
    target = np.ascontiguousarray(target, dtype=np.float32)
    if not np.isfinite(video).all() or not np.isfinite(target).all():
        raise ValueError("Cache values overflow float32 after conversion.")
    return torch.from_numpy(video), torch.from_numpy(target)


def extract_rgb_traces(video, grid_size=3):
    """Return raw-scale global [T,3] and grid [R,T,3] original-pixel means.

    Grid cells use the model's nonoverlapping integer boundaries, in row-major
    order. No image resizing or [0,255]-to-[0,1] conversion is applied.
    """
    if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size < 1:
        raise ValueError("grid_size must be a positive integer.")
    if (not isinstance(video, torch.Tensor) or video.ndim != 4 or video.shape[1] != 3
            or video.shape[0] < 1 or min(video.shape[-2:]) < grid_size):
        raise ValueError("Expected nonempty RGB [T,3,H,W] covering the requested grid.")
    if video.is_complex() or not bool(torch.isfinite(video).all()):
        raise ValueError("RGB trace input must contain finite real values.")
    if not video.is_floating_point():
        video = video.float()
    if float(video.min()) < 0 or float(video.max()) > 255:
        raise ValueError("RGB trace input must be in raw [0,255] units.")
    height, width = video.shape[-2:]
    regions = []
    for row in range(grid_size):
        top, bottom = row * height // grid_size, (row + 1) * height // grid_size
        for col in range(grid_size):
            left, right = col * width // grid_size, (col + 1) * width // grid_size
            regions.append(video[:, :, top:bottom, left:right].mean(dim=(-1, -2)))
    return video.mean(dim=(-1, -2)), torch.stack(regions, dim=0)
