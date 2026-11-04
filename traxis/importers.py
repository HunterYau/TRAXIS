"""Import explicitly mapped scalar probe exports from an external solver/DAQ."""

import hashlib
import json
from pathlib import Path

import numpy as np

from traxis.acquisition import Action
from traxis.bank import BankMetadata, WaveformBank
from traxis.config import Plate, positive
from traxis.resources import ResourceGuard


def import_trace_bundle(manifest_path: Path, guard: ResourceGuard | None = None,
                         max_input_mib: float = 128) -> WaveformBank:
    """Read a JSON manifest plus one two-column CSV per mapped scalar trace.

    Export columns must be `time_s,value`, in the declared signal units. No
    unrecorded unit conversion, time alignment, baseline fitting or resampling
    is performed. Vector probes/electrode voltage must first be selected in
    the external solver/exporter. Labels are a separate measured/geometry NPY.
    """
    manifest_path = Path(manifest_path).resolve()
    positive("input byte limit", max_input_mib)
    if manifest_path.stat().st_size > 1024**2:
        raise MemoryError("trace manifest exceeds 1 MiB")
    spec = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if spec["schema_version"] != 1:
        raise ValueError("unsupported trace bundle schema")
    base = manifest_path.parent
    consumed, hashes, loaded = 0, {}, {}

    def local_file(relative):
        nonlocal consumed
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            raise ValueError("bundle paths must stay within the manifest directory")
        if path not in hashes:
            consumed += path.stat().st_size
            if consumed > max_input_mib * 1024**2:
                raise MemoryError("trace bundle exceeds input byte limit")
            hashes[str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def trace(relative):
        path = local_file(relative)
        if path not in loaded:
            if guard:
                guard.check()
            with path.open(encoding="utf-8-sig") as stream:
                if stream.readline().strip() != "time_s,value":
                    raise ValueError("CSV header must be exactly time_s,value")
                array = np.loadtxt(stream, delimiter=",", ndmin=2)
            if array.shape[1] != 2 or len(array) < 3 or not np.isfinite(array).all():
                raise ValueError("trace must contain finite time,value samples")
            loaded[path] = array
        return loaded[path]

    records = spec["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= 10000:
        raise ValueError("records must be a nonempty bounded list")
    actions, waves, baselines, common_time = [], [], [], None
    for record in records:
        current, reference = trace(record["waveform_csv"]), trace(record["baseline_csv"])
        if common_time is None:
            common_time = current[:, 0]
        for array in (current, reference):
            if array.shape[0] != len(common_time) or not np.allclose(array[:, 0], common_time, rtol=0, atol=1e-14):
                raise ValueError("all imported traces and baselines must share the same time grid")
        actions.append(Action(**record["action"]))
        waves.append(current[:, 1])
        baselines.append(reference[:, 1])
    label = np.load(local_file(spec["depth_mm_npy"]), allow_pickle=False, mmap_mode="r")
    metadata = dict(spec["metadata"])
    metadata["settings"] = {**metadata["settings"], "import_source_sha256": hashes,
                            "import_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    bank = WaveformBank(BankMetadata(**metadata), Plate(**spec["plate"]), spec["sensors_xy_m"],
                        tuple(actions), common_time, waves, baselines, label)
    if guard:
        guard.check(force=True)
    return bank
