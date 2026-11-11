"""Small provenance and atomic-publication helpers."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Callable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _replace_with_retry(source: Path, destination: Path) -> None:
    # Windows readers/virus scanners can briefly deny replacement. Keep the
    # completed temporary file until replacement succeeds or the bound expires.
    delays = (0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.1, 0.1)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if (os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33)
                    or attempt == len(delays)):
                raise
            time.sleep(delays[attempt])


def atomic_write(path: Path, writer: Callable, *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            _replace_with_retry(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value, *, overwrite: bool = False) -> None:
    data = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    atomic_write(path, lambda stream: stream.write(data), overwrite=overwrite)
