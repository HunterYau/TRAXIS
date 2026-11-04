"""Small provenance and atomic-publication helpers."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value, *, overwrite: bool = False) -> None:
    data = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    atomic_write(path, lambda stream: stream.write(data), overwrite=overwrite)
