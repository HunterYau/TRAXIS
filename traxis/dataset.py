"""Specimen-indexed, fingerprinted waveform datasets with bounded lazy reads."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from traxis.bank import WaveformBank
from traxis.resources import ResourceGuard
from traxis.storage import canonical_hash, sha256_file, write_json


@dataclass(frozen=True)
class DatasetEntry:
    path: str
    sha256: str
    specimen_id: str
    parent_specimen_id: str
    domain: str
    split: str
    observable: str
    validation_status: str

    @classmethod
    def from_bank(cls, bank: WaveformBank, path: str, digest: str):
        m = bank.metadata
        return cls(path, digest, m.specimen_id, m.parent_specimen_id, m.domain, m.split,
                   m.observable, m.validation_status)


def validate_entries(entries: tuple[DatasetEntry, ...]) -> None:
    if not entries:
        raise ValueError("dataset must contain at least one specimen")
    ids, parents = set(), {}
    for e in entries:
        if e.specimen_id in ids:
            raise ValueError("duplicate scan/specimen ID in dataset")
        ids.add(e.specimen_id)
        key = (e.domain, e.parent_specimen_id)
        if key in parents and parents[key] != e.split:
            raise ValueError(f"parent specimen {e.parent_specimen_id} crosses splits")
        parents[key] = e.split
        if len(e.sha256) != 64 or any(c not in "0123456789abcdef" for c in e.sha256):
            raise ValueError("invalid bank SHA-256 fingerprint")


class BankDataset:
    """Index all metadata; load one waveform bank at a time, with no RAM cache."""

    def __init__(self, entries: tuple[DatasetEntry, ...], root: Path):
        validate_entries(entries)
        self.entries, self.root = tuple(entries), Path(root).resolve()
        self.fingerprint = canonical_hash([asdict(e) for e in entries])

    @classmethod
    def build(cls, paths: list[Path], manifest_path: Path, guard: ResourceGuard | None = None):
        manifest_path = Path(manifest_path).resolve()
        root = manifest_path.parent
        entries = []
        for path in paths:
            if guard:
                guard.check()
            path = Path(path).resolve()
            if not path.is_relative_to(root):
                raise ValueError("bank paths must stay inside the dataset manifest directory")
            bank = WaveformBank.load(path)
            entries.append(DatasetEntry.from_bank(bank, path.relative_to(root).as_posix(), sha256_file(path)))
            del bank
        dataset = cls(tuple(sorted(entries, key=lambda e: e.specimen_id)), root)
        dataset.save(manifest_path)
        return dataset

    @classmethod
    def load(cls, manifest_path: Path):
        path = Path(manifest_path).resolve()
        if path.stat().st_size > 16 * 1024**2:
            raise MemoryError("dataset manifest exceeds 16 MiB")
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        if manifest["schema_version"] != 1:
            raise ValueError("unsupported dataset manifest schema")
        dataset = cls(tuple(DatasetEntry(**e) for e in manifest["entries"]), path.parent)
        if dataset.fingerprint != manifest["fingerprint"]:
            raise ValueError("dataset manifest fingerprint mismatch")
        return dataset

    def save(self, path: Path):
        if Path(path).resolve().parent != self.root:
            raise ValueError("save the manifest beside its relative bank paths")
        write_json(path, {"schema_version": 1, "entries": [asdict(e) for e in self.entries],
                          "fingerprint": self.fingerprint})

    def select(self, split: str) -> tuple[DatasetEntry, ...]:
        return tuple(e for e in self.entries if e.split == split)

    def read(self, entry: DatasetEntry) -> WaveformBank:
        if entry not in self.entries:
            raise ValueError("entry does not belong to this dataset")
        path = (self.root / entry.path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("bank path escapes dataset directory")
        if sha256_file(path) != entry.sha256:
            raise ValueError("waveform bank changed after dataset indexing")
        bank = WaveformBank.load(path)
        if DatasetEntry.from_bank(bank, entry.path, entry.sha256) != entry:
            raise ValueError("bank metadata no longer matches dataset index")
        return bank
