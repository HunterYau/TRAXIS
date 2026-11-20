"""Look up moved historical outputs without editing scientific receipts."""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def organized_output(path):
    path = Path(path)
    if path.exists():
        return path
    root = PROJECT/'outputs'
    try:
        parts = path.resolve().relative_to(root).parts
    except ValueError:
        return path
    if not parts:
        return path
    try:
        manifest = json.loads((root/'organization.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        return path
    entry = manifest.get('moves',{}).get(parts[0])
    if entry:
        destination = (root/entry['destination']).resolve()
        destination.relative_to(root.resolve())
        return destination.joinpath(*parts[1:])
    return path
